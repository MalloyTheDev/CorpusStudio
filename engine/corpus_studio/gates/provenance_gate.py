"""Per-row dataset **provenance** gate (data-safety).

Reads each row's declared *teacher* — the model/provider that GENERATED the row,
e.g. ``meta.teacher`` — and applies the provider policy per row, so a dataset can
be checked for **non-trainable-provenance** rows *before* it is used to train.
This is the licensing counterpart to ``provider-policy`` (which gates
generation-*time*: which provider you may generate/train *with*) and
``run-provenance`` (which fingerprints a whole run). Neither of those walks an
existing dataset and quarantines rows by who produced them — this does.

Each distinct teacher is bucketed:

* **quarantined** — the teacher maps to a *known* provider whose terms block
  training on its outputs (a frontier ``trainable_output_generator`` block, e.g.
  Anthropic / OpenAI). Training on these rows would distill a restricted model
  into yours — the exact "data availability ≠ data permission" trap.
* **pass** — the teacher is declared trainable-clean: a recognized local/open
  provider, or one the user allow-listed (e.g. an MIT-licensed open model).
* **unknown** — no teacher tag, or an unrecognized teacher whose license can't be
  determined → *quarantine-until-verified* (never assumed safe).

**Honesty boundary (read before trusting the verdict):** this trusts each row's
*self-declared* teacher. A row that omits or misstates its teacher is NOT caught
by content, an ``unknown`` teacher is treated as unsafe (not clean), and a
``pass`` reflects a declared/allow-listed license, not a proof of origin. It is a
licensing/provenance check over declared metadata — not a detector of where text
actually came from.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from corpus_studio.gates.models import GateStatus
from corpus_studio.providers.policy import ProviderRole, resolve_policy

PROVENANCE_ALLOWLIST_FILENAME = "provenance_allowlist.json"
DEFAULT_TEACHER_FIELD = "meta.teacher"
UNTAGGED_LABEL = "(untagged)"

# The role whose licensing terms this gate enforces (shown in the report header).
CHECKED_ROLE = ProviderRole.TRAINABLE_OUTPUT_GENERATOR.value

# Vendor prefixes → provider ids, so a bare model string (``claude-opus-4-8``,
# ``gpt-4o``) resolves to the provider whose policy the gate then applies. Only
# providers present in the policy table are *confidently* quarantined; anything
# else falls through to UNKNOWN (quarantine-until-verified), never a false PASS.
_MODEL_PREFIX_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("text-davinci", "openai"),
    ("davinci", "openai"),
    ("gemini", "google"),
    ("palm", "google"),
)
# Fully-qualified ``vendor/model`` prefixes (as used in ``meta.teacher`` tags).
_VENDOR_ALIASES: dict[str, str] = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "azure-openai": "openai",
    "google": "google",
    "gemini": "google",
    "palm": "google",
    "vertex": "google",
    "vertex-ai": "google",
    "vertexai": "google",
    "openrouter": "openrouter",
    "ollama": "ollama",
}


class TeacherStatus(str, Enum):
    """Per-teacher provenance verdict (distinct from the row-level gate status)."""

    PASS = "pass"  # trainable-clean provenance
    QUARANTINED = "quarantined"  # known-restricted provider — non-trainable
    UNKNOWN = "unknown"  # untagged / unrecognized — quarantine-until-verified


# Display order: most-severe first, so the human table leads with what blocks.
_TEACHER_STATUS_ORDER = {
    TeacherStatus.QUARANTINED: 0,
    TeacherStatus.UNKNOWN: 1,
    TeacherStatus.PASS: 2,
}


class TeacherProvenanceBucket(BaseModel):
    """One distinct teacher's roll-up: its resolved status, row count, and reason."""

    teacher: str  # the declared meta.teacher value, or the untagged label
    provider_id: str = ""  # the provider the teacher resolved to (best-effort)
    status: TeacherStatus
    row_count: int
    note: str = ""  # licensing / reason note (never a raw secret)


class ProvenanceGateReport(BaseModel):
    """Per-row provenance verdict for a dataset. Counts are row tallies, never a
    folded quality score; the verdict is a licensing/provenance judgment over the
    rows' *declared* teachers."""

    role: str = CHECKED_ROLE
    teacher_field: str = DEFAULT_TEACHER_FIELD
    target: str = ""
    generated_at: str | None = None
    total_rows: int = 0
    trainable_rows: int = 0  # PASS
    quarantined_rows: int = 0  # QUARANTINED
    unknown_rows: int = 0  # UNKNOWN
    strict: bool = False  # True ⇒ UNKNOWN rows also BLOCK
    overall_status: GateStatus = GateStatus.PASS
    summary: str = ""
    buckets: list[TeacherProvenanceBucket] = Field(default_factory=list)


def _extract_teacher(row: dict, teacher_field: str) -> str | None:
    """Read a dotted-path field (default ``meta.teacher``) as a trimmed string, or
    None when absent/blank. Non-string scalars are coerced; containers yield None."""
    current: object = row
    for part in teacher_field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if current is None or isinstance(current, (dict, list)):
        return None
    text = str(current).strip()
    return text or None


def _provider_for_teacher(teacher: str) -> str:
    """Best-effort provider id for a teacher model string. A ``vendor/model`` tag
    uses its vendor prefix; a bare model name uses a known prefix; otherwise the
    string is treated as its own provider id (→ unrecognized → UNKNOWN)."""
    normalized = teacher.strip().lower()
    if "/" in normalized:
        prefix = normalized.split("/", 1)[0]
        return _VENDOR_ALIASES.get(prefix, prefix)
    for prefix, provider in _MODEL_PREFIX_PROVIDERS:
        if normalized.startswith(prefix):
            return provider
    return _VENDOR_ALIASES.get(normalized, normalized)


def classify_teacher(
    teacher: str | None,
    allowlist: dict[str, str] | None = None,
) -> tuple[TeacherStatus, str, str]:
    """Classify one teacher into (status, provider_id, note).

    Order of precedence: untagged → UNKNOWN; user allow-listed → PASS; a
    recognized restricted provider → QUARANTINED; a recognized open/local provider
    → PASS; anything else → UNKNOWN (quarantine-until-verified — never a false PASS).
    """
    allowlist = allowlist or {}
    if teacher is None or not teacher.strip():
        return (
            TeacherStatus.UNKNOWN,
            "",
            "untagged — quarantine until the generating model is verified",
        )

    name = teacher.strip()
    provider_id = _provider_for_teacher(name)

    # An explicit user declaration (by exact teacher or by provider) always wins:
    # the allow-list is how an open/MIT teacher the gate can't recognize is cleared.
    for key in (name, name.lower(), provider_id):
        if key in allowlist:
            note = allowlist[key].strip() or "user-declared trainable-clean"
            return TeacherStatus.PASS, provider_id, note

    # Resolve the provider policy WITHOUT project overrides: the provenance verdict
    # is about the teacher's licensing terms, not this project's generation approval
    # (and resolve_policy re-asserts the frontier block regardless of overrides).
    policy = resolve_policy(provider_id, model_id=name)
    blocked = ProviderRole.TRAINABLE_OUTPUT_GENERATOR in policy.blocked_roles
    recognized = policy.default_policy_source != "fallback"

    if blocked and recognized:
        note = policy.license_or_terms_note or (
            "provider terms restrict using outputs to train competing models"
        )
        return TeacherStatus.QUARANTINED, provider_id, note
    if recognized and not blocked:
        note = policy.license_or_terms_note or "recognized local/open provider — trainable"
        return TeacherStatus.PASS, provider_id, note
    return (
        TeacherStatus.UNKNOWN,
        provider_id,
        "unrecognized teacher — quarantine until its license is verified "
        "(allow-list it if you know it is trainable-clean)",
    )


def _verdict(quarantined: int, unknown: int, total: int, strict: bool) -> tuple[GateStatus, str]:
    """Map the tallies to a gate status + human summary. BLOCK on any quarantined
    row (and, under strict, any unknown row); WARN when unknown rows remain."""
    if quarantined > 0:
        extra = f" and {unknown} unknown-provenance row(s)" if (strict and unknown) else ""
        return (
            GateStatus.BLOCK,
            f"BLOCK — {quarantined} row(s) from restricted teacher(s){extra} must be removed "
            "or explicitly authorized before training (data availability ≠ data permission).",
        )
    if unknown > 0 and strict:
        return (
            GateStatus.BLOCK,
            f"BLOCK (strict) — {unknown} row(s) have unknown provenance; verify or allow-list "
            "their teacher before training.",
        )
    if unknown > 0:
        return (
            GateStatus.WARN,
            f"PASS with warnings — all teachers are trainable-clean, but {unknown} row(s) have "
            "unknown provenance (quarantine-until-verified). Re-run with --strict to block them.",
        )
    return (
        GateStatus.PASS,
        f"PASS — all {total} row(s) have trainable-clean provenance.",
    )


def run_provenance_gate(
    rows: list[dict],
    *,
    teacher_field: str = DEFAULT_TEACHER_FIELD,
    allowlist: dict[str, str] | None = None,
    strict: bool = False,
    target: str = "",
    generated_at: str | None = None,
) -> ProvenanceGateReport:
    """Pure per-row provenance gate. Buckets rows by declared teacher, classifies
    each teacher, and returns the roll-up + BLOCK/WARN/PASS verdict. Non-dict rows
    count as untagged (UNKNOWN)."""
    allowlist = allowlist or {}
    # Aggregate rows per teacher label (untagged rows share one bucket).
    counts: dict[str, int] = {}
    labels: dict[str, str] = {}  # label → the raw teacher (or None marker)
    for row in rows:
        teacher = _extract_teacher(row, teacher_field) if isinstance(row, dict) else None
        label = teacher if teacher else UNTAGGED_LABEL
        counts[label] = counts.get(label, 0) + 1
        labels.setdefault(label, teacher or "")

    buckets: list[TeacherProvenanceBucket] = []
    trainable = quarantined = unknown = 0
    for label, count in counts.items():
        raw_teacher = labels[label] or None
        status, provider_id, note = classify_teacher(raw_teacher, allowlist)
        if status == TeacherStatus.PASS:
            trainable += count
        elif status == TeacherStatus.QUARANTINED:
            quarantined += count
        else:
            unknown += count
        buckets.append(
            TeacherProvenanceBucket(
                teacher=label,
                provider_id=provider_id,
                status=status,
                row_count=count,
                note=note,
            )
        )

    # Most-severe teacher first, then by row count desc, then name — stable + readable.
    buckets.sort(key=lambda b: (_TEACHER_STATUS_ORDER[b.status], -b.row_count, b.teacher))

    total = len(rows)
    overall, summary = _verdict(quarantined, unknown, total, strict)
    return ProvenanceGateReport(
        teacher_field=teacher_field,
        target=target,
        generated_at=generated_at,
        total_rows=total,
        trainable_rows=trainable,
        quarantined_rows=quarantined,
        unknown_rows=unknown,
        strict=strict,
        overall_status=overall,
        summary=summary,
        buckets=buckets,
    )


_STATUS_MARK = {
    TeacherStatus.QUARANTINED: "QUARANTINED",
    TeacherStatus.UNKNOWN: "UNKNOWN",
    TeacherStatus.PASS: "PASS",
}


def render_provenance_gate_text(report: ProvenanceGateReport) -> str:
    """Human-readable table + verdict (for the CLI's stderr / a screenshot). The
    machine-readable verdict lives in the JSON report; this mirrors it for people."""
    lines = [
        f"Provenance gate — {report.role} ({report.total_rows} row(s), teacher={report.teacher_field})",
    ]
    for bucket in report.buckets:
        mark = _STATUS_MARK[bucket.status]
        lines.append(
            f"  {mark:<12} {bucket.row_count:>6}  {bucket.teacher}"
            + (f"  ({bucket.note})" if bucket.note else "")
        )
    lines.append(f"VERDICT: {report.summary}")
    return "\n".join(lines)


def provenance_allowlist_path(project_dir: Path | str) -> Path:
    return Path(project_dir) / PROVENANCE_ALLOWLIST_FILENAME


def load_provenance_allowlist(project_dir: Path | str) -> dict[str, str]:
    """Load the project's teacher allow-list (``{teacher-or-provider: license note}``),
    or an empty mapping when absent/unreadable. Fails closed: a non-dict file or a
    non-string value is dropped, so a hand-edited file can never crash the gate or
    silently turn a note into an approval. Values are coerced to strings (the note)."""
    path = provenance_allowlist_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and not isinstance(value, (dict, list)):
            result[key] = "" if value is None else str(value)
    return result


def save_provenance_allowlist(project_dir: Path | str, allowlist: dict[str, str]) -> Path:
    """Write the allow-list atomically (temp file + replace) to avoid partial writes."""
    path = provenance_allowlist_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(allowlist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
