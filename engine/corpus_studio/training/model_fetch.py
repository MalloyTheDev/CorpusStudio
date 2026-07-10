"""Reliable base-model download from the Hugging Face Hub (opt-in ``[train]`` extra).

Grabbing a base model to train on is different from the dependency-light *dataset* import: it is many
large weight files where a dropped connection must **resume**, not restart. So this lazy-imports
``huggingface_hub.snapshot_download`` (pulled in by the ``[train]`` extra) — its built-in resume +
retries survive the flaky connections that break a raw download.

It also surfaces the model's **license** and warns when it is not clearly permissive — the base
model's license governs what you may do with the *trained* model, so "grab an MIT model" is the safe
default (the same data-availability-≠-permission discipline the provenance gate applies to datasets).
The license classification is honest: an unknown or missing license is treated as restricted, not safe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

FetchProgress = Callable[[str], None]
INSTALL_HINT = "pip install corpus-studio-engine[train]"

# A large model over a flaky link (e.g. Windows WinError 10054 resetting most attempts) needs several
# tries — each RESUMES where the last left off — before it completes. Bounded so a genuinely broken
# fetch still fails fast-ish.
_MAX_FETCH_ATTEMPTS = 5

# License ids (SPDX-ish, lowercased) that are safe to train on AND redistribute the result under.
_PERMISSIVE = frozenset(
    {
        "mit",
        "apache-2.0",
        "apache2.0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "cc-by-3.0",
        "isc",
        "unlicense",
        "zlib",
        "mpl-2.0",
    }
)


class ModelFetchResult(BaseModel):
    repo_id: str
    revision: str | None = None
    local_path: str = ""
    license: str | None = None
    license_permissive: bool = False
    weight_files: list[str] = Field(default_factory=list)  # .safetensors / .bin fetched
    total_size_mb: float = 0.0
    warnings: list[str] = Field(default_factory=list)


def classify_license(license_id: str | None) -> tuple[bool, str]:
    """(permissive, human note). A missing/unknown/non-commercial/custom license is NOT permissive —
    treated as restricted-until-verified, mirroring the provenance gate's honesty."""
    if not license_id or not str(license_id).strip():
        return (False, "no declared license — treat as all-rights-reserved until you verify the repo's terms")
    normalized = str(license_id).strip().lower()
    if normalized in _PERMISSIVE:
        return (True, f"{license_id}: permissive — OK to train on and redistribute the result")
    if "nc" in normalized.split("-") or "noncommercial" in normalized or "non-commercial" in normalized:
        return (False, f"{license_id}: NON-COMMERCIAL — you may not use the result commercially")
    if "llama" in normalized:
        return (False, f"{license_id}: Meta Llama custom license — check its acceptable-use policy before training")
    if "gemma" in normalized:
        return (False, f"{license_id}: Gemma custom license — review its terms before training")
    if "openrail" in normalized or "rail" in normalized:
        return (False, f"{license_id}: RAIL license — has behavioural use restrictions; review before use")
    if normalized in {"other", "unknown", "unlicensed"}:
        return (False, f"{license_id}: unspecified license — verify the repo's terms before training")
    return (False, f"{license_id}: not a recognized-permissive license — verify before training/redistribution")


def license_from_readme(model_dir: Path | str) -> str | None:
    """Read ``license:`` from a downloaded model card's YAML front-matter — LOCAL, no network (so it
    survives the flaky Hub metadata API). Returns None when there's no README or no license field."""
    readme = Path(model_dir) / "README.md"
    if not readme.exists():
        return None
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    for line in text[3:end].splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("license:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            return value or None
    return None


def _read_license(repo_id: str, revision: str | None, warnings: list[str]) -> str | None:
    """Best-effort read of the model's declared license from its Hub card metadata."""
    try:
        from huggingface_hub import model_info  # noqa: PLC0415 - lazy heavy import.

        info = model_info(repo_id, revision=revision)
        card = getattr(info, "card_data", None)
        if isinstance(card, dict):
            return card.get("license")
        license_value = getattr(card, "license", None) if card is not None else None
        if license_value:
            return str(license_value)
        # Fall back to a `license:xxx` tag.
        for tag in getattr(info, "tags", []) or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                return tag.split(":", 1)[1]
    except Exception as exc:  # noqa: BLE001 - never let a metadata read block the download.
        warnings.append(f"Could not read the model's license from the Hub: {exc}")
    return None


def fetch_model(
    repo_id: str,
    *,
    local_dir: Path | str | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    progress: FetchProgress | None = None,
) -> ModelFetchResult:
    """Download ``repo_id`` from the Hub (resumable) and report its license. With ``local_dir`` the
    files land there; otherwise they go to the HF cache (so ``train-run --base-model <repo_id>`` finds
    them offline). Raises :class:`RuntimeError` with an install hint if huggingface_hub is absent."""
    warnings: list[str] = []
    if progress is not None:
        progress(f"downloading {repo_id} (resumable)…")

    # hf_transfer (the Rust fast-downloader) does NOT survive flaky connections — on a Windows
    # WinError 10054 reset it aborts instead of resuming, so a big model never finishes. Default it
    # OFF so the pure-Python downloader (which resumes + retries) is used. Set BEFORE importing the
    # hub, which reads this into a module constant at import. A user on a solid link can still opt in
    # by exporting HF_HUB_ENABLE_HF_TRANSFER=1 themselves (setdefault respects an explicit choice).
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - lazy heavy import.
    except ImportError as exc:
        raise RuntimeError(
            f"huggingface_hub is not installed — it comes with the training runtime. {INSTALL_HINT}"
        ) from exc

    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    if local_dir:
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns:
        # Always keep the model card + config so the license (and model shape) are readable locally.
        patterns = list(allow_patterns)
        for keep in ("README.md", "*.json"):
            if keep not in patterns:
                patterns.append(keep)
        kwargs["allow_patterns"] = patterns

    # Resilient resume-loop: snapshot_download resumes, so on a connection that resets most attempts we
    # just try again (up to _MAX_FETCH_ATTEMPTS) and each try continues where the last left off. Only
    # after exhausting the attempts do we surface the failure (a clean RuntimeError → the CLI exits 2).
    path: Path | None = None
    last_error: Exception | None = None
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            if progress is not None and attempt > 1:
                progress(f"retry {attempt}/{_MAX_FETCH_ATTEMPTS} (resuming the partial download)…")
            path = Path(snapshot_download(**kwargs))
            break
        except Exception as exc:  # noqa: BLE001 - retry any transient network/hub error; resume continues.
            last_error = exc
    if path is None:
        raise RuntimeError(
            f"Failed to download {repo_id} after {_MAX_FETCH_ATTEMPTS} attempts: {last_error}. The "
            "partial download is cached — re-run to resume from where it stopped."
        ) from last_error

    # License: the downloaded card first (robust, no network), the Hub API only as a fallback.
    license_id = license_from_readme(path) or _read_license(repo_id, revision, warnings)
    permissive, note = classify_license(license_id)
    if not permissive:
        warnings.append(f"LICENSE: {note}")

    weight_files: list[str] = []
    total_bytes = 0
    has_safetensors = False
    has_pickle = False
    for file in path.rglob("*"):
        if file.is_file():
            total_bytes += file.stat().st_size
            if file.suffix in {".safetensors", ".bin", ".gguf"}:
                weight_files.append(file.name)
                has_safetensors = has_safetensors or file.suffix == ".safetensors"
                has_pickle = has_pickle or file.suffix == ".bin"
    if not weight_files:
        # A fetch that produced no weights is a FAILED fetch (the command promises "exit 2 on failure"),
        # not a success-with-warning — usually a dropped connection after the config/tokenizer landed.
        raise RuntimeError(
            f"No weight files (.safetensors/.bin/.gguf) were downloaded for {repo_id} — the fetch is "
            "incomplete (often a dropped connection after the config/tokenizer). The partial download "
            "is cached; re-run to resume."
        )
    if has_pickle and not has_safetensors:
        # SECURITY: PyTorch .bin weights are Python pickles — loading them can execute arbitrary code.
        warnings.append(
            "SECURITY: this model ships only pickle (.bin) weights, not safetensors — loading it can "
            "run arbitrary code. Prefer a model with .safetensors, or vet this repo before training."
        )

    return ModelFetchResult(
        repo_id=repo_id,
        revision=revision,
        local_path=str(path),
        license=license_id,
        license_permissive=permissive,
        weight_files=sorted(weight_files),
        total_size_mb=round(total_bytes / 1e6, 1),
        warnings=warnings,
    )
