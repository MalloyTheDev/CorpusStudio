"""Which quality signals a dataset's SHAPE supports, and how much of its content they read.

CorpusStudio's quality signals are TEXT signals: near-duplicate detection, low-information counts,
templated-pattern detection and token-length outliers all read prose. On a non-text record - an
object-detection corpus of normalized boxes, a numeric telemetry table, a provenance ledger - they
have nothing to read, and their silence is NOT evidence that the data is clean. Reporting a zero,
or a letter grade computed from those zeros, for a signal that could not run is false assurance.
This module makes the applicability explicit and measured:

* every LEAF field of a schema gets a ROLE from its declared ``FieldType``;
* ``assessed_content_share`` is the share of the dataset's tokens that live in free-text leaves,
  measured over the actual rows rather than assumed from the field count. The count ratio would be
  wrong in both directions: ``raw_text`` declares one prose field among five and ``image_caption``
  one among four, yet the text/caption carries nearly all of the content;
* each signal declares the roles it needs, so a signal with nothing to read is reported
  ``not_applicable`` instead of a pass.

Roles come from the SCHEMA, never from guessing at the values. ``FieldType`` already separates
``text`` / ``markdown`` / ``code`` / ``messages`` (prose) from ``string`` (a short scalar: an id, a
label, a hash, a URL, an enum member), and every builtin schema uses the prose types for its
substance and ``string`` for identifiers. The values cannot be read that way: a license statement
and a URL tokenize like a short sentence while a caption can be shorter than both, so inferring
prose from the data reproduces exactly the guesswork that produced the false assurance. With no
schema the applicability is reported ``unmeasured`` - stated plainly, never assumed away.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final, Literal

from pydantic import BaseModel, Field

from corpus_studio.schemas.base import DatasetSchema, SchemaField
from corpus_studio.quality.text import tokenize_text_values

FieldRole = Literal["free_text", "categorical", "numeric", "path"]

RoleSource = Literal["schema", "unmeasured"]

# A declared FieldType maps to exactly one role. ``string`` is deliberately NOT prose: it is the
# short-scalar type (ids, labels, hashes, URLs, enum members), and ``text`` / ``markdown`` / ``code``
# are the prose types.
SCHEMA_TYPE_ROLES: Final[dict[str, FieldRole]] = {
    "text": "free_text",
    "markdown": "free_text",
    "code": "free_text",
    "messages": "free_text",
    "string": "categorical",
    "boolean": "categorical",
    "integer": "numeric",
    "float": "numeric",
    "file_path": "path",
    "image_path": "path",
}

# A container whose inner shape is not declared, and any field type this module does not know, is
# UNKNOWN content - never counted as prose.
_FALLBACK_ROLE: Final[FieldRole] = "categorical"

ANY_SHAPE: Final[frozenset[FieldRole]] = frozenset()

# The roles each quality signal needs in order to mean anything. An empty requirement is a signal
# that holds for ANY shape: an entirely empty row and a byte-identical duplicate row are debt in any
# schema, and a pasted credential can land in any string field.
SIGNAL_REQUIRED_ROLES: Final[dict[str, frozenset[FieldRole]]] = {
    "empty_rows": ANY_SHAPE,
    "exact_duplicates": ANY_SHAPE,
    "secrets": ANY_SHAPE,
    "personal_data": ANY_SHAPE,
    "normalized_duplicates": frozenset({"free_text"}),
    "low_information": frozenset({"free_text"}),
    "synthetic_patterns": frozenset({"free_text"}),
    "token_length_outliers": frozenset({"free_text"}),
    "category_imbalance": frozenset({"categorical", "numeric"}),
}

# Free-text signals characterize a TEXT corpus. A marginal prose field in a predominantly non-text
# record does not make their verdict meaningful, so they also require that free text be the measured
# majority of the dataset's content. Measured margins are wide on both sides: the builtin text
# corpora sit at 0.90-1.00, a normalized-bounding-box detection corpus at 0.15.
MIN_ASSESSED_CONTENT_SHARE: Final[float] = 0.5

_UNMEASURED_REASON: Final[str] = (
    "Applicability was not assessed: no schema was supplied, and whether a string field holds prose "
    "or an identifier is declared by a schema, not inferable from its values. Pass --schema (with "
    "--project-dir for a project-local schema) to have the text signals report applicability."
)


class QualityApplicability(BaseModel):
    """Whether each quality signal could run, and how much of the content it read.

    ``assessed_content_share`` is ``None`` exactly when ``role_source`` is ``unmeasured``: with no
    schema there is nothing to measure the share against, so it is reported absent rather than as a
    number that would read as a finding.
    """

    role_source: RoleSource
    assessed_content_share: float | None = None
    min_assessed_content_share: float = MIN_ASSESSED_CONTENT_SHARE
    field_roles: dict[str, str] = Field(default_factory=dict)
    applicable_signals: list[str] = Field(default_factory=list)
    not_applicable_signals: list[str] = Field(default_factory=list)
    reason: str = ""

    def applies(self, signal: str) -> bool:
        """Whether ``signal`` could run. An unknown signal name applies (fail-open on naming, not on
        assurance: every signal this module gates is listed in ``SIGNAL_REQUIRED_ROLES``)."""
        return signal not in self.not_applicable_signals

    @property
    def measured(self) -> bool:
        return self.role_source == "schema"


def _leaf_roles(fields: list[SchemaField], prefix: str) -> Iterator[tuple[str, FieldRole]]:
    for field in fields:
        name = f"{prefix}{field.name}"
        if field.type == "object" and field.fields:
            yield from _leaf_roles(field.fields, f"{name}.")
        elif field.type == "list" and field.item_type == "object" and field.item_fields:
            yield from _leaf_roles(field.item_fields, f"{name}[].")
        elif field.type == "list" and field.item_type is not None:
            yield name, SCHEMA_TYPE_ROLES.get(field.item_type, _FALLBACK_ROLE)
        else:
            yield name, SCHEMA_TYPE_ROLES.get(field.type, _FALLBACK_ROLE)


def schema_field_roles(schema: DatasetSchema) -> dict[str, FieldRole]:
    """Role per LEAF field path. Nested objects read ``parent.child``; a list of objects reads
    ``parent[].child``, index-free, because every element of a list is the same declared field."""
    return dict(_leaf_roles(schema.fields, ""))


def free_text_leaf_paths(schema: DatasetSchema) -> frozenset[str]:
    """Leaf paths ``schema`` declares as PROSE.

    These are the only fields templating detection should read: a repeated opening across rows
    means something in prose and nothing in a provenance tag, a path prefix, an enum member or a
    coordinate, all of which repeat by construction. A path not declared here - including a key
    the schema does not declare at all, such as a generation-metadata sidecar - is not prose.
    """
    return frozenset(
        path for path, role in schema_field_roles(schema).items() if role == "free_text"
    )


def _accumulate_tokens(
    data: Any,
    fields: list[SchemaField],
    prefix: str,
    totals: dict[str, int],
) -> None:
    """Add ``data``'s token counts into ``totals``, keyed by the leaf paths of ``fields``.

    Defensive by design: quality runs on UNVALIDATED rows, so wherever the data does not match the
    declared shape the value is counted whole at that leaf instead of being skipped, and any key the
    schema does not declare is counted under its own path. Nothing present goes uncounted - an
    undercounted denominator would overstate the assessed share.
    """
    if not isinstance(data, dict):
        totals[prefix.rstrip(".")] = totals.get(prefix.rstrip("."), 0) + len(
            tokenize_text_values(data)
        )
        return

    declared = {field.name: field for field in fields}
    for key, value in data.items():
        field = declared.get(key)
        name = f"{prefix}{key}"
        if field is None:
            totals[name] = totals.get(name, 0) + len(tokenize_text_values(value))
            continue
        if field.type == "object" and field.fields and isinstance(value, dict):
            _accumulate_tokens(value, field.fields, f"{name}.", totals)
        elif (
            field.type == "list"
            and field.item_type == "object"
            and field.item_fields
            and isinstance(value, list)
        ):
            for element in value:
                _accumulate_tokens(element, field.item_fields, f"{name}[].", totals)
        else:
            totals[name] = totals.get(name, 0) + len(tokenize_text_values(value))


def measure_applicability(
    rows: list[dict[str, Any]],
    schema: DatasetSchema | None = None,
) -> QualityApplicability:
    """Assess which signals apply to ``rows`` given ``schema``.

    With no schema the result is ``unmeasured``: every signal is listed applicable (the assessment
    is what is missing, not the signals) and ``reason`` says so and names the flag that fixes it.
    """
    signals = sorted(SIGNAL_REQUIRED_ROLES)
    if schema is None:
        return QualityApplicability(
            role_source="unmeasured",
            assessed_content_share=None,
            applicable_signals=signals,
            not_applicable_signals=[],
            reason=_UNMEASURED_REASON,
        )

    roles = schema_field_roles(schema)
    totals: dict[str, int] = {}
    for row in rows:
        _accumulate_tokens(row, schema.fields, "", totals)

    total_tokens = sum(totals.values())
    free_text_tokens = sum(
        count for path, count in totals.items() if roles.get(path) == "free_text"
    )
    share = free_text_tokens / total_tokens if total_tokens else 0.0
    roles_present = set(roles.values())

    applicable: list[str] = []
    not_applicable: list[str] = []
    for signal in signals:
        (applicable if _applies(signal, roles_present, share) else not_applicable).append(signal)

    return QualityApplicability(
        role_source="schema",
        assessed_content_share=round(share, 4),
        field_roles=dict(sorted(roles.items())),
        applicable_signals=applicable,
        not_applicable_signals=not_applicable,
        reason=_describe(schema, share, not_applicable),
    )


def _applies(signal: str, roles_present: set[FieldRole], share: float) -> bool:
    required = SIGNAL_REQUIRED_ROLES[signal]
    if not required:
        return True
    if not required & roles_present:
        return False
    if "free_text" in required:
        return share >= MIN_ASSESSED_CONTENT_SHARE
    return True


def _describe(schema: DatasetSchema, share: float, not_applicable: list[str]) -> str:
    percent = f"{share * 100:.1f}%"
    if not not_applicable:
        return (
            f"Schema '{schema.id}' declares free text over {percent} of the measured content; "
            f"every quality signal applies."
        )
    names = ", ".join(not_applicable)
    return (
        f"Schema '{schema.id}' declares free text over only {percent} of the measured content "
        f"(floor {MIN_ASSESSED_CONTENT_SHARE * 100:.0f}%), so these signals could not assess it and "
        f"report no result rather than a pass: {names}."
    )
