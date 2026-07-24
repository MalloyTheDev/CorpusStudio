"""Immutable assurance records + the Phase-1 ChangeSetRecord builder.

Every immutable assurance record shares one envelope::

    {"record_type", "schema_version", "payload", "provenance", "record_digest"}

``record_digest`` is the ``sha256:`` digest of the canonical envelope with the ``record_digest``
field itself EXCLUDED - so a record's integrity can be re-verified by dropping that one field and
re-canonicalizing. Records are never mutated in place; any repository mutation would create a new
``ChangeSetRecord`` root (that lifecycle belongs to a later phase - Phase 1 only builds the root).

A ``ChangeSetRecord`` describes what changed between two source views. The requested ``scope``
selects which two views (the base side is always read from the object store):
  * ``workspace``       - base = tree at ``merge-base(HEAD, --base)``; candidate = the local WORKING
                          tree (includes uncommitted + untracked edits).
  * ``head``            - base = tree at ``merge-base(HEAD, --base)``; candidate = the committed HEAD
                          tree. This is the branch's OWN diff, independent of local uncommitted edits
                          - the view an integration gate binds to the exact commit it validated.
  * ``merge_candidate`` - base = the ``--base`` TIP commit; candidate = the tree a 3-way merge of HEAD
                          into that tip would produce. This is what merging would ADD to the base as
                          it stands now (it accounts for the base having moved since the branch point);
                          a conflicted merge fails closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from assurance import KERNEL_VERSION
from assurance.canonical_json import sha256_digest
from assurance.fingerprint import changed_path_record, compute_fingerprint
from assurance.git_state import (
    AssuranceError,
    GitContext,
    changed_paths_between,
    changed_tracked_paths,
    discover_git_context,
    merge_base,
    resolve_commit,
    untracked_paths,
    write_merge_tree,
)
from assurance.source_views import FileState, GitTreeSourceView, WorkspaceSourceView

RECORD_SCHEMA_VERSION = 1
CHANGE_SET_RECORD_TYPE = "change_set"
# Implemented Git scopes. ``index`` remains a later phase (staged index state is not yet modeled).
SUPPORTED_SCOPES = ("workspace", "head", "merge_candidate")


class ChangeSetError(AssuranceError):
    """A change set cannot be produced for the requested state."""


class ScopeNotImplemented(ChangeSetError):
    """A Git scope the kernel does not implement was requested (e.g. ``index``)."""


class ScopeUnavailable(ChangeSetError):
    """A supported scope cannot be resolved for the current repo state (e.g. a tree scope on an
    unborn HEAD - there is no committed HEAD tree to compare)."""


class ChangeSetUnstable(ChangeSetError):
    """The working tree changed during collection; two snapshots of the same state disagree."""


def seal_record(
    record_type: str, schema_version: int, payload: Any, provenance: Any
) -> dict[str, Any]:
    """Return the sealed record envelope: the four fields plus a ``record_digest`` over them."""
    envelope = {
        "record_type": record_type,
        "schema_version": schema_version,
        "payload": payload,
        "provenance": provenance,
    }
    return {**envelope, "record_digest": sha256_digest(envelope)}


def verify_record(record: dict[str, Any]) -> bool:
    """Re-derive the record digest (excluding the field itself) and compare it to the claim."""
    if "record_digest" not in record:
        return False
    envelope = {key: value for key, value in record.items() if key != "record_digest"}
    return sha256_digest(envelope) == record["record_digest"]


def _states_equal(left: FileState | None, right: FileState | None) -> bool:
    if left is None or right is None:
        return left is right
    return left == right


class _ScopePlan:
    """The resolved Git identities a scope compares (computed once, reused across both passes).

    ``candidate_oid`` is a committed tree-ish for the tree scopes and ``None`` for ``workspace``.
    The labels are the human ``base_view`` / ``candidate_view`` names recorded in the payload.
    """

    __slots__ = ("base_oid", "base_label", "candidate_kind", "candidate_oid", "candidate_label")

    def __init__(self, base_oid: str, base_label: str, candidate_kind: str,
                 candidate_oid: str | None, candidate_label: str) -> None:
        self.base_oid = base_oid
        self.base_label = base_label
        self.candidate_kind = candidate_kind  # "workspace" | "tree"
        self.candidate_oid = candidate_oid
        self.candidate_label = candidate_label


def _resolve_scope(ctx: GitContext, scope: str, base_ref: str) -> _ScopePlan:
    """Resolve the two committed identities a scope compares (read-only). Fails closed on a scope
    that cannot apply to the current repo state."""
    if scope == "workspace":
        base_oid = merge_base(ctx, resolve_commit(ctx.root, base_ref))
        return _ScopePlan(base_oid, "merge_base", "workspace", None, "workspace")
    if scope == "head":
        if not ctx.head_oid:
            raise ScopeUnavailable("the 'head' scope needs a committed HEAD tree; HEAD is unborn")
        base_oid = merge_base(ctx, resolve_commit(ctx.root, base_ref))
        return _ScopePlan(base_oid, "merge_base", "tree", ctx.head_oid, "head")
    if scope == "merge_candidate":
        if not ctx.head_oid:
            raise ScopeUnavailable("the 'merge_candidate' scope needs a committed HEAD; HEAD is unborn")
        base_tip = resolve_commit(ctx.root, base_ref)
        merge_tree = write_merge_tree(ctx.root, base_tip, ctx.head_oid)  # fail-closed on conflict/old git
        return _ScopePlan(base_tip, "base_tip", "tree", merge_tree, "merge_candidate")
    raise ScopeNotImplemented(
        f"scope {scope!r} is not implemented (supported: {', '.join(SUPPORTED_SCOPES)})"
    )


def _candidate_paths(root: Path, plan: _ScopePlan) -> set[str]:
    """The paths whose state might differ on the candidate side of ``plan``."""
    if plan.candidate_kind == "workspace":
        # working tree: tracked diffs vs the base PLUS untracked non-ignored files.
        return set(changed_tracked_paths(root, plan.base_oid)) | set(untracked_paths(root))
    # a committed candidate tree: a pure tree-vs-tree diff (no untracked files exist in a tree).
    assert plan.candidate_oid is not None  # guaranteed by _resolve_scope for the tree kinds
    return set(changed_paths_between(root, plan.base_oid, plan.candidate_oid))


def _collect_change_set(root: Path, scope: str, plan: _ScopePlan) -> tuple[str, list[dict[str, Any]]]:
    """One snapshot pass: return ``(fingerprint, sorted changed-path records)`` for the plan."""
    base_view = GitTreeSourceView(root, plan.base_oid)
    candidate_view: GitTreeSourceView | WorkspaceSourceView = (
        WorkspaceSourceView(root) if plan.candidate_kind == "workspace"
        else GitTreeSourceView(root, plan.candidate_oid)  # type: ignore[arg-type]
    )
    records: list[dict[str, Any]] = []
    for path in sorted(_candidate_paths(root, plan)):
        base_state = base_view.state(path)
        candidate_state = candidate_view.state(path)
        if _states_equal(base_state, candidate_state):
            continue  # defensive: only emit genuinely differing states
        records.append(changed_path_record(path, base_state, candidate_state))
    fingerprint = compute_fingerprint(scope, plan.base_oid, records)
    return fingerprint, records


def build_change_set_record(
    *,
    start_dir: Path,
    scope: str = "workspace",
    base_ref: str = "main",
    _between_passes: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Build a sealed ChangeSetRecord for the selected Git ``scope`` (see the module docstring).

    Two independent snapshot passes are taken and their fingerprints compared; if they disagree
    the working tree moved during collection and the kernel fails closed with
    :class:`ChangeSetUnstable` rather than emit an internally inconsistent record. (The tree scopes
    read only immutable committed objects, so they are deterministic across passes; the guard is
    meaningful for the ``workspace`` scope, whose candidate is the live working tree.)
    ``_between_passes`` is a test-only hook invoked between the two passes.
    """
    if scope not in SUPPORTED_SCOPES:
        raise ScopeNotImplemented(
            f"scope {scope!r} is not implemented (supported: {', '.join(SUPPORTED_SCOPES)})"
        )
    ctx = discover_git_context(start_dir)
    plan = _resolve_scope(ctx, scope, base_ref)

    fingerprint, records = _collect_change_set(ctx.root, scope, plan)
    if _between_passes is not None:
        _between_passes()
    second_fingerprint, _second_records = _collect_change_set(ctx.root, scope, plan)
    if fingerprint != second_fingerprint:
        raise ChangeSetUnstable(
            "the working tree changed during change-set collection (two snapshots disagree); "
            "re-run against a quiescent tree"
        )

    payload: dict[str, Any] = {
        "scope": scope,
        "base_view": plan.base_label,
        "base_oid": plan.base_oid,
        "candidate_view": plan.candidate_label,
        "changed_path_count": len(records),
        "changed_paths": records,
        "fingerprint": fingerprint,
    }
    provenance: dict[str, Any] = {
        "tool": "cs_assure",
        "tool_version": KERNEL_VERSION,
        "record_kernel_schema_version": RECORD_SCHEMA_VERSION,
        "base_ref": base_ref,
        "base_oid": plan.base_oid,
        "head_oid": ctx.head_oid,
        "is_shallow": ctx.is_shallow,
    }
    if plan.candidate_oid is not None:  # tree scopes: pin the exact candidate tree-ish (auditability)
        payload["candidate_oid"] = plan.candidate_oid
        provenance["candidate_oid"] = plan.candidate_oid
    return seal_record(CHANGE_SET_RECORD_TYPE, RECORD_SCHEMA_VERSION, payload, provenance)
