"""Exact checkpoint + resume lineage (#440).

A long first-party run must be resumable with EXACT lineage or not at all. This module is the
control-plane, torch-free half: it seals a :class:`CheckpointManifest` (hash + per-file byte integrity
+ an atomic completion marker), verifies a checkpoint fails closed on any partial / corrupt /
incomplete / externally-changed state, and admits a resume only against a byte-identical, fully
compatible target RunPlan. The worker populates the sealed training-state counters and the file
hashes; nothing here imports torch.

It deliberately does NOT enable automatic resume: execution stays checkpoint-free until a separately
reviewed trainer change consumes a :class:`CheckpointResumeRequest`. Until then this is the reviewed
*design* plus its verifier, and first-party runs expected to exceed 30 minutes remain blocked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from corpus_studio.platform.contracts import (
    CheckpointBoundIdentities,
    CheckpointManifest,
    ResumeLineage,
    RunPlan,
)

MANIFEST_FILENAME = "CheckpointManifest.json"


class CheckpointError(ValueError):
    """A checkpoint is unsealed, corrupt, incomplete, incompatible, or ambiguous. Every path that
    reaches this fails closed: a resume never proceeds on a checkpoint that did not fully verify."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        # A machine-actionable category: incomplete | corrupt | missing_file | external_change |
        # hash_mismatch | incompatible | ambiguous | malformed.
        self.reason = reason


# --------------------------------------------------------------------------------------------------
# Canonical manifest hashing (mirrors planner.compute_plan_hash)
# --------------------------------------------------------------------------------------------------
def checkpoint_manifest_hash_payload(manifest: CheckpointManifest) -> dict[str, Any]:
    """The canonical seal payload: the full manifest body with the self-referential hash field
    removed, so the digest covers every other field exactly once."""

    body = manifest.model_dump(mode="json")
    body.pop("checkpoint_manifest_hash", None)
    return body


def compute_checkpoint_manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_checkpoint_manifest_hash(manifest: CheckpointManifest) -> bool:
    expected = compute_checkpoint_manifest_hash(checkpoint_manifest_hash_payload(manifest))
    return manifest.checkpoint_manifest_hash == expected


# --------------------------------------------------------------------------------------------------
# Deriving the bound identities from a RunPlan (one derivation, used by seal + resumable-into)
# --------------------------------------------------------------------------------------------------
def bound_identities_from_plan(plan: RunPlan) -> CheckpointBoundIdentities:
    """The plan-derivable identity subset a checkpoint must bind. Worker-only fields (worker wheel,
    formatter/chat-template bytes) are left null here and populated by the worker at seal time; the
    admission below compares only what the plan can derive, and the worker re-verifies the rest when
    it restores the bytes."""

    execution = plan.resolved_execution
    if execution is None:
        raise CheckpointError(
            "checkpoint lineage requires a resolved execution configuration",
            reason="ambiguous",
        )
    environment_lock_hash = None
    if execution.environment_binding == "managed_lock" and execution.environment_ref.hash is not None:
        environment_lock_hash = execution.environment_ref.hash.value
    chat_template = getattr(execution, "chat_template_sha256", None)
    return CheckpointBoundIdentities(
        plan_hash=plan.plan_hash,
        execution_configuration_hash=execution.configuration_hash,
        backend_ref=execution.backend_ref,
        environment_lock_hash=environment_lock_hash,
        worker_wheel_sha256=None,
        model_ref=execution.inputs.model.ref,
        tokenizer_ref=execution.inputs.tokenizer.ref,
        dataset_ref=execution.inputs.dataset.ref,
        chat_template_sha256=chat_template,
        formatter_sha256=None,
        objective_ref=execution.objective_ref,
        seed=execution.seed,
        data_seed=execution.data_seed,
    )


# --------------------------------------------------------------------------------------------------
# Sealing (atomic complete marker)
# --------------------------------------------------------------------------------------------------
def seal_checkpoint_manifest(manifest: CheckpointManifest) -> CheckpointManifest:
    """Return the manifest with ``complete=True`` and a freshly computed ``checkpoint_manifest_hash``.
    Sealing is the last step: a manifest is only marked complete after every file exists and its hash
    is recorded, so an unsealed (``complete=False``) or unhashed manifest is never mistaken for
    resumable."""

    completed = manifest.model_copy(update={"complete": True, "checkpoint_manifest_hash": "0" * 64})
    digest = compute_checkpoint_manifest_hash(checkpoint_manifest_hash_payload(completed))
    return completed.model_copy(update={"checkpoint_manifest_hash": digest})


def write_checkpoint_manifest(manifest: CheckpointManifest, checkpoint_dir: str | Path) -> Path:
    """Crash-safe write: serialize to a temp file, fsync, then ``os.replace`` onto the final path.
    Combined with the ``complete`` marker, a torn write leaves either the prior manifest or nothing -
    never a half-written one a resume could read."""

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / MANIFEST_FILENAME
    tmp = directory / f".{MANIFEST_FILENAME}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(manifest.model_dump_json(indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final)
    return final


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    target = Path(path)
    if target.is_dir():
        target = target / MANIFEST_FILENAME
    if not target.is_file():
        raise CheckpointError(f"no checkpoint manifest at {target}", reason="missing_file")
    try:
        return CheckpointManifest.model_validate_json(target.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CheckpointError(f"malformed checkpoint manifest: {exc}", reason="malformed") from exc


# --------------------------------------------------------------------------------------------------
# Integrity verification (fail closed)
# --------------------------------------------------------------------------------------------------
def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_checkpoint_integrity(checkpoint_dir: str | Path) -> CheckpointManifest:
    """Load and fully verify the checkpoint under ``checkpoint_dir``. Fails closed on: a missing or
    malformed manifest; an unsealed (``complete=False``) manifest; a manifest-hash mismatch (tamper);
    a missing required file; or any file whose bytes/size no longer match the sealed entry (external
    change). Returns the verified manifest only when every check passes."""

    directory = Path(checkpoint_dir)
    manifest = load_checkpoint_manifest(directory)
    if not manifest.complete:
        raise CheckpointError(
            "checkpoint manifest is not marked complete; the write did not finish",
            reason="incomplete",
        )
    if not verify_checkpoint_manifest_hash(manifest):
        raise CheckpointError(
            "checkpoint manifest hash does not match its body; the manifest was altered",
            reason="hash_mismatch",
        )
    for entry in manifest.files:
        file_path = directory / entry.path
        if not file_path.is_file():
            raise CheckpointError(
                f"checkpoint file is missing: {entry.path}",
                reason="missing_file",
            )
        actual_hash, actual_size = _sha256_file(file_path)
        if actual_size != entry.size_bytes or actual_hash != entry.sha256:
            raise CheckpointError(
                f"checkpoint file changed on disk since it was sealed: {entry.path}",
                reason="external_change",
            )
    return manifest


# --------------------------------------------------------------------------------------------------
# Resume compatibility (fail closed) + admission
# --------------------------------------------------------------------------------------------------
# Identity fields the target RunPlan can derive; every one must match exactly.
_PLAN_DERIVABLE_FIELDS = (
    "plan_hash",
    "execution_configuration_hash",
    "environment_lock_hash",
    "model_ref",
    "tokenizer_ref",
    "dataset_ref",
    "objective_ref",
    "seed",
    "data_seed",
)


def verify_resumable_into(manifest: CheckpointManifest, plan: RunPlan) -> None:
    """Fail closed unless ``manifest`` binds identities compatible with ``plan``. Every plan-derivable
    identity (plan hash, execution-configuration hash, environment lock, model/tokenizer/dataset,
    objective, seeds, backend) must match exactly; any mismatch is an incompatible resume. The
    manifest's worker-only fields (worker wheel, formatter/chat-template bytes) are re-verified by the
    worker when it restores the bytes."""

    target = bound_identities_from_plan(plan)
    bound = manifest.bound
    if bound.backend_ref.id != target.backend_ref.id or (
        bound.backend_ref.hash.value if bound.backend_ref.hash else None
    ) != (target.backend_ref.hash.value if target.backend_ref.hash else None):
        raise CheckpointError(
            "checkpoint was produced under a different backend than the target run",
            reason="incompatible",
        )
    for field in _PLAN_DERIVABLE_FIELDS:
        bound_value = getattr(bound, field)
        target_value = getattr(target, field)
        if hasattr(bound_value, "id"):  # a Ref: compare id + hash
            bound_value = (bound_value.id, bound_value.hash.value if bound_value.hash else None)
            target_value = (target_value.id, target_value.hash.value if target_value.hash else None)
        if bound_value != target_value:
            raise CheckpointError(
                f"checkpoint binds a different {field} than the target run; resume is incompatible",
                reason="incompatible",
            )
    if (
        target.chat_template_sha256 is not None
        and bound.chat_template_sha256 is not None
        and target.chat_template_sha256 != bound.chat_template_sha256
    ):
        raise CheckpointError(
            "checkpoint binds a different chat template than the target run",
            reason="incompatible",
        )


def admit_resume(
    plan: RunPlan,
    checkpoint_dir: str | Path,
    *,
    resumed_run_id: str,
) -> ResumeLineage:
    """Verify a checkpoint is byte-intact AND compatible with ``plan``, then return the
    :class:`ResumeLineage` a fresh resumed run records. Fails closed if either check fails, so a
    resume is admitted only against a fully verified, fully compatible checkpoint. ``resumed_run_id``
    is the fresh run identity minted by the caller - never the parent run's id."""

    manifest = verify_checkpoint_integrity(checkpoint_dir)
    verify_resumable_into(manifest, plan)
    if resumed_run_id == manifest.source_run_id:
        raise CheckpointError(
            "a resumed run must mint a fresh run id, not reuse the source run",
            reason="ambiguous",
        )
    return ResumeLineage(
        parent_run_id=manifest.source_run_id,
        parent_checkpoint_id=manifest.checkpoint_id,
        parent_checkpoint_hash=manifest.checkpoint_manifest_hash,
        resumed_from_global_step=manifest.state.global_optimizer_step,
    )
