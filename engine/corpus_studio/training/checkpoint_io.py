"""Exact checkpoint write + resume execution for the first-party worker (#440).

This is the torch half that :mod:`corpus_studio.platform.checkpoint` (torch-free) was designed to
seal and verify. It serializes the resumable training state - adapter/trainable weights, optimizer,
scheduler, scaler, every RNG stream, the sampler/dataloader order and cursor, and the exact
epoch/step/microstep/grad-accumulation position - into a run-scoped checkpoint directory, then hands
the file hashes and counters to the control-plane sealer so the published checkpoint carries an
atomic completion marker and per-file byte integrity. Resume is the mirror image: the checkpoint is
fully verified (integrity + request pin + plan/worker identity) BEFORE any tensor is loaded, the
adapter weights are restored onto the live parameters, and the optimizer is rebuilt over those live
parameters so the resumed optimizer never references stale parameter objects.

Design invariants:

- **torch is lazy.** Nothing here imports torch at module load; the caller passes ``torch_module``
  exactly as the rest of :mod:`corpus_studio.training.trainer` does. Importing this module pulls no
  torch, so the control plane stays dependency-light.
- **The sealed hash is the trust anchor.** State files are ``torch.load``-ed only AFTER
  :func:`verify_checkpoint_integrity` proves every byte matches the sealed manifest and that no member
  is a symlink / hard link / escaping path. A tampered or substituted file fails closed and is never
  deserialized.
- **Atomic publish.** Every file is written into a run-scoped temp directory, fsynced, hashed, and
  sealed; the whole directory is then ``os.replace``-d onto the final path. A crash before the rename
  leaves an orphan temp dir, never a half-published checkpoint; a crash after it leaves a complete one.
- **Never mutate the parent.** Resume reads the parent checkpoint read-only and writes only under the
  fresh resumed run's own scope. Nothing here deletes or rewrites a parent run or checkpoint.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from corpus_studio.platform.checkpoint import (
    CheckpointError,
    _sha256_file,
    seal_checkpoint_manifest,
    verify_checkpoint_integrity,
    verify_matches_request,
    verify_resumable_into,
    write_checkpoint_manifest,
)
from corpus_studio.platform.contracts import (
    CheckpointBoundIdentities,
    CheckpointFileEntry,
    CheckpointManifest,
    CheckpointResumeRequest,
    RunPlan,
    SealedTrainingState,
)

# Fixed member filenames + their contract roles. The manifest pins each by exact bytes; the optimizer
# file is mandatory for a resumable checkpoint (enforced by the CheckpointManifest validator).
ADAPTER_FILE = "adapter_state.pt"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"
SCALER_FILE = "scaler.pt"
RNG_FILE = "rng.pt"
SAMPLER_FILE = "sampler.pt"
TRAINER_STATE_FILE = "trainer_state.json"


# --------------------------------------------------------------------------------------------------
# Low-level durable writes (fsync files + directories where the platform supports it)
# --------------------------------------------------------------------------------------------------
def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync so a rename/creation is durable. Skipped where the platform cannot
    open a directory for fsync (e.g. Windows), which never affects correctness of the atomic rename."""

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - non-Linux directory fsync unsupported; validated off-CI.
        return
    try:
        try:
            os.fsync(fd)
        except OSError:  # pragma: no cover - directory fsync unsupported on this filesystem.
            pass
    finally:
        os.close(fd)


def _torch_save_fsync(torch_module: Any, obj: Any, path: Path) -> None:
    torch_module.save(obj, str(path))
    _fsync_file(path)


# --------------------------------------------------------------------------------------------------
# RNG capture / restore (every stream that can affect byte-equivalent continuation)
# --------------------------------------------------------------------------------------------------
def capture_rng_state(torch_module: Any, *, include_numpy: bool = False) -> dict[str, Any]:
    """Snapshot every RNG stream a resume must restore: torch CPU, torch CUDA (per device), Python's
    ``random``, and - only when the run actually uses it - NumPy. Returns a plain dict serialized into
    the checkpoint's ``rng.pt``; :func:`rng_algorithm_descriptor` names the captured streams."""

    state: dict[str, Any] = {
        "torch_cpu": torch_module.get_rng_state(),
        "python": random.getstate(),
    }
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        state["cuda"] = cuda.get_rng_state_all()
    if include_numpy:
        import numpy as np  # noqa: PLC0415 - only when the run uses NumPy randomness.

        state["numpy"] = np.random.get_state()
    return state


def rng_algorithm_descriptor(rng_state: dict[str, Any]) -> str:
    """A stable, non-empty descriptor of which RNG streams a checkpoint captured (the sealed state's
    ``rng_algorithm``). Order is fixed so the same capture always yields the same descriptor."""

    streams = []
    if "torch_cpu" in rng_state:
        streams.append("torch-cpu-mt19937")
    if "cuda" in rng_state:
        streams.append("torch-cuda-philox")
    if "python" in rng_state:
        streams.append("python-mt19937")
    if "numpy" in rng_state:
        streams.append("numpy-mt19937")
    return "+".join(streams) if streams else "none"


def restore_rng_state(rng_state: dict[str, Any], torch_module: Any) -> None:
    """Restore every captured RNG stream so post-resume sampling/dropout reproduces the uninterrupted
    trajectory. CUDA state is restored only when the checkpoint captured it and CUDA is available."""

    if "torch_cpu" in rng_state:
        torch_module.set_rng_state(rng_state["torch_cpu"])
    cuda = getattr(torch_module, "cuda", None)
    if "cuda" in rng_state and cuda is not None and cuda.is_available():
        cuda.set_rng_state_all(rng_state["cuda"])
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        import numpy as np  # noqa: PLC0415 - only when the checkpoint captured NumPy state.

        np.random.set_state(rng_state["numpy"])


# --------------------------------------------------------------------------------------------------
# Live-parameter identity (the resumed optimizer must own the restored live parameters)
# --------------------------------------------------------------------------------------------------
def trainable_param_ids(model: Any) -> tuple[int, ...]:
    """The object ids of the model's currently trainable parameters, sorted. Used to prove the resumed
    optimizer references the live restored parameters, not stale ones."""

    ids = [id(param) for _, param in model.named_parameters() if param.requires_grad]
    return tuple(sorted(ids))


def optimizer_param_ids(optimizer: Any) -> tuple[int, ...]:
    ids: list[int] = []
    for group in optimizer.param_groups:
        ids.extend(id(param) for param in group["params"])
    return tuple(sorted(ids))


def assert_optimizer_over_live_params(optimizer: Any, model: Any) -> None:
    """Fail closed unless the optimizer's parameter set is EXACTLY the model's live trainable
    parameters (by object identity). Guards the resume ordering: adapter weights are restored onto the
    live parameters, then the optimizer is built over those same live objects - never a stale copy."""

    live = trainable_param_ids(model)
    owned = optimizer_param_ids(optimizer)
    if not live or owned != live:
        raise CheckpointError(
            "resumed optimizer does not reference the restored live trainable parameters",
            reason="incompatible",
        )


# --------------------------------------------------------------------------------------------------
# Save (run-scoped temp dir -> seal complete -> atomic directory publish)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepPosition:
    """The exact place a checkpoint sits on the training timeline. ``global_optimizer_step`` is the
    number of completed optimizer steps (>=1 for a checkpoint); ``microstep_within_step`` and
    ``consumed_microsteps`` place a resume at the precise next micro-batch under gradient accumulation."""

    epoch: float
    global_optimizer_step: int
    gradient_accumulation_steps: int
    microstep_within_step: int = 0
    consumed_microsteps: int = 0


def save_checkpoint(
    *,
    torch_module: Any,
    final_dir: str | Path,
    adapter_state: dict[str, Any],
    optimizer: Any,
    position: StepPosition,
    bound: CheckpointBoundIdentities,
    source_run_id: str,
    checkpoint_id: str,
    created_at: str,
    lr_scheduler: Any | None = None,
    scaler: Any | None = None,
    rng_state: dict[str, Any] | None = None,
    sampler_state: Any | None = None,
    trainer_state: dict[str, Any] | None = None,
    parent: tuple[str, str] | None = None,
) -> CheckpointManifest:
    """Serialize the full resumable state into a run-scoped temp directory, seal a complete manifest
    with per-file byte integrity, then atomically publish the directory onto ``final_dir``. Returns the
    sealed :class:`CheckpointManifest`. Refuses to overwrite an existing ``final_dir`` (a checkpoint id
    is unique, and a resume must never race a clobber)."""

    final = Path(final_dir)
    if final.exists():
        raise CheckpointError(
            f"refusing to overwrite an existing checkpoint directory: {final}",
            reason="incompatible",
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    temp = final.parent / f".{final.name}.tmp.{os.getpid()}"
    if temp.exists():  # pragma: no cover - a same-pid orphan is only possible after a hard crash.
        _rmtree(temp)
    temp.mkdir(parents=True)

    # 1. Serialize every present component (optimizer is mandatory). Order is irrelevant; the manifest
    #    sorts entries by path.
    entries: list[CheckpointFileEntry] = []
    _torch_save_fsync(torch_module, adapter_state, temp / ADAPTER_FILE)
    entries.append(_entry(temp, ADAPTER_FILE, "adapter_weights"))
    _torch_save_fsync(torch_module, optimizer.state_dict(), temp / OPTIMIZER_FILE)
    entries.append(_entry(temp, OPTIMIZER_FILE, "optimizer"))

    scheduler_captured = lr_scheduler is not None
    if lr_scheduler is not None:
        _torch_save_fsync(torch_module, lr_scheduler.state_dict(), temp / SCHEDULER_FILE)
        entries.append(_entry(temp, SCHEDULER_FILE, "scheduler"))
    scaler_captured = bool(scaler is not None and scaler.state_dict())
    if scaler is not None and scaler_captured:
        _torch_save_fsync(torch_module, scaler.state_dict(), temp / SCALER_FILE)
        entries.append(_entry(temp, SCALER_FILE, "scaler"))
    rng_captured = rng_state is not None
    if rng_state is not None:
        _torch_save_fsync(torch_module, rng_state, temp / RNG_FILE)
        entries.append(_entry(temp, RNG_FILE, "rng"))
    sampler_captured = sampler_state is not None
    if sampler_state is not None:
        _torch_save_fsync(torch_module, sampler_state, temp / SAMPLER_FILE)
        entries.append(_entry(temp, SAMPLER_FILE, "sampler"))
    if trainer_state is not None:
        import json  # noqa: PLC0415

        path = temp / TRAINER_STATE_FILE
        path.write_text(json.dumps(trainer_state, sort_keys=True, indent=2), encoding="utf-8")
        _fsync_file(path)
        entries.append(_entry(temp, TRAINER_STATE_FILE, "trainer_state"))

    # 2. Describe the sealed state (presence flags + exact timeline position).
    state = SealedTrainingState(
        scheduler_captured=scheduler_captured,
        scaler_captured=bool(scaler_captured),
        rng_captured=rng_captured,
        sampler_state_captured=sampler_captured,
        rng_algorithm=rng_algorithm_descriptor(rng_state) if rng_state is not None else None,
        epoch=position.epoch,
        global_optimizer_step=position.global_optimizer_step,
        microstep_within_step=position.microstep_within_step,
        gradient_accumulation_steps=position.gradient_accumulation_steps,
        consumed_microsteps=position.consumed_microsteps,
    )

    # 3. Build, seal (complete=True + hash), and atomically write the manifest INSIDE the temp dir.
    parent_id, parent_hash = parent if parent is not None else (None, None)
    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        checkpoint_manifest_hash="0" * 64,
        source_run_id=source_run_id,
        parent_checkpoint_id=parent_id,
        parent_checkpoint_hash=parent_hash,
        created_at=created_at,
        complete=False,
        bound=bound,
        state=state,
        files=sorted(entries, key=lambda entry: entry.path),
    )
    sealed = seal_checkpoint_manifest(manifest)
    write_checkpoint_manifest(sealed, temp)

    # 4. Atomically publish: fsync the temp dir, rename it onto the final path, fsync the parent.
    _fsync_dir(temp)
    os.replace(temp, final)
    _fsync_dir(final.parent)
    return sealed


def _entry(directory: Path, name: str, role: str) -> CheckpointFileEntry:
    sha256, size = _sha256_file(directory / name)
    return CheckpointFileEntry(path=name, role=role, sha256=sha256, size_bytes=size)  # type: ignore[arg-type]


def _rmtree(path: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------------------------------
# Restore (verify-before-load, then rebuild over live parameters)
# --------------------------------------------------------------------------------------------------
@dataclass
class RestoreResult:
    """The verified checkpoint plus the live objects a resumed run continues from. ``position`` is the
    exact timeline place; ``resumed_from_global_step`` is the last completed optimizer step, so the
    resumed loop's first NEW optimizer step is ``resumed_from_global_step + 1``."""

    manifest: CheckpointManifest
    optimizer: Any
    lr_scheduler: Any | None
    scaler: Any | None
    sampler_state: Any | None
    position: SealedTrainingState
    resumed_from_global_step: int


def restore_checkpoint(
    *,
    torch_module: Any,
    request: CheckpointResumeRequest,
    plan: RunPlan,
    model: Any,
    apply_adapter_state: Callable[[Any, dict[str, Any]], None],
    build_optimizer: Callable[[], Any],
    build_lr_scheduler: Callable[[Any], Any] | None = None,
    build_scaler: Callable[[], Any] | None = None,
    worker_wheel_sha256: str | None = None,
    formatter_sha256: str | None = None,
    chat_template_sha256: str | None = None,
    map_location: Any = "cpu",
) -> RestoreResult:
    """Verify a checkpoint fully, then restore it onto ``model`` and fresh optimizer/scheduler/scaler.

    Verification runs to completion BEFORE any tensor is loaded: byte integrity + safe members
    (:func:`verify_checkpoint_integrity`), the exact request pin (:func:`verify_matches_request`), the
    plan-derivable identities (:func:`verify_resumable_into`), and the worker-only identities the plan
    cannot derive (worker wheel, formatter, chat template). Only then are the adapter weights restored
    onto the live parameters, the optimizer rebuilt over those live parameters and loaded, and every
    RNG stream restored. Fails closed on any mismatch; never mutates the parent checkpoint."""

    directory = Path(request.checkpoint_dir)
    manifest = verify_checkpoint_integrity(directory)
    verify_matches_request(manifest, request)
    verify_resumable_into(manifest, plan)
    _verify_worker_identities(
        manifest,
        worker_wheel_sha256=worker_wheel_sha256,
        formatter_sha256=formatter_sha256,
        chat_template_sha256=chat_template_sha256,
    )

    def _load(name: str) -> Any:
        return torch_module.load(str(directory / name), map_location=map_location, weights_only=False)

    # Restore adapter/trainable weights onto the LIVE parameters first, so the optimizer built next
    # owns exactly those objects.
    apply_adapter_state(model, _load(ADAPTER_FILE))

    optimizer = build_optimizer()
    optimizer.load_state_dict(_load(OPTIMIZER_FILE))
    assert_optimizer_over_live_params(optimizer, model)

    lr_scheduler = None
    if manifest.state.scheduler_captured:
        if build_lr_scheduler is None:
            raise CheckpointError(
                "checkpoint captured a scheduler but no scheduler builder was provided",
                reason="incompatible",
            )
        lr_scheduler = build_lr_scheduler(optimizer)
        lr_scheduler.load_state_dict(_load(SCHEDULER_FILE))

    scaler = None
    if manifest.state.scaler_captured:
        if build_scaler is None:
            raise CheckpointError(
                "checkpoint captured a scaler but no scaler builder was provided",
                reason="incompatible",
            )
        scaler = build_scaler()
        scaler.load_state_dict(_load(SCALER_FILE))

    if manifest.state.rng_captured:
        restore_rng_state(_load(RNG_FILE), torch_module)

    sampler_state = _load(SAMPLER_FILE) if manifest.state.sampler_state_captured else None

    return RestoreResult(
        manifest=manifest,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        sampler_state=sampler_state,
        position=manifest.state,
        resumed_from_global_step=manifest.state.global_optimizer_step,
    )


def _verify_worker_identities(
    manifest: CheckpointManifest,
    *,
    worker_wheel_sha256: str | None,
    formatter_sha256: str | None,
    chat_template_sha256: str | None,
) -> None:
    """Re-verify the identities the RunPlan cannot derive but the worker knows: the worker wheel, the
    formatter bytes, and the chat template. When the checkpoint bound one and the caller supplies the
    current value, they must match exactly - otherwise the resume runs different worker bytes."""

    bound = manifest.bound
    checks = (
        ("worker wheel", bound.worker_wheel_sha256, worker_wheel_sha256),
        ("formatter", bound.formatter_sha256, formatter_sha256),
        ("chat template", bound.chat_template_sha256, chat_template_sha256),
    )
    for label, sealed, current in checks:
        if sealed is not None and current is not None and sealed != current:
            raise CheckpointError(
                f"checkpoint binds a different {label} than this worker; resume is incompatible",
                reason="incompatible",
            )


# --------------------------------------------------------------------------------------------------
# Cadence + retention + lineage chaining driver
# --------------------------------------------------------------------------------------------------
class CheckpointCoordinator:
    """Drives checkpoint writing over a run: it decides when a checkpoint is due (an optimizer-step
    cadence), chains each checkpoint to the previous one as its parent, and prunes this run's own
    oldest checkpoints past ``keep_last``. It writes only under ``checkpoints_root`` (the fresh run's
    scope) and never touches a parent RUN - only this run's own intermediate checkpoints.

    A cadence of ``None`` is the checkpoint-free policy: :meth:`maybe_checkpoint` always returns None,
    so a short run behaves exactly as it did before checkpoints existed."""

    def __init__(
        self,
        *,
        torch_module: Any,
        checkpoints_root: str | Path,
        source_run_id: str,
        bound: CheckpointBoundIdentities,
        clock: Callable[[], str],
        cadence_optimizer_steps: int | None,
        keep_last: int | None = None,
    ) -> None:
        if cadence_optimizer_steps is not None and cadence_optimizer_steps < 1:
            raise CheckpointError(
                "checkpoint cadence must be a positive optimizer-step count",
                reason="incompatible",
            )
        if keep_last is not None and keep_last < 1:
            raise CheckpointError("keep_last must be a positive count", reason="incompatible")
        self._torch = torch_module
        self._root = Path(checkpoints_root)
        self._source_run_id = source_run_id
        self._bound = bound
        self._clock = clock
        self._cadence = cadence_optimizer_steps
        self._keep_last = keep_last
        self._last: CheckpointManifest | None = None
        self._written: list[Path] = []

    @property
    def enabled(self) -> bool:
        return self._cadence is not None

    @property
    def last_manifest(self) -> CheckpointManifest | None:
        return self._last

    def is_due(self, global_optimizer_step: int) -> bool:
        return self._cadence is not None and global_optimizer_step % self._cadence == 0

    def maybe_checkpoint(
        self,
        *,
        global_optimizer_step: int,
        epoch: float,
        gradient_accumulation_steps: int,
        adapter_state: dict[str, Any],
        optimizer: Any,
        lr_scheduler: Any | None = None,
        scaler: Any | None = None,
        rng_state: dict[str, Any] | None = None,
        sampler_state: Any | None = None,
        microstep_within_step: int = 0,
        consumed_microsteps: int = 0,
        trainer_state: dict[str, Any] | None = None,
    ) -> CheckpointManifest | None:
        """Write a checkpoint iff one is due at ``global_optimizer_step``; else return None. The new
        checkpoint's parent is the previous one written by this coordinator (lineage chain)."""

        if not self.is_due(global_optimizer_step):
            return None
        checkpoint_id = f"{self._source_run_id}-ckpt-step-{global_optimizer_step:08d}"
        final_dir = self._root / f"step-{global_optimizer_step:08d}"
        parent = (
            (self._last.checkpoint_id, self._last.checkpoint_manifest_hash)
            if self._last is not None
            else None
        )
        manifest = save_checkpoint(
            torch_module=self._torch,
            final_dir=final_dir,
            adapter_state=adapter_state,
            optimizer=optimizer,
            position=StepPosition(
                epoch=epoch,
                global_optimizer_step=global_optimizer_step,
                gradient_accumulation_steps=gradient_accumulation_steps,
                microstep_within_step=microstep_within_step,
                consumed_microsteps=consumed_microsteps,
            ),
            bound=self._bound,
            source_run_id=self._source_run_id,
            checkpoint_id=checkpoint_id,
            created_at=self._clock(),
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            rng_state=rng_state,
            sampler_state=sampler_state,
            trainer_state=trainer_state,
            parent=parent,
        )
        self._last = manifest
        self._written.append(final_dir)
        self._prune()
        return manifest

    def _prune(self) -> None:
        """Remove this run's own oldest checkpoints beyond ``keep_last``. Only directories this
        coordinator wrote are ever removed; the freshest ``keep_last`` are always kept."""

        if self._keep_last is None or len(self._written) <= self._keep_last:
            return
        stale = self._written[: len(self._written) - self._keep_last]
        self._written = self._written[len(self._written) - self._keep_last :]
        for directory in stale:
            _rmtree(directory)
