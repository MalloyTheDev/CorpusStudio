"""Real runners that plug into the platform run supervisor — platform slice 4.

The :class:`~corpus_studio.platform.supervisor.EchoRunner` proves the harness with zero heavy deps;
this module adds the runner that executes an ACTUAL training run. :class:`TrainingRunner` dispatches
by the RunPlan's chosen backend (``backend_ref`` — ``corpus_studio`` → ``trainer.run_training``,
``unsloth`` → ``unsloth_trainer.run_unsloth_training``; ``cpu_toy`` always uses the first-party CPU
smoke path): it reads the trainer config from the RunPlan's ``training_config_snapshot``, adapts the
``(step, total, loss)`` progress into ``RunEvent`` metrics, cooperatively aborts on cancel, classifies
a missing runtime as an ``ENVIRONMENT_FAILURE`` (never a crash), and returns the produced adapter
artifact. Both backends share the ``(config, *, progress_callback) -> TrainResult`` shape, so the
progress/cancel/error-classification harness is identical — "pick your framework", one runner.

The heavy stack (torch/transformers/…) is lazy-imported **inside** ``run()`` — importing this module
pulls only the platform contracts + the (import-light) trainer module, so the dependency-light
boundary holds. ``cpu_toy=True`` runs the tiny CPU smoke path (needs the ``[train]`` extra but no
GPU); the real GPU QLoRA path can only be user-smoke-tested.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from corpus_studio.platform.common import MemoryMetrics
from corpus_studio.platform.contracts import EventMetrics, RunPlan
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.gpu_health import classify_gpu_health
from corpus_studio.platform.supervisor import (
    ProducedArtifact,
    RunCancelled,
    RunContext,
    RunnerFailure,
)
from corpus_studio.platform.watchdog import MemorySampler, RunWatchdog, sample_gpu_memory

if TYPE_CHECKING:
    from corpus_studio.training.trainer import TrainResult, TrainRunConfig

    TrainerFn = Callable[..., TrainResult]


class _CancelTraining(Exception):
    """Internal: raised from inside the trainer's progress callback to abort a cancelled run — the
    only cooperative hook into an otherwise-blocking ``trainer.train()`` call."""


# Substrings that reliably identify a CUDA out-of-memory in a torch/transformers error message. The
# exception TYPE ``OutOfMemoryError`` (torch.cuda.OutOfMemoryError) is matched by name so torch need
# not be imported to classify.
_OOM_MARKERS = ("out of memory", "cuda oom", "cublas_status_alloc_failed", "cuda_error_out_of_memory")
# A NaN/Inf numeric signal — matched as a WHOLE WORD so "information" / "reinforcement" / "infinite"
# don't count as "inf" — co-occurring with a loss/gradient signal.
_NUMERIC_SIGNAL = re.compile(r"\b(nan|inf|infinity)\b")
_LOSS_GRAD_SIGNAL = re.compile(r"\b(loss|grad\w*)\b")


def classify_training_error(exc: BaseException) -> tuple[FailureTaxonomy, str | None]:
    """Map a training runtime exception to a :class:`FailureTaxonomy` + a remediation hint. Only the
    signatures we can identify with confidence are promoted from the generic ``FAIL``:

    * a CUDA out-of-memory (``OutOfMemoryError`` / "out of memory") → ``OOM``;
    * a NaN/Inf loss or gradient → ``NUMERICAL_FAILURE``.

    A genuine ``KERNEL_STALL`` (the sm_120 fused-attention deadlock) is a HANG and an
    ``ACCIDENTAL_SPILL`` is a slowdown — neither raises, so both belong to a watchdog + a
    memory-signature classifier, not to error-string matching. Anything unrecognized stays ``FAIL``
    rather than being mislabeled."""
    message = str(exc).lower()
    if classify_gpu_health(message) == "wedged":
        # 'device not ready' & friends: the GPU/driver is in a poisoned transient state (classically the
        # WSL2 GPU-PV wedge a prior crashed run leaves behind) — NOT a config bug. Surface the reset,
        # or the operator burns runs chasing a phantom that a `wsl --terminate` clears (this session's
        # own trap). Checked first: a wedged GPU can masquerade as any downstream failure.
        return (
            FailureTaxonomy.ENVIRONMENT_FAILURE,
            "The GPU appears WEDGED (a prior crashed CUDA process poisoned the driver / GPU-PV state) — "
            "this is NOT a config problem, so re-running the same command will keep failing. Reset the "
            "GPU and re-run: on WSL, `wsl --terminate <distro>` (or `wsl --shutdown`) from Windows "
            "PowerShell; on native Windows, restart or reset the display driver (Win+Ctrl+Shift+B); on "
            "Linux, `nvidia-smi --gpu-reset` or reboot.",
        )
    if type(exc).__name__ == "OutOfMemoryError" or any(marker in message for marker in _OOM_MARKERS):
        return (
            FailureTaxonomy.OOM,
            "reduce sequence_len or micro_batch_size, enable gradient checkpointing / offload, "
            "or use a smaller base model.",
        )
    if _NUMERIC_SIGNAL.search(message) and _LOSS_GRAD_SIGNAL.search(message):
        return (
            FailureTaxonomy.NUMERICAL_FAILURE,
            "lower the learning rate, add warmup, or check the dataset for malformed rows.",
        )
    return FailureTaxonomy.FAIL, None


# The peft export weights filename we reload to verify equivalence. A single named constant instead of
# the scattered literal (it also appears in artifacts.py / probes.py; a fully shared home is out of
# scope for this slice).
_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"


def _reload_verify_adapter(adapter_dir: str, expected_after_sha256: str) -> tuple[bool, str | None]:
    """Reload the exported adapter weights from disk and assert they reproduce the trained export state
    - the ``reload_verified`` guarantee ("the producing backend reloaded these weights and asserted
    equivalence"). Reads only the adapter-sized safetensors, never the base model.

    Returns ``(verified, reason)``: ``reason`` is ``None`` on success and a short human-readable string
    when an adapter that SHOULD reload does not (missing / unreadable / corrupt / mismatched), so the
    runner can surface WHY on the run's event stream instead of a bad artifact failing silently. A
    failure is always ``(False, reason)`` - never a crash of success admission, never a false claim.
    """
    from pathlib import Path  # noqa: PLC0415

    weights = Path(adapter_dir) / _ADAPTER_WEIGHTS_FILENAME
    if not weights.is_file():
        return False, f"exported adapter weights not found ({_ADAPTER_WEIGHTS_FILENAME})"
    try:
        import torch  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        from corpus_studio.training.trainer import capture_adapter_export_state  # noqa: PLC0415

        reloaded = load_file(str(weights))
        snapshot = capture_adapter_export_state(reloaded, torch, stage=StageMarker.export)
    except Exception as exc:  # noqa: BLE001 - any reload/parse failure means NOT verified, never a crash.
        return False, f"could not reload the exported adapter: {type(exc).__name__}: {exc}"
    if snapshot.state_sha256 != expected_after_sha256:
        return False, "the reloaded adapter does not reproduce the trained export state"
    return True, None


# Progress names the first-party trainer emits that are DELIBERATE sub-stage NOTES, not typed stages:
# they are logged, never promoted onto the enum-bound StageMarker spine (which the sealed IEEE research
# matrix pins exactly, so a new typed stage would need an amendment). Listed explicitly so an
# UNRECOGNIZED name is treated as a trainer bug and surfaced, rather than silently normalized (#752).
_INTENTIONAL_PROGRESS_NOTES = frozenset({"precision_verified"})


def _classify_progress_name(name: str) -> tuple[StageMarker | None, bool]:
    """Map a trainer progress name to ``(marker, is_intentional_note)``: a typed ``StageMarker`` gives
    ``(marker, False)``; a known intentional sub-stage note gives ``(None, True)``; anything else gives
    ``(None, False)`` so an unexpected/typo'd name surfaces instead of being silently logged as progress."""
    try:
        return StageMarker(name), False
    except ValueError:
        return None, name in _INTENTIONAL_PROGRESS_NOTES


class TrainingRunner:
    """Executes a real training run through ``training.trainer.run_training`` under the supervisor.

    ``cpu_toy`` selects the matching worker lane but never mutates the plan. ``max_steps`` is retained
    as a compatibility assertion only: it must equal the sealed schedule. A missing training runtime
    surfaces as an ``ENVIRONMENT_FAILURE``; absent/stale resolved execution is unsupported."""

    def __init__(
        self,
        *,
        cpu_toy: bool = False,
        max_steps: int | None = None,
        memory_sampler: MemorySampler = sample_gpu_memory,
        heartbeat_timeout_s: float = 600.0,
        poll_interval_s: float = 5.0,
    ) -> None:
        self.cpu_toy = cpu_toy
        self.max_steps = max_steps
        # The watchdog samples GPU memory (peak → the MEASURED fit; a spill → a warning) and flags a
        # heartbeat stall as an OBSERVABILITY SIGNAL only — a stderr heads-up + a warning, never an
        # abort and never a KERNEL_STALL manifest verdict (an in-process CUDA hang can't be killed or
        # classified; that's the subprocess-worker slice). `heartbeat_timeout_s` defaults high (a
        # spilling step is legitimately slow — minutes — so only a long silence trips the heads-up).
        # The sampler is injectable so the integration is testable without a GPU.
        self.memory_sampler = memory_sampler
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.poll_interval_s = poll_interval_s
        self.name = "cpu_toy" if cpu_toy else "training"

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        from corpus_studio.platform.planner import is_trivial_physical_execution  # noqa: PLC0415

        physical = ctx.plan.physical_execution
        supported_resource = True
        if physical is not None:
            resource = physical.resources[0]
            supported_resource = (
                resource.tier.value in {"pinned_ram", "pageable_ram"}
                and resource.device_kind is not None
                and resource.device_kind.value == "cpu"
                and resource.device_id == "cpu:0"
                if self.cpu_toy
                else resource.tier.value == "gpu"
                and resource.device_kind is not None
                and resource.device_kind.value == "cuda"
                and resource.device_id == "cuda:0"
            )
        if not is_trivial_physical_execution(physical) or not supported_resource:
            raise RunnerFailure(
                "the selected training runner cannot consume this physical execution spec",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation=(
                    "choose a backend whose isolated worker implements and functionally verifies the "
                    "plan's placement, offload, and parallel groups"
                ),
            )
        trainer_fn, backend_label = self._resolve_trainer(ctx.plan)
        # The manifest target reflects the backend that actually ran ("cpu_toy" for the smoke path).
        self.name = backend_label
        from corpus_studio.training.trainer import (  # noqa: PLC0415
            ExecutionPlacementDeviation,
            TrainingEvidenceError,
            TrainerEnvironmentError,
            TrainerError,
        )

        ctx.emit_stage(
            StageMarker.process_start,
            f"training run [{backend_label}]: validating sealed execution inputs",
        )
        config = self._resolve_config(ctx.plan, ctx.run_id)
        execution = ctx.plan.resolved_execution
        if execution is None:  # pragma: no cover - _resolve_config rejects this first.
            raise RunnerFailure(
                "training plan has no resolved execution configuration",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.process_start,
            )
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            verify_run_scoped_output_path,
        )

        try:
            verify_run_scoped_output_path(execution, ctx.run_id)
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.process_start,
                remediation="repair the sealed output root before dispatching a new run",
            ) from exc

        def _on_stall() -> None:
            # A true CUDA hang can't be force-killed in-process (the training thread is stuck in the
            # kernel) NOR classified onto the manifest (the run never returns) — killing a hang is the
            # subprocess-worker slice. So this is a heads-up ONLY, never an abort: print a signal so a
            # stuck run says so instead of dying silently. Runs on the watchdog thread → a bare stderr
            # write (not ctx.emit_*, whose seq is not thread-safe).
            print(
                f"[watchdog] no training progress for >{self.heartbeat_timeout_s:.0f}s. Normal during a "
                "long model download/load; if training is underway this may be a WDDM spill (10-25x "
                "slowdown) or a hung kernel (e.g. fused attention on sm_120). Kill the process if it "
                "never recovers.",
                file=sys.stderr,
                flush=True,
            )

        watchdog = RunWatchdog(
            sampler=self.memory_sampler,
            heartbeat_timeout_s=self.heartbeat_timeout_s,
            poll_interval_s=self.poll_interval_s,
            on_stall=_on_stall,
        )

        last_stage = StageMarker.process_start
        # Per-step evidence measured on the WORKER side (this closure runs in the managed child that
        # owns CUDA). ``last_step_monotonic`` is the previous optimizer-step boundary; the delta to the
        # current callback is that step's wall time. ``pending_tokens`` holds the trainer's per-step
        # token counts (emitted just before the loss log for the same step) so one metric record
        # carries loss + timing + tokens + allocator memory together.
        last_step_monotonic: list[float | None] = [None]
        pending_tokens: dict[int, tuple[int, int, int]] = {}

        def _token_counts(
            step: int, nonpadding_tokens: int, supervised_tokens: int, observed_microbatches: int
        ) -> None:
            # Best-effort, worker-sourced; the trainer never lets a counting fault reach here, but this
            # sink still tolerates any step (it is read by ``_progress`` for the matching step only).
            # ``observed_microbatches`` distinguishes an unavailable count (observer never fired -> map to
            # null) from a real observation, so a missed observer never becomes a fabricated 0.0 rate.
            pending_tokens[step] = (nonpadding_tokens, supervised_tokens, observed_microbatches)

        def _worker_gpu_memory() -> MemoryMetrics | None:
            # The child owns CUDA, so its torch allocator view (allocated/reserved/peak) is real here;
            # a probe fault must never fail training, so it degrades to null (never zero-filled).
            try:
                return self.memory_sampler()
            except Exception:  # noqa: BLE001 - observability only; a probe fault is not a run failure
                return None

        def _progress(step: int, total: int, loss: float | None) -> None:
            nonlocal last_stage
            if ctx.cancelled:
                raise _CancelTraining
            watchdog.beat()
            watchdog.sample()  # per-step peak capture (the thread also samples between steps)
            now = time.monotonic()
            previous = last_step_monotonic[0]
            step_time = (now - previous) if previous is not None else None
            last_step_monotonic[0] = now
            nonpadding, supervised, observed = pending_tokens.pop(step, (None, None, None))
            # An observer that never fired (observed == 0 or missing) yields UNAVAILABLE counts (null),
            # never a measured zero. Only a real observation contributes counts and derived rates; the
            # raw counts travel alongside the rates so the summary can validate rate == count / step_time.
            observed_ok = observed is not None and observed > 0
            nonpadding_tokens = nonpadding if observed_ok else None
            supervised_tokens = supervised if observed_ok else None
            tokens_per_sec = (
                nonpadding_tokens / step_time
                if nonpadding_tokens is not None and step_time and step_time > 0
                else None
            )
            supervised_per_sec = (
                supervised_tokens / step_time
                if supervised_tokens is not None and step_time and step_time > 0
                else None
            )
            metrics = EventMetrics(
                loss=loss,
                step_time_seconds=step_time,
                memory=_worker_gpu_memory(),
                nonpadding_tokens=nonpadding_tokens,
                supervised_tokens=supervised_tokens,
                observed_microbatches=observed,
                tokens_per_sec=tokens_per_sec,
                supervised_tokens_per_sec=supervised_per_sec,
            )
            ctx.emit_metric(optimizer_step=step, metrics=metrics, message=f"[{step}/{total}] step")
            last_stage = StageMarker.loss

        def _stage(name: str, message: str) -> None:
            # A setup milestone (model_loaded / quantized / …). Beat the watchdog so a long silent LOAD
            # doesn't look like a stall, and emit a stage RunEvent — which, over the worker pipe, resets
            # the subprocess supervisor's silence timer. Real progress, not a liveness heartbeat.
            watchdog.beat()
            nonlocal last_stage
            marker, intentional_note = _classify_progress_name(name)
            if marker is None:
                # Not a typed stage -> emit a LOG. A known intentional sub-stage note (precision_verified)
                # is deliberate; an unrecognized name is a trainer bug, marked so a typo surfaces rather
                # than being silently normalized into the progress stream (#752).
                prefix = "" if intentional_note else "unrecognized progress stage "
                ctx.emit_log(f"{prefix}{name}: {message}")
                return
            last_stage = marker
            ctx.emit_stage(marker, message)

        # Exact-lineage checkpoint WRITING (#440/#486): when the sealed policy enables a cadence, hand the
        # SFT trainer the plan-derived bound identities + a run-scoped checkpoints root so the reviewed
        # CheckpointCoordinator seals a checkpoint every cadence. A checkpoint-free policy passes nothing,
        # so a short run is byte-identical to before. (Only an enabled policy - an adapter-SFT-lane concept
        # today - reaches this; run_training fails closed if the identities/root are missing.)
        checkpoint_kwargs: dict[str, Any] = {}
        if (
            execution is not None
            and execution.checkpoint_policy.cadence_optimizer_steps is not None
        ):
            from corpus_studio.platform.checkpoint import (  # noqa: PLC0415
                bound_identities_from_plan,
            )
            from corpus_studio.platform.execution_config import (  # noqa: PLC0415
                run_scoped_training_output,
            )

            checkpoint_kwargs = {
                "checkpoint_bound": bound_identities_from_plan(ctx.plan),
                "source_run_id": ctx.run_id,
                "checkpoints_root": str(
                    run_scoped_training_output(execution, ctx.run_id, leaf="checkpoints")
                ),
            }
        try:
            with watchdog:
                result = trainer_fn(
                    config,
                    progress_callback=_progress,
                    stage_callback=_stage,
                    token_callback=_token_counts,
                    **checkpoint_kwargs,
                )
        except _CancelTraining:
            raise RunCancelled from None
        except ExecutionPlacementDeviation as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.placement_deviation,
                remediation="regenerate the RunPlan or use a backend that enforces its device map",
            ) from exc
        except TrainingEvidenceError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=exc.taxonomy,
                stage=exc.stage,
                remediation=exc.remediation,
            ) from exc
        except TrainerEnvironmentError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                stage=last_stage,
                remediation="run 'corpus-studio train-check' and verify the sealed environment",
            ) from exc
        except TrainerError as exc:
            # Unclassified trainer refusals are configuration deviations. Actual missing runtime
            # paths use RunnerFailure/TrainerEnvironmentError explicitly; never label every semantic
            # evidence failure as an environment problem.
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=last_stage,
                remediation="preserve the failed run and inspect the sealed execution contract",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — classify the runtime failure, don't leak it as FAIL
            taxonomy, remediation = classify_training_error(exc)
            raise RunnerFailure(
                str(exc) or type(exc).__name__,
                taxonomy=taxonomy,
                stage=last_stage,
                remediation=remediation,
            ) from exc
        finally:
            # Record whatever the watchdog measured — on EVERY terminal path. A failed/cancelled run
            # that spilled is the richest diagnostic; capturing it only on success would discard that.
            # This layer can record only an UNPROVEN observation. The supervisor promotes it after
            # output containment, adapter bytes, artifact integrity, losses, optimizer, and update
            # evidence all pass. A measured spill remains classified on either path.
            ctx.measured_peak = watchdog.peak
            ctx.final_fit = watchdog.measured_fit(proven=False)
            if watchdog.spilled:
                ctx.emit_warning(
                    "MEASURED a GPU-memory spill to shared system RAM during training (10-25x "
                    "slowdown, not a clean OOM) — reduce sequence_len / micro_batch_size, or offload."
                )
            if watchdog.ever_stalled:
                ctx.emit_warning(
                    "the run went >"
                    f"{self.heartbeat_timeout_s:.0f}s without progress at least once (a very slow "
                    "step or a stall, likely a WDDM spill) — see the stderr watchdog note."
                )

        if result.checkpoints and execution.checkpoint_policy.cadence_optimizer_steps is None:
            # Checkpoints from a run whose sealed policy DISABLED intermediate saving is a first-party-worker
            # bug. An ENABLED cadence legitimately writes them (the CheckpointCoordinator), so allow those.
            raise RunnerFailure(
                "trainer produced intermediate checkpoints despite the sealed disabled save policy",
                taxonomy=FailureTaxonomy.CHECKPOINT_FAILURE,
                stage=StageMarker.export,
                remediation="preserve the failed-run evidence and repair the first-party worker",
            )
        try:
            verify_run_scoped_output_path(
                execution,
                ctx.run_id,
                observed_path=result.output_dir,
                require_exists=True,
            )
            verify_run_scoped_output_path(
                execution,
                ctx.run_id,
                observed_path=result.adapter_path,
                require_exists=True,
            )
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="use the exact backend adapter for this execution contract",
            ) from exc

        from corpus_studio.training.artifact_registry import (  # noqa: PLC0415
            compute_weight_content_hash,
        )

        content_hash = compute_weight_content_hash(result.adapter_path)
        if content_hash is None:
            raise RunnerFailure(
                "training returned without readable adapter weight bytes",
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="inspect the trainer output and rerun from a new derived plan",
            )
        if result.execution_evidence is None:
            raise RunnerFailure(
                "training returned without sealed optimizer, loss, gradient, and update evidence",
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
                remediation="preserve the failed run and repair the first-party worker",
            )
        if result.steps != result.execution_evidence.completed_optimizer_steps:
            raise RunnerFailure(
                "trainer step count disagrees with its sealed execution evidence",
                taxonomy=FailureTaxonomy.OPTIMIZER_FAILURE,
                stage=StageMarker.optimizer_step,
            )
        ctx.training_execution_evidence = result.execution_evidence
        # reload_verified (#747): reload the exported adapter and assert it reproduces the trained export
        # state. The trained after-digest lives on the sealed execution evidence; without it (echo /
        # cpu-toy) there is nothing to assert equivalence against, so the artifact stays unverified.
        export_state = (
            result.execution_evidence.adapter_export_state
            if result.execution_evidence is not None
            else None
        )
        reload_verified = False
        if export_state is not None:
            reload_verified, reload_reason = _reload_verify_adapter(
                result.adapter_path, export_state.after_sha256
            )
            if reload_reason is not None:
                ctx.emit_log(f"reload_verified=false: {reload_reason}")
        ctx.emit_stage(StageMarker.export, f"adapter saved: {result.adapter_path}")
        artifact = ProducedArtifact(
            artifact_id=f"{ctx.run_id}-adapter-{content_hash[:12]}",
            kind="adapter",
            path=result.adapter_path,
            reload_verified=reload_verified,
        )
        ctx.emit_artifact(artifact)
        return [artifact]

    def _resolve_trainer(self, plan: RunPlan) -> tuple[TrainerFn, str]:
        """Resolve only a current backend manifest with an exact execution adapter."""
        from corpus_studio.platform.backends import (  # noqa: PLC0415
            backend_manifest_ref,
            get_backend,
        )
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            PREFERENCE_NOT_EXECUTABLE_REASON,
            PRETRAINING_NOT_EXECUTABLE_REASON,
        )

        if plan.resolved_execution is None and plan.resolved_preference_execution is not None:
            # Refuse a sealed preference (DPO) plan with a TYPED reason at the FIRST resolution step on the
            # direct-runner path (this runs before _resolve_config), so the refusal is reachable rather
            # than surfacing the generic "backend does not implement the sealed execution contract" below.
            raise RunnerFailure(
                PREFERENCE_NOT_EXECUTABLE_REASON,
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="await the DPO worker milestone that promotes preference_dpo to "
                "workload_verified; do not hand-edit the plan to the SFT lane",
            )
        if plan.resolved_execution is None and plan.resolved_pretraining_execution is not None:
            # Same defense-in-depth for a sealed pretraining plan: a typed refusal at the first resolution
            # step rather than the generic "backend does not implement the sealed execution contract".
            raise RunnerFailure(
                PRETRAINING_NOT_EXECUTABLE_REASON,
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="dispatch the plan through platform-run so required_runner_lane selects "
                "the pretraining lane; do not hand-edit the plan to the SFT lane",
            )
        backend_id = plan.backend_ref.id
        backend = get_backend(backend_id)
        if backend is None:
            raise RunnerFailure(
                f"no training runner is registered for backend '{backend_id}'",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        if plan.backend_ref != backend_manifest_ref(backend):
            raise RunnerFailure(
                "the current backend manifest differs from the one sealed into the RunPlan",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="regenerate the RunPlan against the current backend implementation",
            )
        execution = plan.resolved_execution
        if (
            execution is None
            or execution.contract_version not in backend.execution_contract_versions
        ):
            raise RunnerFailure(
                f"backend '{backend_id}' does not implement the sealed execution contract",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        if backend_id != "corpus_studio":
            raise RunnerFailure(
                f"backend '{backend_id}' has no exact resolved-execution adapter",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        try:
            from corpus_studio.training.trainer import run_training  # noqa: PLC0415

            if self.cpu_toy:
                return run_training, "cpu_toy"
            return run_training, "corpus_studio"
        except ImportError as exc:  # pragma: no cover - defensive; the trainer modules import cleanly
            raise RunnerFailure(
                f"the training runtime module could not be imported: {exc}",
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                stage=StageMarker.env_loaded,
                remediation="install the training extra: pip install '.[train]'",
            ) from exc
    def _resolve_config(self, plan: RunPlan, run_id: str) -> TrainRunConfig:
        """Consume the sealed policy and derive only its declared run-scoped output path."""
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            PREFERENCE_NOT_EXECUTABLE_REASON,
            PRETRAINING_NOT_EXECUTABLE_REASON,
            ExecutionConfigurationError,
            run_scoped_training_output,
            verify_execution_configuration_hash,
            verify_execution_non_dataset_inputs,
            verify_execution_objective,
        )
        from corpus_studio.training.trainer import train_config_from_resolved  # noqa: PLC0415

        execution = plan.resolved_execution
        if execution is None:
            if plan.resolved_preference_execution is not None:
                # Defense-in-depth: _resolve_trainer already refuses a preference plan with this typed
                # reason before this method runs; repeat it here so a direct _resolve_config call is legible
                # too (admit-at-planning / refuse-at-execution).
                raise RunnerFailure(
                    PREFERENCE_NOT_EXECUTABLE_REASON,
                    taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                    stage=StageMarker.env_loaded,
                    remediation="await the DPO worker milestone that promotes preference_dpo to "
                    "workload_verified; do not hand-edit the plan to the SFT lane",
                )
            if plan.resolved_pretraining_execution is not None:
                raise RunnerFailure(
                    PRETRAINING_NOT_EXECUTABLE_REASON,
                    taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                    stage=StageMarker.env_loaded,
                    remediation="dispatch the plan through platform-run so required_runner_lane "
                    "selects the pretraining lane; do not hand-edit the plan to the SFT lane",
                )
            raise RunnerFailure(
                "the RunPlan carries no ResolvedExecutionConfiguration to execute",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        if not verify_execution_configuration_hash(execution):
            raise RunnerFailure(
                "the resolved execution configuration hash does not match its body",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        sealed_cpu_toy = execution.runtime_mode == "cpu_toy"
        if self.cpu_toy != sealed_cpu_toy:
            raise RunnerFailure(
                "the selected runner lane does not match the sealed runtime_mode",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        if self.max_steps is not None and self.max_steps != execution.schedule.max_steps:
            raise RunnerFailure(
                "max_steps is execution-affecting and cannot override the sealed schedule",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
                remediation="create a derived RunPlan with a new execution hash",
            )
        # Intermediate checkpoint WRITING is supported on this adapter SFT lane: a sealed cadence drives the
        # reviewed CheckpointCoordinator in run_training (the trainer_fn call above threads the plan's bound
        # identities + a run-scoped checkpoints root). A checkpoint-free plan writes nothing. Consuming a
        # checkpoint to RESUME remains a separate slice; the planner refuses a cadence on the full-parameter
        # lane, whose worker does not yet write checkpoints.
        try:
            # The trainer owns one stable read/hash/capture of the dataset and parses those exact
            # bytes. Revalidating it here would create a redundant full-corpus pass.
            verify_execution_non_dataset_inputs(execution)
            verify_execution_objective(execution, task_type=plan.task_type.value)
            config = train_config_from_resolved(execution)
            return config.model_copy(
                update={"output_dir": str(run_scoped_training_output(execution, run_id))}
            )
        except (ExecutionConfigurationError, ValidationError) as exc:
            raise RunnerFailure(
                f"the resolved execution configuration is not executable: {exc}",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            ) from exc


class PretrainingRunner:
    """Executes a sealed from-scratch / continued pretraining run through
    ``training.pretraining_trainer.run_pretraining`` - the full-parameter sibling of ``TrainingRunner``.

    It dispatches the proven worker, emits the run's stages, samples the measured peak, and reports the
    worker-PROPOSED full-model success evidence on the context. It never runs an SFT / adapter path and
    it NEVER self-admits success: ``execute_run`` independently re-verifies the saved model before the
    evidence may reach the manifest. Whether a pretraining plan is admitted for a production run is gated
    upstream (the CLI / worker select this runner only once ``pretraining`` is workload_verified)."""

    def __init__(
        self,
        *,
        cpu_toy: bool = False,
        corpus_root: str = ".",
        memory_sampler: MemorySampler = sample_gpu_memory,
    ) -> None:
        # No watchdog/heartbeat here yet (the SFT runner's stall-observability is a later slice for the
        # pretraining lane); expose no no-op knobs until it is wired.
        self.cpu_toy = cpu_toy
        self.corpus_root = corpus_root
        self.memory_sampler = memory_sampler
        self.name = "pretraining_cpu_toy" if cpu_toy else "pretraining"

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        execution = ctx.plan.resolved_pretraining_execution
        if execution is None:
            raise RunnerFailure(
                "the pretraining runner requires a resolved pretraining execution",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.process_start,
            )
        # Run-scope the output exactly like the SFT lane: derive <output_dir>/runs/<run_id>/artifacts/
        # model from the sealed root + fresh run id, so two runs of one plan never collide and the sealed
        # output_layout="run_scoped_v1" claim is ENFORCED, not merely declared.
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            run_scoped_pretraining_output,
            verify_run_scoped_pretraining_output_path,
        )

        scoped_output = run_scoped_pretraining_output(execution, ctx.run_id)
        ctx.emit_stage(
            StageMarker.process_start,
            f"pretraining run [{self.name}]: dispatching the full-parameter worker",
        )
        from corpus_studio.training.pretraining_trainer import (  # noqa: PLC0415
            PretrainingError,
            run_pretraining,
        )

        try:
            result = run_pretraining(
                execution, corpus_root=self.corpus_root, output_dir=str(scoped_output)
            )
        except PretrainingError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
                remediation="preserve the failed run and inspect the first-party pretraining worker",
            ) from exc

        try:
            verify_run_scoped_pretraining_output_path(
                execution, ctx.run_id, observed_path=result.output_dir, require_exists=True
            )
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="preserve the failed run and repair the pretraining run-scoped output layout",
            ) from exc

        if result.execution_evidence is None:
            # The worker must return full-model success evidence; a missing proof is a hard failure,
            # never a silently-emitted artifact that execute_run would then reject downstream.
            raise RunnerFailure(
                "the pretraining worker returned without full-model success evidence",
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
                remediation="preserve the failed run and repair the first-party pretraining worker",
            )
        # The runner REPORTS the worker-proposed success evidence; execute_run re-verifies before admit.
        ctx.pretraining_success_evidence = result.execution_evidence
        try:
            ctx.measured_peak = self.memory_sampler()
        except Exception:  # noqa: BLE001 - observability only; a probe fault is not a run failure
            ctx.measured_peak = None

        artifact = ProducedArtifact(
            artifact_id=f"{ctx.run_id}-model-{result.execution_evidence.model_safetensors_sha256[:12]}",
            kind="model",
            path=result.output_dir,
        )
        ctx.emit_stage(StageMarker.export, f"full model saved: {result.output_dir}")
        ctx.emit_artifact(artifact)
        return [artifact]


class FullFinetuneRunner:
    """Executes a sealed full-parameter SFT run through ``training.full_finetune_trainer.run_full_finetune``
    - the full-MODEL sibling of ``TrainingRunner`` (adapter) using the same SFT data. It dispatches the
    worker, emits stages, samples the measured peak, and reports the worker-PROPOSED full-model success
    evidence on the context. It NEVER self-admits: ``execute_run`` independently reloads the saved model
    before the evidence may reach the manifest. Selected only once ``dense_full_finetune`` is
    workload_verified."""

    def __init__(
        self, *, cpu_toy: bool = False, memory_sampler: MemorySampler = sample_gpu_memory
    ) -> None:
        self.cpu_toy = cpu_toy
        self.memory_sampler = memory_sampler
        self.name = "full_finetune"

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        execution = ctx.plan.resolved_full_finetune_execution
        if execution is None:
            raise RunnerFailure(
                "the full-finetune runner requires a resolved full-finetune execution",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.process_start,
            )
        # Full-parameter SFT emits a full MODEL (leaf="model"), run-scoped exactly like the pretraining lane.
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            run_scoped_training_output,
            verify_run_scoped_output_path,
        )

        scoped_output = run_scoped_training_output(execution, ctx.run_id, leaf="model")
        ctx.emit_stage(
            StageMarker.process_start,
            f"full-parameter SFT run [{self.name}]: dispatching the full-parameter worker",
        )
        from corpus_studio.training.full_finetune_trainer import (  # noqa: PLC0415
            FullFinetuneError,
            run_full_finetune,
        )

        try:
            result = run_full_finetune(
                execution, output_dir=str(scoped_output), cpu_toy=self.cpu_toy
            )
        except FullFinetuneError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
                remediation="preserve the failed run and inspect the first-party full-finetune worker",
            ) from exc

        try:
            verify_run_scoped_output_path(
                execution, ctx.run_id, observed_path=result.output_dir,
                require_exists=True, leaf="model",
            )
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="preserve the failed run and repair the full-finetune run-scoped output layout",
            ) from exc

        # The runner REPORTS the worker-proposed success evidence; execute_run re-verifies before admit.
        ctx.full_finetune_success_evidence = result.success_evidence
        try:
            ctx.measured_peak = self.memory_sampler()
        except Exception:  # noqa: BLE001 - observability only; a probe fault is not a run failure
            ctx.measured_peak = None

        artifact = ProducedArtifact(
            artifact_id=(
                f"{ctx.run_id}-model-{result.success_evidence.model_safetensors_sha256[:12]}"
            ),
            kind="model",
            path=result.output_dir,
        )
        ctx.emit_stage(StageMarker.export, f"full model saved: {result.output_dir}")
        ctx.emit_artifact(artifact)
        return [artifact]


class PreferenceRunner:
    """Executes a sealed offline-DPO (preference) run through ``training.preference_worker.run_preference``
    - the adapter sibling of ``PretrainingRunner``. It dispatches the config-consuming DPO worker (which
    reuses the #781 sequence-chunked primitive), emits the run's stages, samples the measured peak, and
    reports the worker-PROPOSED adapter success evidence on the context. It NEVER self-admits: ``execute_run``
    independently re-verifies the saved adapter before the evidence may reach the manifest. Whether a
    preference plan is admitted is gated upstream (selected only once ``preference_dpo`` is
    workload_verified)."""

    def __init__(self, *, memory_sampler: MemorySampler = sample_gpu_memory) -> None:
        self.memory_sampler = memory_sampler
        self.name = "preference"

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        execution = ctx.plan.resolved_preference_execution
        if execution is None:
            raise RunnerFailure(
                "the preference runner requires a resolved preference execution",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.process_start,
            )
        # DPO exports a PEFT adapter, so it reuses the SFT run-scoped output (leaf="adapter").
        from corpus_studio.platform.execution_config import (  # noqa: PLC0415
            ExecutionConfigurationError,
            run_scoped_training_output,
            verify_run_scoped_output_path,
        )

        scoped_output = run_scoped_training_output(execution, ctx.run_id)
        ctx.emit_stage(
            StageMarker.process_start,
            f"preference (DPO) run [{self.name}]: dispatching the config-consuming worker",
        )
        from corpus_studio.training.preference_worker import (  # noqa: PLC0415
            PreferenceWorkerError,
            run_preference,
        )

        try:
            result = run_preference(execution, output_dir=str(scoped_output))
        except PreferenceWorkerError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UPDATE_FAILURE,
                stage=StageMarker.optimizer_step,
                remediation="preserve the failed run and inspect the first-party DPO worker",
            ) from exc

        try:
            verify_run_scoped_output_path(
                execution, ctx.run_id, observed_path=result.output_dir, require_exists=True
            )
        except ExecutionConfigurationError as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ARTIFACT_FAILURE,
                stage=StageMarker.export,
                remediation="preserve the failed run and repair the preference run-scoped output layout",
            ) from exc

        # The runner REPORTS the worker-proposed success evidence; execute_run re-verifies before admit.
        ctx.preference_success_evidence = result.success_evidence
        try:
            ctx.measured_peak = self.memory_sampler()
        except Exception:  # noqa: BLE001 - observability only; a probe fault is not a run failure
            ctx.measured_peak = None

        artifact = ProducedArtifact(
            artifact_id=(
                f"{ctx.run_id}-adapter-{result.success_evidence.adapter_safetensors_sha256[:12]}"
            ),
            kind="adapter",
            path=result.output_dir,
        )
        ctx.emit_stage(StageMarker.export, f"DPO adapter saved: {result.output_dir}")
        ctx.emit_artifact(artifact)
        return [artifact]


def build_lane_runner(
    runner_name: str, *, max_steps: int | None = None, corpus_root: str = "."
) -> Any:
    """Single source of truth for lane name -> Runner, shared by the in-process ``platform_run`` path and
    the subprocess worker (``worker._build_runner``) so the two cannot drift. The in-process CLI path once
    lacked the pretraining lanes while the worker had them - exactly the drift one factory prevents.
    ``max_steps`` is the CLI compatibility cap for the SFT trainer lane; ``corpus_root`` anchors a
    pretraining plan's relative shard locations. Echo / cpu_toy / training fall through as before."""
    from corpus_studio.platform.supervisor import EchoRunner  # noqa: PLC0415

    if runner_name == "echo":
        return EchoRunner()
    if runner_name == "preference":
        return PreferenceRunner()
    if runner_name == "full_finetune":
        return FullFinetuneRunner()
    if runner_name in ("pretraining", "pretraining_cpu_toy"):
        return PretrainingRunner(
            cpu_toy=(runner_name == "pretraining_cpu_toy"), corpus_root=corpus_root
        )
    return TrainingRunner(cpu_toy=(runner_name == "cpu_toy"), max_steps=max_steps)


def demo_training_plan(plan_id: str = "demo-cpu-toy") -> RunPlan:
    """A fully sealed CPU-toy plan. Missing train packages fail at execution, not plan parsing."""

    import importlib.metadata  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from corpus_studio.platform.backends import get_backend  # noqa: PLC0415
    from corpus_studio.platform.common import HashRef, PackageLock, Ref  # noqa: PLC0415
    from corpus_studio.platform.contracts import (  # noqa: PLC0415
        CapabilityReport,
        EffectiveCapabilities,
        EnvironmentProfile,
        ExecutionCapabilityCombination,
        ProbeResult,
    )
    from corpus_studio.platform.execution_config import stable_file_sha256  # noqa: PLC0415
    from corpus_studio.platform.planner import PlannerConstraints, build_run_plan  # noqa: PLC0415

    backend = get_backend("corpus_studio")
    assert backend is not None
    dataset = Path(__file__).resolve().parents[3] / "examples/datasets/instruction/train.jsonl"
    dataset_digest = stable_file_sha256(dataset)
    package_names = ["accelerate", "datasets", "peft", "torch", "transformers", "trl"]
    packages: list[PackageLock] = []
    for name in package_names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        packages.append(PackageLock(name=name, version=version))
    signature = "d" * 64
    profile = EnvironmentProfile.model_validate(
        {
            "environment_signature": signature,
            "host": {"os": "windows" if sys.platform == "win32" else "linux"},
        }
    )
    combination = ExecutionCapabilityCombination.model_validate(
        {
            "runtime_mode": "cpu_toy",
            "device": "cpu",
            "precision": "fp32",
            "quantization": "none",
            "adapter_method": "lora",
            "attention_impl": "eager",
            "attention_kernel": "eager",
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": "cpu_lora_execution",
        }
    )
    trainer_proof = ProbeResult(
        probe="trainer_contract",
        outcome=FailureTaxonomy.PASS,
        proves={
            "trainer_field": backend.trainer_fields,
            "trainer_init_field": backend.trainer_init_fields,
        },
    )
    execution_proof = ProbeResult(
        probe="cpu_lora_execution",
        outcome=FailureTaxonomy.PASS,
        proves={
            "adapter": ["lora"],
            "attention": ["eager"],
            "attention_kernel": ["eager"],
            "checkpoint": ["adapter_only"],
            "loss": ["cross_entropy"],
            "optimizer": ["adamw_torch"],
            "precision": ["fp32"],
        },
        execution_combinations=[combination],
    )
    report = CapabilityReport(
        backend_id="corpus_studio",
        backend_version=backend.backend_version,
        environment_ref=Ref(id=signature),
        readiness="cpu_toy_only",
        installed_packages=packages,
        probe_results=[execution_proof, trainer_proof],
        effective_capabilities=EffectiveCapabilities.model_validate(
            {
                "adapter_methods": ["lora"],
                "precision_modes": ["fp32"],
                "attention_impls": ["eager"],
                "attention_kernels": ["eager"],
                "checkpoint_impls": ["adapter_only"],
                "execution_contract_versions": ["1.0.0"],
                "execution_combinations": [combination.model_dump(mode="json")],
                "loss_impls": ["cross_entropy"],
                "optimizers": ["adamw_torch"],
                "trainer_fields": backend.trainer_fields,
                "trainer_init_fields": backend.trainer_init_fields,
            }
        ),
    )
    return build_run_plan(
        profile=profile,
        capabilities=report,
        dataset_ref=Ref(id="demo-dataset", hash=HashRef(value=dataset_digest)),
        constraints=PlannerConstraints(
            base_model="hf-internal-testing/tiny-random-LlamaForCausalLM",
            model_revision="9fb191250dd56d0ba7ec9785a025ed29c03d5998",
            dataset_path=str(dataset),
            dataset_content_sha256=dataset_digest,
            sequence_len=64,
            lora_r=4,
            lora_alpha=8,
            max_steps=2,
            allow_cpu_toy=True,
        ),
        plan_id=plan_id,
    )
