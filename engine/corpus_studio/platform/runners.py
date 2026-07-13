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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from corpus_studio.platform.contracts import RunPlan
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
            TrainerError,
        )

        config = self._resolve_config(ctx.plan, ctx.run_id)
        ctx.emit_stage(
            StageMarker.process_start, f"training run [{backend_label}]: {config.base_model}"
        )

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

        def _progress(step: int, total: int, loss: float | None) -> None:
            if ctx.cancelled:
                raise _CancelTraining
            watchdog.beat()
            watchdog.sample()  # per-step peak capture (the thread also samples between steps)
            ctx.emit_metric(optimizer_step=step, loss=loss, message=f"[{step}/{total}] step")

        def _stage(name: str, message: str) -> None:
            # A setup milestone (model_loaded / quantized / …). Beat the watchdog so a long silent LOAD
            # doesn't look like a stall, and emit a stage RunEvent — which, over the worker pipe, resets
            # the subprocess supervisor's silence timer. Real progress, not a liveness heartbeat.
            watchdog.beat()
            try:
                marker = StageMarker(name)
            except ValueError:
                ctx.emit_log(f"{name}: {message}")
                return
            ctx.emit_stage(marker, message)

        succeeded = False
        try:
            with watchdog:
                result = trainer_fn(config, progress_callback=_progress, stage_callback=_stage)
            succeeded = True
        except _CancelTraining:
            raise RunCancelled from None
        except ExecutionPlacementDeviation as exc:
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.placement_deviation,
                remediation="regenerate the RunPlan or use a backend that enforces its device map",
            ) from exc
        except TrainerError as exc:
            # A clean "can't run this request" (runtime/deps/GPU missing, bad config) — not a crash.
            raise RunnerFailure(
                str(exc),
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                stage=StageMarker.env_loaded,
                remediation="run 'corpus-studio train-check' to see what's missing",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — classify the runtime failure, don't leak it as FAIL
            taxonomy, remediation = classify_training_error(exc)
            raise RunnerFailure(
                str(exc) or type(exc).__name__,
                taxonomy=taxonomy,
                remediation=remediation,
            ) from exc
        finally:
            # Record whatever the watchdog measured — on EVERY terminal path. A failed/cancelled run
            # that spilled is the richest diagnostic; capturing it only on success would discard that.
            # But proven=succeeded so a FAILED run never gets a NATIVE_SAFE "fit proven" verdict from
            # its partial peak (a spill still classifies honestly). The supervisor writes ctx.final_fit
            # onto the manifest for any terminal state.
            ctx.final_fit = watchdog.measured_fit(proven=succeeded)
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

        for checkpoint in result.checkpoints:
            ctx.emit_log(f"checkpoint: {checkpoint}")
        expected_output = Path(config.output_dir).resolve(strict=False)
        observed_output = Path(result.output_dir).resolve(strict=False)
        observed_adapter = Path(result.adapter_path).resolve(strict=False)
        if observed_output != expected_output or observed_adapter != expected_output:
            raise RunnerFailure(
                "trainer output deviated from the sealed run-scoped adapter directory",
                taxonomy=FailureTaxonomy.CHECKPOINT_FAILURE,
                stage=StageMarker.export,
                remediation="use the exact backend adapter for this execution contract",
            )

        from corpus_studio.training.artifact_registry import (  # noqa: PLC0415
            compute_weight_content_hash,
        )

        content_hash = compute_weight_content_hash(result.adapter_path)
        if content_hash is None:
            raise RunnerFailure(
                "training returned without readable adapter weight bytes",
                taxonomy=FailureTaxonomy.CHECKPOINT_FAILURE,
                stage=StageMarker.export,
                remediation="inspect the trainer output and rerun from a new derived plan",
            )
        ctx.emit_stage(StageMarker.export, f"adapter saved: {result.adapter_path}")
        artifact = ProducedArtifact(
            artifact_id=f"{ctx.run_id}-adapter-{content_hash[:12]}",
            kind="adapter",
            path=result.adapter_path,
        )
        ctx.emit_artifact(artifact)
        return [artifact]

    def _resolve_trainer(self, plan: RunPlan) -> tuple[TrainerFn, str]:
        """Resolve only a current backend manifest with an exact execution adapter."""
        from corpus_studio.platform.backends import (  # noqa: PLC0415
            backend_manifest_ref,
            get_backend,
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
            ExecutionConfigurationError,
            run_scoped_training_output,
            verify_execution_configuration_hash,
            verify_execution_inputs,
            verify_execution_objective,
        )
        from corpus_studio.training.trainer import train_config_from_resolved  # noqa: PLC0415

        execution = plan.resolved_execution
        if execution is None:
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
        try:
            verify_execution_inputs(execution)
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
