"""Real runners that plug into the platform run supervisor — platform slice 4.

The :class:`~corpus_studio.platform.supervisor.EchoRunner` proves the harness with zero heavy deps;
this module adds the runner that executes an ACTUAL training run. :class:`TrainingRunner` wraps
``training.trainer.run_training``: it reads the trainer config from the RunPlan's
``training_config_snapshot``, adapts the ``(step, total, loss)`` progress into ``RunEvent`` metrics,
cooperatively aborts on cancel, classifies a missing runtime as an ``ENVIRONMENT_FAILURE`` (never a
crash), and returns the produced adapter artifact.

The heavy stack (torch/transformers/…) is lazy-imported **inside** ``run()`` — importing this module
pulls only the platform contracts + the (import-light) trainer module, so the dependency-light
boundary holds. ``cpu_toy=True`` runs the tiny CPU smoke path (needs the ``[train]`` extra but no
GPU); the real GPU QLoRA path can only be user-smoke-tested.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import ValidationError

from corpus_studio.platform.contracts import RunPlan
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.supervisor import (
    ProducedArtifact,
    RunCancelled,
    RunContext,
    RunnerFailure,
    demo_run_plan,
)

if TYPE_CHECKING:
    from corpus_studio.training.trainer import TrainRunConfig


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

    ``cpu_toy=True`` forces the tiny CPU smoke path; ``max_steps`` caps optimizer steps. A missing
    training runtime surfaces as an ``ENVIRONMENT_FAILURE`` FailureRecord; a plan with no / an invalid
    ``training_config_snapshot`` surfaces as ``UNSUPPORTED_CONFIGURATION``."""

    def __init__(self, *, cpu_toy: bool = False, max_steps: int | None = None) -> None:
        self.cpu_toy = cpu_toy
        self.max_steps = max_steps
        self.name = "cpu_toy" if cpu_toy else "training"

    def run(self, ctx: RunContext) -> Sequence[ProducedArtifact]:
        try:
            # Lazy: keeps importing this module dependency-light. The trainer module itself is
            # import-light (torch is lazy-imported inside run_training), so this rarely fails —
            # a truly-missing runtime instead surfaces as the TrainerError below.
            from corpus_studio.training.trainer import (  # noqa: PLC0415
                TrainerError,
                run_training,
            )
        except ImportError as exc:  # pragma: no cover - defensive; the trainer module imports cleanly
            raise RunnerFailure(
                f"the training runtime module could not be imported: {exc}",
                taxonomy=FailureTaxonomy.ENVIRONMENT_FAILURE,
                stage=StageMarker.env_loaded,
                remediation="install the training extra: pip install '.[train]'",
            ) from exc

        config = self._resolve_config(ctx.plan)
        ctx.emit_stage(StageMarker.process_start, f"training run: {config.base_model}")

        def _progress(step: int, total: int, loss: float | None) -> None:
            if ctx.cancelled:
                raise _CancelTraining
            ctx.emit_metric(optimizer_step=step, loss=loss, message=f"[{step}/{total}] step")

        try:
            result = run_training(config, progress_callback=_progress)
        except _CancelTraining:
            raise RunCancelled from None
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

        for checkpoint in result.checkpoints:
            ctx.emit_log(f"checkpoint: {checkpoint}")
        ctx.emit_stage(StageMarker.export, f"adapter saved: {result.adapter_path}")
        artifact = ProducedArtifact(
            artifact_id=f"{ctx.run_id}-adapter",
            kind="adapter",
            path=result.adapter_path,
        )
        ctx.emit_artifact(artifact)
        return [artifact]

    def _resolve_config(self, plan: RunPlan) -> TrainRunConfig:
        """Build the trainer config from the plan's rendered snapshot, applying the runner's
        cpu_toy / max_steps overrides. An empty or invalid snapshot is a clean, classified failure."""
        from corpus_studio.training.trainer import TrainRunConfig  # noqa: PLC0415

        snapshot = dict(plan.training_config_snapshot)
        if not snapshot:
            raise RunnerFailure(
                "the RunPlan carries no training_config_snapshot to execute",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            )
        if self.cpu_toy:
            snapshot["cpu_toy"] = True
        if self.max_steps is not None:
            snapshot["max_steps"] = self.max_steps
        try:
            return TrainRunConfig.model_validate(snapshot)
        except ValidationError as exc:
            raise RunnerFailure(
                f"the training_config_snapshot is not a valid trainer config: {exc}",
                taxonomy=FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
                stage=StageMarker.env_loaded,
            ) from exc


def demo_training_plan(plan_id: str = "demo-cpu-toy") -> RunPlan:
    """A minimal, valid :class:`RunPlan` carrying a tiny cpu-toy trainer config in
    ``training_config_snapshot`` — so ``platform-run --demo --runner cpu_toy`` and the tests exercise
    the :class:`TrainingRunner`. Actually training it needs the ``[train]`` extra (a tiny random-weight
    model, a few CPU steps); without it the run cleanly classifies ``ENVIRONMENT_FAILURE``. Rebuilt
    via ``model_validate`` (not ``model_copy``) so the string fields coerce back to their enums."""
    body = demo_run_plan(plan_id).model_dump(mode="json")
    body["task_type"] = "sft"
    body["base_model"] = "hf-internal-testing/tiny-random-gpt2"
    body["training_config_snapshot"] = {
        "base_model": "hf-internal-testing/tiny-random-gpt2",
        "dataset_path": "examples/datasets/instruction/train.jsonl",
        "dataset_format": "instruction",
        "sequence_len": 64,
        "lora_r": 4,
        "lora_alpha": 8,
        "max_steps": 2,
    }
    return RunPlan.model_validate(body)
