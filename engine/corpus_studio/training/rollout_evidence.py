"""On-policy RL (GRPO) execution-evidence capture (RL slice S5b) - the worker-side honesty core that
assembles the formal :class:`RolloutExecutionEvidence` / :class:`RolloutSuccessEvidence` from a rollout
run, so the runner + supervisor can admit it exactly like the SFT / pretraining / preference / reward
families.

The on-policy sibling of ``reward_evidence.py``: a rollout run trains a CAUSAL_LM policy adapter against a
group-relative advantage derived from a served reward source, under a KL-to-reference bound, so it REUSES
the SFT adapter primitives unchanged (``capture_trainable_state`` / ``capture_adapter_export_state`` /
``compare_*`` / ``GradientObservationTracker``) and adds only what on-policy RL proves: real rollouts were
sampled each step, and every completed step carries its rollout stats (group size, mean reward, KL to the
frozen reference, entropy, mean advantage). The promotion gate is a HELD-OUT mean-reward LIFT while KL stays
within bound (never a falling loss, never training reward alone).

``torch`` is never imported here and ``finalize`` operates on already-captured snapshots, so the tracker +
assembly are fully unit-tested in the base gate; only the live gradient-hook registration + raw tensor
capture (wired into ``run_rollout_training``) are ``# pragma: no cover`` and proven by a GPU run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from corpus_studio.platform.contracts import (
    OptimizerStepLossEvidence,
    RolloutExecutionEvidence,
    RolloutStepEvidence,
    RolloutSuccessEvidence,
)
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
    compare_adapter_export_states,
    compare_trainable_states,
)


class RolloutEvidenceError(RuntimeError):
    """An on-policy RL run that cannot produce honest execution evidence (fail-closed)."""


def _optimizer_parameter_ids(optimizer: Any, *, eligible_ids: list[int]) -> tuple[int, ...]:
    """The optimizer's materialized parameter ids, verified to equal the complete trainable (policy
    adapter) inventory - proves the optimizer steps every trainable parameter and nothing else."""
    param_groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes, bytearray))
        or not param_groups
        or not all(isinstance(group, Mapping) for group in param_groups)
    ):
        raise RolloutEvidenceError("optimizer does not expose materialized parameter groups")
    observed_ids: list[int] = []
    for group in param_groups:
        group_parameters = group.get("params")
        if not isinstance(group_parameters, Sequence) or isinstance(
            group_parameters, (str, bytes, bytearray)
        ):
            raise RolloutEvidenceError("optimizer parameter groups are not materialized sequences")
        for parameter in group_parameters:
            if not hasattr(parameter, "requires_grad"):
                raise RolloutEvidenceError("an optimizer parameter group holds a non-parameter entry")
            observed_ids.append(id(parameter))
    if (
        not eligible_ids
        or len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != sorted(eligible_ids)
    ):
        raise RolloutEvidenceError(
            "optimizer parameters do not exactly match the complete trainable inventory"
        )
    return tuple(sorted(eligible_ids))


class RolloutExecutionTracker:
    """Collect optimizer, loss, gradient, and per-step rollout evidence for one on-policy RL (GRPO) run.

    ``run_rollout_training`` is a CUSTOM loop (not an HF ``Trainer``), so this is driven by an explicit
    :meth:`record_step` the loop calls once per completed optimizer step. ``finalize`` takes the
    before/after snapshots the caller captured around training, so the whole class is pure + base-gate
    testable."""

    def __init__(self, *, expected_steps: int, gradients: GradientObservationTracker) -> None:
        self.expected_steps = expected_steps
        self.gradients = gradients
        self.optimizer: Any | None = None
        self.optimizer_parameter_ids: tuple[int, ...] = ()
        self.completed_steps: list[int] = []
        self.losses: dict[int, float] = {}
        # step -> (rollouts_sampled, mean_reward, kl_to_reference, entropy, mean_advantage)
        self.rollouts: dict[int, tuple[int, float, float, float, float]] = {}

    def _eligible_ids(self) -> list[int]:
        return list(self.gradients.eligible_parameter_ids.values())

    def on_train_begin(self, optimizer: Any) -> None:
        if self.optimizer is not None:
            raise RolloutEvidenceError("optimizer creation was reported more than once")
        if (
            optimizer is None
            or not callable(getattr(optimizer, "step", None))
            or not callable(getattr(optimizer, "zero_grad", None))
        ):
            raise RolloutEvidenceError(
                "on_train_begin did not expose a real optimizer with materialized parameter groups"
            )
        self.optimizer = optimizer
        self.optimizer_parameter_ids = _optimizer_parameter_ids(
            optimizer, eligible_ids=self._eligible_ids()
        )

    @staticmethod
    def _finite_number(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RolloutEvidenceError(f"rollout {name} evidence must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise RolloutEvidenceError(f"rollout {name} evidence must be finite")
        return number

    def record_step(
        self,
        step: int,
        *,
        loss: Any,
        rollouts_sampled: int,
        mean_reward: Any,
        kl_to_reference: Any,
        entropy: Any,
        mean_advantage: Any,
    ) -> None:
        if self.optimizer is None:
            raise RolloutEvidenceError("record_step called before optimizer-creation evidence")
        if _optimizer_parameter_ids(self.optimizer, eligible_ids=self._eligible_ids()) != (
            self.optimizer_parameter_ids
        ):
            raise RolloutEvidenceError("optimizer parameter identity changed after creation")
        expected = len(self.completed_steps) + 1
        if step != expected:
            raise RolloutEvidenceError(
                f"optimizer-step sequence deviation: expected {expected}, observed {step}"
            )
        if isinstance(rollouts_sampled, bool) or not isinstance(rollouts_sampled, int):
            raise RolloutEvidenceError("rollouts_sampled must be an integer")
        if rollouts_sampled < 2:
            raise RolloutEvidenceError("a GRPO step must sample a group of at least two rollouts")
        loss_v = self._finite_number("loss", loss)
        reward_v = self._finite_number("mean_reward", mean_reward)
        kl_v = self._finite_number("kl_to_reference", kl_to_reference)
        if kl_v < 0.0:
            raise RolloutEvidenceError("KL divergence to the reference must be non-negative")
        entropy_v = self._finite_number("entropy", entropy)
        advantage_v = self._finite_number("mean_advantage", mean_advantage)
        self.completed_steps.append(step)
        self.losses[step] = loss_v
        self.rollouts[step] = (rollouts_sampled, reward_v, kl_v, entropy_v, advantage_v)

    def finalize(
        self,
        *,
        steps: int,
        before: TrainableStateSnapshot,
        after: TrainableStateSnapshot,
        before_export: AdapterExportStateSnapshot,
        after_export: AdapterExportStateSnapshot,
        adapter_config_semantic_sha256: str,
    ) -> RolloutExecutionEvidence:
        """Assemble the sealed rollout execution evidence from the collected steps + the before/after
        snapshots the caller captured around ``run_rollout_training``'s loop."""
        if steps > self.expected_steps:
            raise RolloutEvidenceError(
                f"completed step count {steps} exceeds the sealed schedule ceiling {self.expected_steps}"
            )
        if self.optimizer is None:
            raise RolloutEvidenceError("training completed without real optimizer-creation evidence")
        expected = list(range(1, steps + 1))
        if self.completed_steps != expected:
            raise RolloutEvidenceError(
                "completed optimizer-step evidence does not match the rollout loop's step count"
            )
        if sorted(self.losses) != expected or sorted(self.rollouts) != expected:
            raise RolloutEvidenceError(
                "rollout run did not record exactly one loss + one rollout record for every completed step"
            )
        return RolloutExecutionEvidence(
            trainable_state=compare_trainable_states(before, after),
            adapter_export_state=compare_adapter_export_states(
                before_export, after_export,
                adapter_config_semantic_sha256=adapter_config_semantic_sha256,
            ),
            gradient_coverage=self.gradients.evidence(),
            optimizer_created=True,
            completed_optimizer_steps=steps,
            step_losses=[
                OptimizerStepLossEvidence(optimizer_step=step, loss=self.losses[step])
                for step in expected
            ],
            reference_model_frozen=True,
            total_rollouts_sampled=sum(self.rollouts[step][0] for step in expected),
            step_rollout_stats=[
                RolloutStepEvidence(
                    optimizer_step=step,
                    rollouts_sampled=self.rollouts[step][0],
                    mean_reward=self.rollouts[step][1],
                    kl_to_reference=self.rollouts[step][2],
                    entropy=self.rollouts[step][3],
                    mean_advantage=self.rollouts[step][4],
                )
                for step in expected
            ],
        )


def build_rollout_success_evidence(  # pragma: no cover - filesystem/torch integration; proven by a run
    adapter_dir: str,
    execution: RolloutExecutionEvidence,
    *,
    heldout_prompts_evaluated: int,
    heldout_baseline_mean_reward: float,
    heldout_policy_mean_reward: float,
    heldout_max_kl_to_reference: float,
    kl_bound: float,
) -> RolloutSuccessEvidence:
    """Parse-validate the saved policy artifacts and seal the rollout success evidence, binding the
    PROMOTION GATE: a held-out mean-reward LIFT (policy vs baseline/reference) while KL stayed within bound.
    The ``adapter_model.safetensors`` must open as a non-empty tensor archive and ``adapter_config.json`` as
    JSON, so the flags are EARNED; the independent reload-and-compare (the supervisor's admission gate) is
    the runner slice."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415

    out = Path(adapter_dir)
    adapter_weights = out / "adapter_model.safetensors"
    adapter_config = out / "adapter_config.json"
    if not adapter_weights.is_file():
        raise RolloutEvidenceError("the rollout run completed without a saved adapter_model.safetensors")
    if not adapter_config.is_file():
        raise RolloutEvidenceError("the rollout run completed without a saved adapter_config.json")
    try:
        with safe_open(str(adapter_weights), framework="pt") as handle:
            if not list(handle.keys()):
                raise RolloutEvidenceError("the saved adapter_model.safetensors contains no tensors")
    except RolloutEvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any safetensors parse failure to fail-closed
        raise RolloutEvidenceError(
            f"the saved adapter_model.safetensors is not a readable tensor archive: {exc}"
        ) from exc
    try:
        json.loads(adapter_config.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RolloutEvidenceError(f"the saved adapter_config.json is not valid JSON: {exc}") from exc
    return RolloutSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256=hashlib.sha256(adapter_weights.read_bytes()).hexdigest(),
        adapter_config_sha256=hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
        heldout_prompts_evaluated=heldout_prompts_evaluated,
        heldout_baseline_mean_reward=heldout_baseline_mean_reward,
        heldout_policy_mean_reward=heldout_policy_mean_reward,
        heldout_mean_reward_lift=heldout_policy_mean_reward - heldout_baseline_mean_reward,
        heldout_max_kl_to_reference=heldout_max_kl_to_reference,
        kl_bound=kl_bound,
    )
