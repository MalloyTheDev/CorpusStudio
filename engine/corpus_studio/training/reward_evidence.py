"""Pairwise reward-model execution-evidence capture (RL slice S5a) - the worker-side honesty core that
assembles the formal :class:`RewardExecutionEvidence` / :class:`RewardSuccessEvidence` from a reward run,
so the runner + supervisor can admit it exactly like the SFT / pretraining / preference families.

The ADAPTER sibling of ``preference_evidence.py``: a reward model trains a SEQ_CLS LoRA score head, so it
REUSES the SFT adapter primitives unchanged (``capture_trainable_state`` / ``capture_adapter_export_state``
/ ``compare_*`` / ``GradientObservationTracker``) and adds only what pairwise reward modeling proves: real
preference PAIRS were consumed and every completed step carries the reward margin
(``score(chosen) - score(rejected)``) the Bradley-Terry loss was built from. Unlike DPO there is NO
reference model - a reward model scores directly - so there is no reference-frozen signal; the promotion
gate is HELD-OUT pairwise ranking accuracy (never a falling loss).

``torch`` is never imported here and ``finalize`` operates on already-captured snapshots, so the tracker +
assembly are fully unit-tested in the base gate; only the live gradient-hook registration + raw tensor
capture (wired into ``run_reward_training``) are ``# pragma: no cover`` and proven by a GPU run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from corpus_studio.platform.contracts import (
    OptimizerStepLossEvidence,
    PreferenceRewardMarginEvidence,
    RewardExecutionEvidence,
    RewardSuccessEvidence,
)
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
    compare_adapter_export_states,
    compare_trainable_states,
)


class RewardEvidenceError(RuntimeError):
    """A reward run that cannot produce honest execution evidence (fail-closed)."""


def _optimizer_parameter_ids(optimizer: Any, *, eligible_ids: list[int]) -> tuple[int, ...]:
    """The optimizer's materialized parameter ids, verified to equal the complete trainable (adapter +
    score head) inventory - proves the optimizer steps every trainable parameter and nothing else."""
    param_groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes, bytearray))
        or not param_groups
        or not all(isinstance(group, Mapping) for group in param_groups)
    ):
        raise RewardEvidenceError("optimizer does not expose materialized parameter groups")
    observed_ids: list[int] = []
    for group in param_groups:
        group_parameters = group.get("params")
        if not isinstance(group_parameters, Sequence) or isinstance(
            group_parameters, (str, bytes, bytearray)
        ):
            raise RewardEvidenceError("optimizer parameter groups are not materialized sequences")
        for parameter in group_parameters:
            if not hasattr(parameter, "requires_grad"):
                raise RewardEvidenceError("an optimizer parameter group holds a non-parameter entry")
            observed_ids.append(id(parameter))
    if (
        not eligible_ids
        or len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != sorted(eligible_ids)
    ):
        raise RewardEvidenceError(
            "optimizer parameters do not exactly match the complete trainable inventory"
        )
    return tuple(sorted(eligible_ids))


class RewardExecutionTracker:
    """Collect optimizer, loss, gradient, and reward-margin evidence for one pairwise reward-model run.

    ``run_reward_training`` is a CUSTOM loop (not an HF ``Trainer``), so this is driven by an explicit
    :meth:`record_step` the loop calls once per completed optimizer step. ``finalize`` takes the
    before/after snapshots the caller captured around training, so the whole class is pure + base-gate
    testable. The recorded chosen/rejected values are the EXPLICIT SEQ_CLS scores."""

    def __init__(self, *, expected_steps: int, gradients: GradientObservationTracker) -> None:
        self.expected_steps = expected_steps
        self.gradients = gradients
        self.optimizer: Any | None = None
        self.optimizer_parameter_ids: tuple[int, ...] = ()
        self.completed_steps: list[int] = []
        self.losses: dict[int, float] = {}
        self.rewards: dict[int, tuple[float, float, float]] = {}

    def _eligible_ids(self) -> list[int]:
        return list(self.gradients.eligible_parameter_ids.values())

    def on_train_begin(self, optimizer: Any) -> None:
        if self.optimizer is not None:
            raise RewardEvidenceError("optimizer creation was reported more than once")
        if (
            optimizer is None
            or not callable(getattr(optimizer, "step", None))
            or not callable(getattr(optimizer, "zero_grad", None))
        ):
            raise RewardEvidenceError(
                "on_train_begin did not expose a real optimizer with materialized parameter groups"
            )
        self.optimizer = optimizer
        self.optimizer_parameter_ids = _optimizer_parameter_ids(
            optimizer, eligible_ids=self._eligible_ids()
        )

    @staticmethod
    def _finite_number(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RewardEvidenceError(f"reward {name} evidence must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise RewardEvidenceError(f"reward {name} evidence must be finite")
        return number

    def record_step(
        self, step: int, *, loss: Any, chosen_reward: Any, rejected_reward: Any, margin: Any
    ) -> None:
        if self.optimizer is None:
            raise RewardEvidenceError("record_step called before optimizer-creation evidence")
        if _optimizer_parameter_ids(self.optimizer, eligible_ids=self._eligible_ids()) != (
            self.optimizer_parameter_ids
        ):
            raise RewardEvidenceError("optimizer parameter identity changed after creation")
        expected = len(self.completed_steps) + 1
        if step != expected:
            raise RewardEvidenceError(
                f"optimizer-step sequence deviation: expected {expected}, observed {step}"
            )
        loss_v = self._finite_number("loss", loss)
        chosen_v = self._finite_number("chosen_reward", chosen_reward)
        rejected_v = self._finite_number("rejected_reward", rejected_reward)
        margin_v = self._finite_number("reward_margin", margin)
        if not math.isclose(margin_v, chosen_v - rejected_v, rel_tol=1e-4, abs_tol=1e-4):
            raise RewardEvidenceError("reward margin must equal chosen_reward - rejected_reward")
        self.completed_steps.append(step)
        self.losses[step] = loss_v
        self.rewards[step] = (chosen_v, rejected_v, margin_v)

    def finalize(
        self,
        *,
        steps: int,
        before: TrainableStateSnapshot,
        after: TrainableStateSnapshot,
        before_export: AdapterExportStateSnapshot,
        after_export: AdapterExportStateSnapshot,
        adapter_config_semantic_sha256: str,
        reward_pairs_consumed: int,
    ) -> RewardExecutionEvidence:
        """Assemble the sealed reward execution evidence from the collected steps + the before/after
        snapshots the caller captured around ``run_reward_training``'s loop."""
        if steps > self.expected_steps:
            raise RewardEvidenceError(
                f"completed step count {steps} exceeds the sealed schedule ceiling {self.expected_steps}"
            )
        if self.optimizer is None:
            raise RewardEvidenceError("training completed without real optimizer-creation evidence")
        expected = list(range(1, steps + 1))
        if self.completed_steps != expected:
            raise RewardEvidenceError(
                "completed optimizer-step evidence does not match the reward loop's step count"
            )
        if sorted(self.losses) != expected or sorted(self.rewards) != expected:
            raise RewardEvidenceError(
                "reward run did not record exactly one loss + one reward margin for every completed step"
            )
        return RewardExecutionEvidence(
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
            reward_pairs_consumed=reward_pairs_consumed,
            step_reward_margins=[
                PreferenceRewardMarginEvidence(
                    optimizer_step=step,
                    chosen_reward=self.rewards[step][0],
                    rejected_reward=self.rewards[step][1],
                    margin=self.rewards[step][2],
                )
                for step in expected
            ],
        )


def build_reward_success_evidence(  # pragma: no cover - filesystem/torch integration; proven by a run
    adapter_dir: str,
    execution: RewardExecutionEvidence,
    *,
    heldout_pairwise_accuracy: float,
    heldout_pairs_evaluated: int,
) -> RewardSuccessEvidence:
    """Parse-validate the saved reward artifacts and seal the reward success evidence, binding the
    HELD-OUT pairwise ranking accuracy as the promotion gate. The ``adapter_model.safetensors`` must open
    as a non-empty tensor archive and ``adapter_config.json`` as JSON, so the flags are EARNED; the
    independent reload-and-compare (the supervisor's admission gate) is the runner slice."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415

    out = Path(adapter_dir)
    adapter_weights = out / "adapter_model.safetensors"
    adapter_config = out / "adapter_config.json"
    if not adapter_weights.is_file():
        raise RewardEvidenceError("the reward run completed without a saved adapter_model.safetensors")
    if not adapter_config.is_file():
        raise RewardEvidenceError("the reward run completed without a saved adapter_config.json")
    try:
        with safe_open(str(adapter_weights), framework="pt") as handle:
            if not list(handle.keys()):
                raise RewardEvidenceError("the saved adapter_model.safetensors contains no tensors")
    except RewardEvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any safetensors parse failure to fail-closed
        raise RewardEvidenceError(
            f"the saved adapter_model.safetensors is not a readable tensor archive: {exc}"
        ) from exc
    try:
        json.loads(adapter_config.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RewardEvidenceError(f"the saved adapter_config.json is not valid JSON: {exc}") from exc
    return RewardSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256=hashlib.sha256(adapter_weights.read_bytes()).hexdigest(),
        adapter_config_sha256=hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
        heldout_pairwise_accuracy=heldout_pairwise_accuracy,
        heldout_pairs_evaluated=heldout_pairs_evaluated,
    )
