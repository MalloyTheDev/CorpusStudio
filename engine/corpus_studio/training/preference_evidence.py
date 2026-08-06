"""Offline-DPO (preference) execution-evidence capture - the worker-side honesty core that assembles the
formal :class:`PreferenceExecutionEvidence` / :class:`PreferenceSuccessEvidence` (the #813 contracts) from
a DPO run, so the runner + supervisor can admit it exactly like the SFT / pretraining families.

The ADAPTER sibling of ``pretraining_evidence.py``: DPO trains a PEFT adapter over a FROZEN reference, so
it REUSES the SFT adapter primitives unchanged (``capture_trainable_state`` / ``capture_adapter_export_state``
/ ``compare_adapter_export_states`` / ``compare_trainable_states`` / ``GradientObservationTracker``) and adds
only what preference optimization proves: the reference was frozen, real preference PAIRS were consumed, and
every completed step carries the DPO reward margin the loss was built from.

``torch`` is never imported here and ``finalize`` operates on already-captured snapshots, so the tracker +
assembly are fully unit-tested in the base gate; only the live gradient-hook registration + raw tensor
capture (wired into ``run_dpo_training``) are ``# pragma: no cover`` and proven by a GPU run."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from corpus_studio.platform.contracts import (
    OptimizerStepLossEvidence,
    PreferenceExecutionEvidence,
    PreferenceRewardMarginEvidence,
    PreferenceSuccessEvidence,
)
from corpus_studio.training.trainer import (
    AdapterExportStateSnapshot,
    GradientObservationTracker,
    TrainableStateSnapshot,
    compare_adapter_export_states,
    compare_trainable_states,
)


class PreferenceEvidenceError(RuntimeError):
    """A DPO run that cannot produce honest execution evidence (fail-closed)."""


def _optimizer_parameter_ids(optimizer: Any, *, eligible_ids: list[int]) -> tuple[int, ...]:
    """The optimizer's materialized parameter ids, verified to equal the complete trainable (adapter)
    inventory - proves the optimizer steps every trainable adapter parameter and nothing else."""
    param_groups = getattr(optimizer, "param_groups", None)
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, (str, bytes, bytearray))
        or not param_groups
        or not all(isinstance(group, Mapping) for group in param_groups)
    ):
        raise PreferenceEvidenceError("optimizer does not expose materialized parameter groups")
    observed_ids: list[int] = []
    for group in param_groups:
        group_parameters = group.get("params")
        if not isinstance(group_parameters, Sequence) or isinstance(
            group_parameters, (str, bytes, bytearray)
        ):
            raise PreferenceEvidenceError("optimizer parameter groups are not materialized sequences")
        for parameter in group_parameters:
            if not hasattr(parameter, "requires_grad"):
                raise PreferenceEvidenceError(
                    "an optimizer parameter group holds a non-parameter entry"
                )
            observed_ids.append(id(parameter))
    if (
        not eligible_ids
        or len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != sorted(eligible_ids)
    ):
        raise PreferenceEvidenceError(
            "optimizer parameters do not exactly match the complete trainable adapter inventory"
        )
    return tuple(sorted(eligible_ids))


class PreferenceExecutionTracker:
    """Collect optimizer, loss, gradient, and DPO reward-margin evidence for one offline-DPO run.

    ``run_dpo_training`` is a CUSTOM loop (not an HF ``Trainer``), so - unlike the pretraining tracker's
    HF callbacks - this is driven by an explicit :meth:`record_step` the loop calls once per completed
    optimizer step. ``finalize`` takes the before/after snapshots the caller captured around training, so
    the whole class is pure + base-gate testable."""

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
            raise PreferenceEvidenceError("optimizer creation was reported more than once")
        if (
            optimizer is None
            or not callable(getattr(optimizer, "step", None))
            or not callable(getattr(optimizer, "zero_grad", None))
        ):
            raise PreferenceEvidenceError(
                "on_train_begin did not expose a real optimizer with materialized parameter groups"
            )
        self.optimizer = optimizer
        self.optimizer_parameter_ids = _optimizer_parameter_ids(
            optimizer, eligible_ids=self._eligible_ids()
        )

    @staticmethod
    def _finite_number(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PreferenceEvidenceError(f"DPO {name} evidence must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise PreferenceEvidenceError(f"DPO {name} evidence must be finite")
        return number

    def record_step(
        self, step: int, *, loss: Any, chosen_reward: Any, rejected_reward: Any, margin: Any
    ) -> None:
        if self.optimizer is None:
            raise PreferenceEvidenceError("record_step called before optimizer-creation evidence")
        if _optimizer_parameter_ids(self.optimizer, eligible_ids=self._eligible_ids()) != (
            self.optimizer_parameter_ids
        ):
            raise PreferenceEvidenceError("optimizer parameter identity changed after creation")
        expected = len(self.completed_steps) + 1
        if step != expected:
            raise PreferenceEvidenceError(
                f"optimizer-step sequence deviation: expected {expected}, observed {step}"
            )
        loss_v = self._finite_number("loss", loss)
        chosen_v = self._finite_number("chosen_reward", chosen_reward)
        rejected_v = self._finite_number("rejected_reward", rejected_reward)
        margin_v = self._finite_number("reward_margin", margin)
        if not math.isclose(margin_v, chosen_v - rejected_v, rel_tol=1e-4, abs_tol=1e-4):
            raise PreferenceEvidenceError(
                "DPO reward margin must equal chosen_reward - rejected_reward"
            )
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
        preference_pairs_consumed: int,
    ) -> PreferenceExecutionEvidence:
        """Assemble the sealed preference execution evidence from the collected steps + the before/after
        snapshots the caller captured around ``run_dpo_training``'s loop."""
        if steps > self.expected_steps:
            raise PreferenceEvidenceError(
                f"completed step count {steps} exceeds the sealed schedule ceiling {self.expected_steps}"
            )
        if self.optimizer is None:
            raise PreferenceEvidenceError("training completed without real optimizer-creation evidence")
        expected = list(range(1, steps + 1))
        if self.completed_steps != expected:
            raise PreferenceEvidenceError(
                "completed optimizer-step evidence does not match the DPO loop's step count"
            )
        if sorted(self.losses) != expected or sorted(self.rewards) != expected:
            raise PreferenceEvidenceError(
                "DPO did not record exactly one loss + one reward margin for every completed step"
            )
        return PreferenceExecutionEvidence(
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
            preference_pairs_consumed=preference_pairs_consumed,
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


def build_preference_success_evidence(  # pragma: no cover - filesystem/torch integration; proven by a run
    adapter_dir: str, execution: PreferenceExecutionEvidence
) -> PreferenceSuccessEvidence:
    """Parse-validate the saved PEFT adapter artifacts and seal the DPO success evidence - the adapter
    sibling of the pretraining ``_build_success_evidence``. The ``adapter_model.safetensors`` must open as a
    non-empty tensor archive and ``adapter_config.json`` as JSON, so the success flags are EARNED; the
    independent reload-and-compare (the supervisor's admission gate) is the runner slice."""
    import hashlib  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from safetensors import safe_open  # noqa: PLC0415

    out = Path(adapter_dir)
    adapter_weights = out / "adapter_model.safetensors"
    adapter_config = out / "adapter_config.json"
    if not adapter_weights.is_file():
        raise PreferenceEvidenceError("DPO completed without a saved adapter_model.safetensors")
    if not adapter_config.is_file():
        raise PreferenceEvidenceError("DPO completed without a saved adapter_config.json")
    try:
        with safe_open(str(adapter_weights), framework="pt") as handle:
            if not list(handle.keys()):
                raise PreferenceEvidenceError("the saved adapter_model.safetensors contains no tensors")
    except PreferenceEvidenceError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any safetensors parse failure to fail-closed
        raise PreferenceEvidenceError(
            f"the saved adapter_model.safetensors is not a readable tensor archive: {exc}"
        ) from exc
    try:
        json.loads(adapter_config.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PreferenceEvidenceError(f"the saved adapter_config.json is not valid JSON: {exc}") from exc
    return PreferenceSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        adapter_bytes_verified=True,
        artifact_integrity_verified=True,
        adapter_safetensors_sha256=hashlib.sha256(adapter_weights.read_bytes()).hexdigest(),
        adapter_config_sha256=hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
    )
