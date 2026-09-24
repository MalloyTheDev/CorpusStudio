"""Torch-free fixtures for the subprocess parent's variant success admission (#860).

They write real-shaped saved exports (a hand-built Safetensors file plus its config) and the
worker-proposed success evidence that matches those bytes for every non-SFT execution variant. The
tensor-state identity is computed with the same canonical function the trainer uses, so the parent's
dependency-light re-verification sees exactly what a genuine worker would have saved.

This module is imported by ``test_subprocess_variant_admission.py`` and by the child worker script
that module spawns (``child_worker_script``), so it must stay importable without pytest or torch.
Not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

from corpus_studio.platform import contracts as C
from corpus_studio.platform.parameter_accounting import canonical_tensor_state_sha256

_A, _B, _C = "a" * 64, "b" * 64, "c" * 64

ADAPTER_TENSORS = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": 1.0,
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": 2.0,
}
# A reward export is a SEQ_CLS adapter that also carries the trained score head (modules_to_save).
REWARD_TENSORS = {**ADAPTER_TENSORS, "base_model.model.score.weight": 3.0}
MODEL_TENSORS = {"model.embed_tokens.weight": 1.0, "model.layers.0.mlp.up_proj.weight": 2.0}

ADAPTER_LANES = ("preference", "reward", "rollout")
MODEL_LANES = ("full_finetune", "pretraining")


def write_safetensors(path: Path, values: dict[str, float]) -> list[dict[str, Any]]:
    """Write a minimal valid Safetensors file (one F32 scalar per tensor) and return its records."""

    names = sorted(values)
    data = b"".join(struct.pack("<f", values[name]) for name in names)
    header = json.dumps(
        {
            name: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, (i + 1) * 4]}
            for i, name in enumerate(names)
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return [
        {
            "name": name,
            "dtype": "F32",
            "shape": [1],
            "content_sha256": hashlib.sha256(data[i * 4 : (i + 1) * 4]).hexdigest(),
        }
        for i, name in enumerate(names)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common_execution(names: list[str], steps: int) -> dict[str, Any]:
    return {
        "trainable_state": C.TrainableStateChangeEvidence(
            before_sha256=_A,
            after_sha256=_B,
            trainable_tensor_count=len(names),
            trainable_tensor_names=names,
            changed_tensor_count=1,
            changed_tensor_names=[names[0]],
        ),
        "gradient_coverage": C.GradientCoverageEvidence(
            eligible_tensor_count=len(names),
            eligible_tensor_names=names,
            observed_tensor_count=1,
            observed_tensor_names=[names[0]],
        ),
        "optimizer_created": True,
        "completed_optimizer_steps": steps,
        "step_losses": [
            C.OptimizerStepLossEvidence(optimizer_step=step, loss=0.5)
            for step in range(1, steps + 1)
        ],
    }


def _margins(steps: int) -> list[C.PreferenceRewardMarginEvidence]:
    return [
        C.PreferenceRewardMarginEvidence(
            optimizer_step=step, chosen_reward=0.4, rejected_reward=-0.1, margin=0.5
        )
        for step in range(1, steps + 1)
    ]


def write_adapter_export(lane: str, out: Path, steps: int) -> Any:
    """Save a PEFT-shaped adapter export under ``out`` and return the lane's matching evidence."""

    out.mkdir(parents=True, exist_ok=True)
    tensors = REWARD_TENSORS if lane == "reward" else ADAPTER_TENSORS
    records = write_safetensors(out / "adapter_model.safetensors", tensors)
    (out / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 4}), encoding="utf-8"
    )
    # tokenizer.save_pretrained output next to the adapter; not a weight payload.
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    names = sorted(tensors)
    export = C.AdapterExportStateEvidence(
        before_sha256=_C,
        after_sha256=canonical_tensor_state_sha256(records),
        tensor_count=len(names),
        tensor_names=names,
        changed_tensor_count=1,
        changed_tensor_names=[names[0]],
        adapter_config_semantic_sha256=_A,
    )
    digests = {
        "output_path_verified": True,
        "adapter_bytes_verified": True,
        "artifact_integrity_verified": True,
        "adapter_safetensors_sha256": _sha256(out / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256(out / "adapter_config.json"),
    }
    common = _common_execution(names, steps)
    if lane == "preference":
        return C.PreferenceSuccessEvidence(
            execution=C.PreferenceExecutionEvidence(
                **common,
                adapter_export_state=export,
                reference_model_frozen=True,
                preference_pairs_consumed=4,
                step_reward_margins=_margins(steps),
            ),
            **digests,
        )
    if lane == "reward":
        return C.RewardSuccessEvidence(
            execution=C.RewardExecutionEvidence(
                **common,
                adapter_export_state=export,
                reward_pairs_consumed=4,
                step_reward_margins=_margins(steps),
            ),
            heldout_pairwise_accuracy=1.0,
            heldout_pairs_evaluated=2,
            **digests,
        )
    if lane == "rollout":
        return C.RolloutSuccessEvidence(
            execution=C.RolloutExecutionEvidence(
                **common,
                adapter_export_state=export,
                reference_model_frozen=True,
                total_rollouts_sampled=steps * 4,
                step_rollout_stats=[
                    C.RolloutStepEvidence(
                        optimizer_step=step,
                        rollouts_sampled=4,
                        mean_reward=0.5,
                        kl_to_reference=0.02,
                        entropy=1.5,
                        mean_advantage=0.0,
                    )
                    for step in range(1, steps + 1)
                ],
            ),
            heldout_prompts_evaluated=2,
            heldout_baseline_mean_reward=0.2,
            heldout_policy_mean_reward=0.7,
            heldout_mean_reward_lift=0.5,
            heldout_max_kl_to_reference=0.05,
            kl_bound=0.1,
            **digests,
        )
    raise ValueError(f"not an adapter lane: {lane}")


def write_model_export(out: Path, steps: int) -> C.PretrainingSuccessEvidence:
    """Save a full-model export under ``out`` and return the matching full-model evidence."""

    out.mkdir(parents=True, exist_ok=True)
    records = write_safetensors(out / "model.safetensors", MODEL_TENSORS)
    (out / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    names = sorted(MODEL_TENSORS)
    export = C.FullModelExportStateEvidence(
        before_sha256=_C,
        after_sha256=canonical_tensor_state_sha256(records),
        tensor_count=len(names),
        tensor_names=names,
        changed_tensor_count=1,
        changed_tensor_names=[names[0]],
        model_config_semantic_sha256=_A,
    )
    return C.PretrainingSuccessEvidence(
        execution=C.PretrainingExecutionEvidence(
            **_common_execution(names, steps), model_export_state=export
        ),
        output_path_verified=True,
        model_bytes_verified=True,
        artifact_integrity_verified=True,
        model_safetensors_sha256=_sha256(out / "model.safetensors"),
        model_config_sha256=_sha256(out / "config.json"),
    )


def write_lane_export(lane: str, out: Path, steps: int) -> Any:
    if lane in MODEL_LANES:
        return write_model_export(out, steps)
    return write_adapter_export(lane, out, steps)


# The child runs the REAL worker entrypoint (hello -> run_dispatch -> run_worker -> execute_run ->
# the lane runner). Only the lane's ML training function (replaced by a writer of genuine-shaped
# bytes and evidence) and the worker-side torch reload-verify are substituted: this interpreter has
# no torch. The fakes accept the runner's extra keyword arguments (the verified dataset rows, stage
# callbacks) without using them. The subprocess PARENT, which is what the tests exercise, runs
# unmodified.
_CHILD_TEMPLATE = r'''
import sys
sys.path.insert(0, {tests_dir!r})
from pathlib import Path
import _variant_admission_fixtures as fx
import corpus_studio.platform.supervisor as sup
sup._reload_verify_adapter = lambda *a, **k: (True, None)
sup._reload_verify_full_model = lambda *a, **k: (True, None)
LANE = {lane!r}
def _steps(execution):
    return execution.schedule.max_steps or 1
if LANE == "preference":
    import corpus_studio.training.preference_worker as m
    def fake(execution, *, output_dir, **_kwargs):
        ev = fx.write_adapter_export("preference", Path(output_dir), _steps(execution))
        return m.PreferenceRunResult(output_dir=output_dir, success_evidence=ev)
    m.run_preference = fake
elif LANE == "reward":
    import corpus_studio.training.reward_worker as m
    def fake(execution, *, output_dir, **_kwargs):
        ev = fx.write_adapter_export("reward", Path(output_dir), _steps(execution))
        return m.RewardRunResult(output_dir=output_dir, success_evidence=ev)
    m.run_reward = fake
elif LANE == "full_finetune":
    import corpus_studio.training.full_finetune_trainer as m
    def fake(execution, *, output_dir, **_kwargs):
        ev = fx.write_model_export(Path(output_dir), _steps(execution))
        return m.FullFinetuneRunResult(output_dir=output_dir, success_evidence=ev)
    m.run_full_finetune = fake
else:
    import corpus_studio.training.pretraining_trainer as m
    def fake(execution, *, corpus_root=".", output_dir=None, **_kwargs):
        ev = fx.write_model_export(Path(output_dir), _steps(execution))
        return m.PretrainResult(
            output_dir=output_dir, cpu_toy=True, steps=ev.execution.completed_optimizer_steps,
            vocab_size=32, num_blocks=1, coverage_ratio=1.0, tokenizer_source="train",
            execution_evidence=ev,
        )
    m.run_pretraining = fake
from corpus_studio.platform import worker
sys.argv = ["corpus-studio-worker", "--runner", {runner!r}, *{identity!r}]
worker.main()
'''


def child_worker_script(runner: str, identity_argv: list[str]) -> str:
    """The ``python -c`` source of a real worker child for ``runner`` (a sealed runner lane)."""

    lane = "pretraining" if runner.startswith("pretraining") else runner
    return _CHILD_TEMPLATE.format(
        tests_dir=str(Path(__file__).resolve().parent),
        lane=lane,
        runner=runner,
        identity=list(identity_argv),
    )
