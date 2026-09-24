"""Planning and runner-admission refusals for sealed loader values the newer workers cannot lower (#863).

The DPO, reward, full-parameter SFT and on-policy RL workers lower the sealed model/tokenizer identity,
placement, precision and attention policy through the adapter SFT lane's loader helpers. Whatever those
helpers cannot lower exactly is refused by ``execution_config.verify_loader_policy_supported`` at planning,
before dispatch, and in the worker; the runners also re-hash local model/tokenizer bindings before the
dataset is read or the worker module is imported, and record the admitted identity as structured
evidence. Every sealed configuration here comes from the real planner, then (where a shape is not
planner-reachable) is edited and re-validated against its own contract and re-sealed, so each refused
shape is one the contract itself accepts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_platform_planner import _REWARD_BASE, _REWARD_BRINGUP, _plan, _profile, _report

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import (
    EnvHost,
    EnvironmentProfile,
    ExecutionInputBinding,
    GpuDevice,
    ResolvedFullFinetuneExecutionConfiguration,
    ResolvedPreferenceExecutionConfiguration,
    ResolvedRewardExecutionConfiguration,
    ResolvedRolloutExecutionConfiguration,
)
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import (
    ExecutionConfigurationError,
    full_finetune_execution_configuration_hash_for,
    preference_execution_configuration_hash_for,
    required_runner_lane,
    reward_execution_configuration_hash_for,
    rollout_execution_configuration_hash_for,
    stable_directory_sha256,
    stable_file_sha256,
    verify_execution_non_dataset_inputs,
    verify_loader_policy_supported,
)
from corpus_studio.platform.planner import PlannerError
from corpus_studio.platform.runners import RolloutRunner, RewardRunner, build_lane_runner
from corpus_studio.platform.supervisor import CancelToken, RunContext, RunnerFailure, execute_run
from corpus_studio.training.trainer import (
    ExecutionPlacementDeviation,
    TrainerEnvironmentError,
    TrainerError,
    TrainingEvidenceError,
)

_REWARD_SOURCE = {
    "reward_source_manifest": str(_REWARD_BRINGUP / "runs/run-reward-sealed-0001/RunManifest.json"),
    "reward_source_plan": str(_REWARD_BRINGUP / "reward-bringup.RunPlan.json"),
}
_LANE_KW: dict[str, dict[str, Any]] = {
    "preference": {"task_type": "preference", "objective_id": "dpo_qlora"},
    "reward": {"task_type": "reward", "objective_id": "reward_model"},
    "full_finetune": {
        "task_type": "sft",
        "adapter_method": "full_finetune",
        "export_format": "merged_safetensors",
    },
    "rollout": {
        "task_type": "grpo",
        "objective_id": "grpo",
        "base_model": _REWARD_BASE,
        **_REWARD_SOURCE,
    },
}
_HASHERS: dict[type[Any], Callable[[Any], str]] = {
    ResolvedPreferenceExecutionConfiguration: preference_execution_configuration_hash_for,
    ResolvedRewardExecutionConfiguration: reward_execution_configuration_hash_for,
    ResolvedFullFinetuneExecutionConfiguration: full_finetune_execution_configuration_hash_for,
    ResolvedRolloutExecutionConfiguration: rollout_execution_configuration_hash_for,
}


def _rows(lane: str) -> list[dict[str, Any]]:
    if lane in {"preference", "reward"}:
        return [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(6)]
    if lane == "rollout":
        return [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(6)]
    return [{"instruction": f"i{i}", "output": "o"} for i in range(6)]


def _sealed_plan(tmp_path: Path, lane: str, *, profile=None, report=None, **extra: Any):
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in _rows(lane)), encoding="utf-8")
    return _plan(
        profile or _profile(cc_major=8),
        report or _report(),
        dataset_path=str(data),
        dataset_content_sha256=stable_file_sha256(data),
        **{**_LANE_KW[lane], **extra},
    )


def _execution(plan) -> Any:
    return (
        plan.resolved_preference_execution
        or plan.resolved_reward_execution
        or plan.resolved_full_finetune_execution
        or plan.resolved_rollout_execution
    )


def _reseal(execution: Any, mutate: Callable[[dict[str, Any]], None]) -> Any:
    """Edit a sealed execution's JSON body, re-validate it against its OWN contract, and re-seal it."""
    payload = execution.model_dump(mode="json")
    mutate(payload)
    draft = type(execution).model_validate({**payload, "configuration_hash": "0" * 64})
    return draft.model_copy(update={"configuration_hash": _HASHERS[type(execution)](draft)})


def _local_model(tmp_path: Path, name: str = "model") -> Path:
    model = tmp_path / name
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"sealed-weights")
    (model / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    return model


def _refused(execution: Any, lane: str) -> str:
    with pytest.raises(ExecutionConfigurationError) as refused:
        verify_loader_policy_supported(execution, lane=lane)  # type: ignore[arg-type]
    message = str(refused.value)
    assert message.isascii()
    return message


# --- the planner's own shapes are exactly what the workers lower ---------------------------------------


@pytest.mark.parametrize(
    ("lane", "report_kw", "plan_kw"),
    [
        ("preference", {}, {}),
        ("preference", {"precisions": ("fp32",)}, {}),
        ("reward", {}, {}),
        ("rollout", {}, {}),
        ("full_finetune", {}, {}),
        ("full_finetune", {"readiness": "cpu_toy_only"}, {"allow_cpu_toy": True}),
    ],
)
def test_every_planner_shape_passes_the_loader_policy_check(tmp_path, lane, report_kw, plan_kw):
    plan = _sealed_plan(tmp_path, lane, report=_report(**report_kw), **plan_kw)
    execution = _execution(plan)
    verify_loader_policy_supported(execution, lane=lane)
    if report_kw.get("precisions") == ("fp32",):
        # A plan without proven bf16 seals an fp32 4-bit compute dtype; it is lowered, not replaced.
        assert execution.precision.dequantization_dtype.value == "fp32"
    if report_kw.get("readiness") == "cpu_toy_only":
        assert execution.runtime_mode == "cpu_toy"
        assert [(e.module, e.device) for e in execution.device_map] == [("", "cpu")]


# --- refusals: values with no exact lowering -------------------------------------------------------------


def _precision(**updates: Any) -> Callable[[dict[str, Any]], None]:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["precision"].update(updates)

    return _mutate


@pytest.mark.parametrize(
    ("lane", "mutate", "expected"),
    [
        ("preference", _precision(quantized_storage_format="int4"), "quantization 'int4' has no lowering"),
        ("reward", _precision(quantized_storage_format="int4"), "rather than run nf4"),
        ("rollout", _precision(quantized_storage_format="int4"), "rather than run nf4"),
        ("reward", _precision(dequantization_dtype="fp16"), "different from the forward dtype"),
        ("rollout", _precision(dequantization_dtype="fp16"), "different from the forward dtype"),
        ("reward", _precision(master_weight_dtype=None), "master-weight dtype"),
        ("rollout", _precision(master_weight_dtype=None), "master-weight dtype"),
        (
            "preference",
            _precision(dequantization_dtype="fp8", forward_compute_dtype="fp8"),
            "dequantization dtype",
        ),
        (
            "full_finetune",
            _precision(weight_storage_dtype="tf32", forward_compute_dtype="tf32"),
            "weight-storage dtype 'tf32' has no lowering",
        ),
        # The QLoRA workers neither choose nor observe gradient or optimizer-state dtypes, so only what
        # their update path materializes is admitted (the planner's own shapes pass above).
        ("preference", _precision(gradient_dtype="bf16"), "sealed gradient dtype 'bf16' has no lowering"),
        ("reward", _precision(gradient_dtype="bf16"), "sealed gradient dtype 'bf16' has no lowering"),
        ("rollout", _precision(gradient_dtype="fp16"), "sealed gradient dtype 'fp16' has no lowering"),
        (
            "preference",
            _precision(optimizer_state_dtype="bf16"),
            "keeps 'adamw_torch' optimizer state in 'fp32'; the sealed optimizer-state dtype 'bf16'",
        ),
        ("reward", _precision(optimizer_state_dtype="int8"), "optimizer-state dtype 'int8' has no lowering"),
        ("rollout", _precision(optimizer_state_dtype="bf16"), "optimizer-state dtype 'bf16' has no lowering"),
        (
            "preference",
            _precision(master_weight_dtype="bf16", gradient_dtype="bf16"),
            "optimizer state in 'bf16'; the sealed optimizer-state dtype 'fp32' has no lowering",
        ),
        ("reward", _precision(optimizer_auxiliary_dtype="bf16"), "auxiliary dtype 'bf16' has no lowering"),
    ],
)
def test_precision_without_an_exact_lowering_is_refused(tmp_path, lane, mutate, expected):
    execution = _reseal(_execution(_sealed_plan(tmp_path, lane)), mutate)

    assert expected in _refused(execution, lane)


@pytest.mark.parametrize("lane", ["preference", "reward", "rollout"])
def test_a_consistent_non_default_update_precision_is_lowered(tmp_path, lane):
    # bf16 master weights accumulate bf16 gradients and adamw_torch keeps bf16 moments: nothing is
    # substituted, so the seal is admitted.
    execution = _reseal(
        _execution(_sealed_plan(tmp_path, lane)),
        _precision(master_weight_dtype="bf16", gradient_dtype="bf16", optimizer_state_dtype="bf16"),
    )

    verify_loader_policy_supported(execution, lane=lane)


def _paged_8bit(state: str) -> Callable[[dict[str, Any]], None]:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["optimizer"]["impl"] = "paged_adamw_8bit"
        payload["precision"]["optimizer_state_dtype"] = state

    return _mutate


@pytest.mark.parametrize("lane", ["preference", "reward", "rollout"])
def test_a_paged_8bit_optimizer_admits_only_8bit_state(tmp_path, lane):
    # The planner seals int8 state exactly when the optimizer is 8-bit; bitsandbytes keeps 8-bit moments.
    execution = _execution(_sealed_plan(tmp_path, lane))
    verify_loader_policy_supported(_reseal(execution, _paged_8bit("int8")), lane=lane)

    refused = _refused(_reseal(execution, _paged_8bit("fp32")), lane)

    assert "keeps 'paged_adamw_8bit' optimizer state in 'int8'" in refused
    assert "optimizer-state dtype 'fp32' has no lowering" in refused


def _device_map(*entries: tuple[str, str]) -> Callable[[dict[str, Any]], None]:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["device_map"] = [{"module": module, "device": device} for module, device in entries]

    return _mutate


def _cpu_toy(payload: dict[str, Any]) -> None:
    payload["runtime_mode"] = "cpu_toy"
    payload["device_map"] = [{"module": "", "device": "cpu"}]


@pytest.mark.parametrize(
    ("lane", "mutate", "expected"),
    [
        # The reward/rollout contracts accept split and rootless maps; no worker lowers them.
        ("reward", _device_map(("", "cuda:0"), ("lm_head", "cpu")), "one root placement"),
        ("rollout", _device_map(("model.layers", "cuda:1")), "one root placement"),
        # Nor do they refuse a repeated root; a dict would silently keep the last device (cuda:0).
        ("reward", _device_map(("", "cuda:1"), ("", "cuda:0")), "exactly one root placement"),
        ("rollout", _device_map(("", "cuda:1"), ("", "cuda:0")), "exactly one root placement"),
        ("reward", _device_map(("", "cuda:0"), ("", "cuda:0")), "exactly one root placement"),
        ("preference", _device_map(("", "cuda:1")), "sealed device 'cuda:1' cannot be honored"),
        ("reward", _device_map(("", "cuda")), "sealed device 'cuda' cannot be honored"),
        ("full_finetune", _device_map(("", "cuda:1")), "sealed device 'cuda:1' cannot be honored"),
        ("preference", _cpu_toy, "runs on CUDA only"),
        ("reward", _cpu_toy, "runs on CUDA only"),
        # A cpu_toy full-parameter seal must use the eager reference attention, not an SDPA GPU kernel.
        ("full_finetune", _cpu_toy, "eager reference attention"),
    ],
)
def test_placement_without_an_exact_lowering_is_refused(tmp_path, lane, mutate, expected):
    execution = _reseal(_execution(_sealed_plan(tmp_path, lane)), mutate)

    assert expected in _refused(execution, lane)


def _xformers(payload: dict[str, Any]) -> None:
    payload["attention"].update(
        model_attention_api="xformers",
        effective_backend_required="xformers",
        flash_sdp_enabled=False,
        mem_efficient_sdp_enabled=False,
        math_sdp_enabled=True,
    )


def _local_file_binding(kind: str, tmp_path: Path) -> Callable[[dict[str, Any]], None]:
    def _mutate(payload: dict[str, Any]) -> None:
        weights = tmp_path / f"{kind}.safetensors"
        weights.write_bytes(b"single-file")
        digest = stable_file_sha256(weights)
        payload["inputs"][kind] = {
            "kind": kind,
            "ref": {"id": f"{kind}-file", "hash": {"algo": "sha256", "value": digest}},
            "source": "local_file",
            "location": str(weights),
            "content_sha256": digest,
        }

    return _mutate


def test_identity_and_attention_values_without_a_lowering_are_refused(tmp_path):
    preference = _execution(_sealed_plan(tmp_path, "preference"))

    assert "source 'local_file' has no loader lowering" in _refused(
        _reseal(preference, _local_file_binding("model", tmp_path)), "preference"
    )
    assert "sealed tokenizer" in _refused(
        _reseal(preference, _local_file_binding("tokenizer", tmp_path)), "preference"
    )
    assert "'xformers' is not an attn_implementation" in _refused(
        _reseal(preference, _xformers), "preference"
    )
    assert "cannot consume a ResolvedPreferenceExecutionConfiguration" in _refused(
        preference, "reward"
    )
    assert "unknown first-party loader lane" in _refused(preference, "sft")
    # The contract pins these literals; an in-memory copy that bypassed validation is still refused.
    for unsafe in ({"trust_remote_code": True}, {"use_safetensors": False}):
        assert "trust_remote_code=False and use_safetensors=True" in _refused(
            preference.model_copy(update=unsafe), "preference"
        )


def test_a_served_reward_base_without_a_pinned_identity_is_refused(tmp_path):
    rollout = _execution(_sealed_plan(tmp_path, "rollout"))
    verify_loader_policy_supported(rollout, lane="rollout")

    def _other_base(payload: dict[str, Any]) -> None:
        payload["reward_source"]["reward_base_model"] = "review/other-reward-base"

    message = _refused(_reseal(rollout, _other_base), "rollout")
    assert "'review/other-reward-base' has no pinned identity" in message
    assert "refuse an unpinned reward model" in message


# --- pre-load re-hash of local model/tokenizer bindings on the newer lanes -------------------------------


@pytest.mark.parametrize("lane", ["preference", "full_finetune"])
def test_local_model_and_tokenizer_bytes_are_rehashed_for_the_newer_lanes(tmp_path, lane):
    model = _local_model(tmp_path)
    plan = _sealed_plan(
        tmp_path,
        lane,
        base_model=str(model),
        model_content_sha256=stable_directory_sha256(model),
    )
    execution = _execution(plan)
    assert execution.inputs.model.source == "local_directory"
    verify_execution_non_dataset_inputs(execution)  # unchanged bytes pass; no network for HF bindings

    (model / "model.safetensors").write_bytes(b"swapped-weights")
    with pytest.raises(ExecutionConfigurationError, match="model input bytes changed after planning"):
        verify_execution_non_dataset_inputs(execution)


def test_a_separate_local_tokenizer_binding_is_rehashed(tmp_path):
    model = _local_model(tmp_path)
    tokenizer = _local_model(tmp_path, "tokenizer")
    execution = _execution(
        _sealed_plan(
            tmp_path,
            "reward",
            base_model=str(model),
            model_content_sha256=stable_directory_sha256(model),
        )
    )
    digest = stable_directory_sha256(tokenizer)

    def _separate_tokenizer(payload: dict[str, Any]) -> None:
        payload["inputs"]["tokenizer"] = ExecutionInputBinding(
            kind="tokenizer",
            ref=Ref(id="tokenizer-dir", hash=HashRef(value=digest)),
            source="local_directory",
            location=str(tokenizer),
            content_sha256=digest,
        ).model_dump(mode="json")

    execution = _reseal(execution, _separate_tokenizer)
    verify_execution_non_dataset_inputs(execution)
    (tokenizer / "tokenizer.json").write_text('{"version": "2.0"}', encoding="utf-8")
    with pytest.raises(ExecutionConfigurationError, match="tokenizer input bytes changed"):
        verify_execution_non_dataset_inputs(execution)


# --- planning-time refusals ------------------------------------------------------------------------------


def _profile_on_gpu_index(index: int) -> EnvironmentProfile:
    reference = _profile(cc_major=8)
    return EnvironmentProfile(
        environment_signature=reference.environment_signature,
        host=EnvHost(os="linux"),
        gpus=[
            GpuDevice(
                index=index, kind="cuda", name="GPU", vram_total_bytes=12_000_000_000,
                compute_capability="8.0", compute_capability_major=8,
            )
        ],
    )


@pytest.mark.parametrize("lane", ["preference", "reward", "full_finetune"])
def test_planner_refuses_a_device_the_newer_workers_cannot_honor(tmp_path, lane):
    # The only visible GPU is cuda:1, so the planner would seal device_map {'': 'cuda:1'}; the worker's
    # kernel probe and the full-parameter HF Trainer run on cuda:0, so the plan is refused at planning.
    with pytest.raises(PlannerError) as refused:
        _sealed_plan(tmp_path, lane, profile=_profile_on_gpu_index(1))

    message = str(refused.value)
    assert "cannot be executed by the first-party worker" in message
    assert "sealed device 'cuda:1' cannot be honored" in message


@pytest.mark.parametrize(
    "plan_kw",
    [
        {"base_model": "Qwen/Qwen2.5-7B-Instruct"},  # a different base than the reward run's
        {"model_revision": "2" * 40},  # the reward run's base, at another commit
        {"tokenizer_revision": "3" * 40},  # the reward run's base with another tokenizer commit
    ],
)
def test_planner_refuses_a_reward_source_whose_bindings_differ_from_the_policy(tmp_path, plan_kw):
    with pytest.raises(PlannerError, match="must share the policy's pinned model and tokenizer"):
        _sealed_plan(tmp_path, "rollout", **plan_kw)


def test_planner_binds_the_served_reward_base_to_the_pinned_policy_base(tmp_path):
    rollout = _execution(_sealed_plan(tmp_path, "rollout"))

    assert rollout.reward_source.reward_base_model == rollout.inputs.model.location == _REWARD_BASE
    assert rollout.inputs.model.resolved_revision == "1" * 40


# --- runner admission: before the dataset read and before the worker module is imported ----------------


def _events(timeline: list[Any]) -> list[Any]:
    return [item for item in timeline if getattr(item, "event_type", None) is not None]


@pytest.mark.parametrize("lane", ["preference", "reward", "full_finetune"])
def test_a_local_model_changed_after_sealing_is_refused_before_any_read(tmp_path, monkeypatch, lane):
    import corpus_studio.training.full_finetune_trainer as full_finetune_trainer
    import corpus_studio.training.preference_worker as preference_worker
    import corpus_studio.training.reward_worker as reward_worker

    monkeypatch.chdir(tmp_path)
    model = _local_model(tmp_path)
    plan = _sealed_plan(
        tmp_path,
        lane,
        base_model=str(model),
        model_content_sha256=stable_directory_sha256(model),
    )

    def _must_not_dispatch(*_a: Any, **_k: Any) -> None:
        pytest.fail("the worker ran on model bytes that no longer match the seal")

    for module, name in (
        (preference_worker, "run_preference"),
        (reward_worker, "run_reward"),
        (full_finetune_trainer, "run_full_finetune"),
    ):
        monkeypatch.setattr(module, name, _must_not_dispatch)
    (model / "model.safetensors").write_bytes(b"swapped-weights")
    timeline: list[Any] = []

    result = execute_run(
        plan, build_lane_runner(required_runner_lane(plan)), run_id=f"run-{lane}", sink=timeline.append
    )

    failure = result.manifest.failure
    assert result.manifest.state == "failed" and failure is not None
    assert failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert failure.stage == StageMarker.env_loaded
    assert "model input bytes changed after planning" in failure.message
    stages = [event.stage for event in _events(timeline) if event.event_type == "stage"]
    assert StageMarker.dataset_verification not in stages  # refused before the dataset was read
    assert StageMarker.execution_config_verified not in stages


def _run_context(plan, execution_field: str, execution: Any, timeline: list[Any]) -> RunContext:
    return RunContext(
        plan.model_copy(update={execution_field: execution}), "run-x", timeline.append, CancelToken()
    )


def test_an_unlowerable_seal_is_refused_before_dispatch(tmp_path, monkeypatch):
    import corpus_studio.training.reward_worker as reward_worker

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(tmp_path, "reward")
    split = _reseal(plan.resolved_reward_execution, _device_map(("", "cuda:0"), ("lm_head", "cpu")))
    monkeypatch.setattr(
        reward_worker, "run_reward", lambda *_a, **_k: pytest.fail("dispatched an unlowerable seal")
    )
    timeline: list[Any] = []

    with pytest.raises(RunnerFailure, match="one root placement") as refused:
        RewardRunner(memory_sampler=lambda: None).run(
            _run_context(plan, "resolved_reward_execution", split, timeline)
        )

    assert refused.value.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION
    assert refused.value.stage == StageMarker.env_loaded
    assert "regenerate the RunPlan" in (refused.value.remediation or "")
    assert StageMarker.dataset_verification not in [event.stage for event in _events(timeline)]


def test_rollout_admission_refuses_an_unpinned_reward_base_before_dispatch(tmp_path, monkeypatch):
    import corpus_studio.training.rollout_worker as rollout_worker

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(tmp_path, "rollout")

    def _other_base(payload: dict[str, Any]) -> None:
        payload["reward_source"]["reward_base_model"] = "review/other-reward-base"

    unpinned = _reseal(plan.resolved_rollout_execution, _other_base)
    monkeypatch.setattr(
        rollout_worker, "run_rollout", lambda *_a, **_k: pytest.fail("dispatched an unpinned base")
    )

    with pytest.raises(RunnerFailure, match="refuse an unpinned reward model"):
        RolloutRunner(memory_sampler=lambda: None).run(
            _run_context(plan, "resolved_rollout_execution", unpinned, [])
        )


class _WorkerStop(Exception):
    """Raised by a fake worker after it has observed what the runner handed it."""


def test_admission_records_the_pinned_identity_before_the_dataset_is_read(tmp_path, monkeypatch):
    import corpus_studio.training.preference_worker as preference_worker

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(tmp_path, "preference", tokenizer_revision="b" * 40)
    execution = plan.resolved_preference_execution

    def _fake_run_preference(execution, *, dataset, output_dir=None, stage_callback=None):
        raise preference_worker.PreferenceWorkerError("stopped after admission")

    monkeypatch.setattr(preference_worker, "run_preference", _fake_run_preference)
    timeline: list[Any] = []

    result = execute_run(plan, build_lane_runner("preference"), run_id="run-id", sink=timeline.append)

    assert result.manifest.state == "failed"
    stage_events = [event for event in _events(timeline) if event.event_type == "stage"]
    admitted = [e for e in stage_events if e.stage == StageMarker.execution_config_verified]
    assert len(admitted) == 1
    assert admitted[0].payload == {
        "model": {
            "source": "huggingface",
            "location": "Qwen/Qwen2.5-7B-Instruct",
            "resolved_revision": "1" * 40,
            "content_sha256": None,
        },
        "tokenizer": {
            "source": "huggingface",
            "location": "Qwen/Qwen2.5-7B-Instruct",
            "resolved_revision": "b" * 40,
            "content_sha256": None,
        },
        "execution_configuration_hash": execution.configuration_hash,
    }
    first_dataset = next(
        index for index, e in enumerate(stage_events) if e.stage == StageMarker.dataset_verification
    )
    assert stage_events.index(admitted[0]) < first_dataset


@pytest.mark.parametrize(
    ("raised", "taxonomy", "stage"),
    [
        (
            ExecutionPlacementDeviation("PLACEMENT_DEVIATION: parameters outside cuda:0"),
            FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            StageMarker.placement_deviation,
        ),
        (
            TrainingEvidenceError(
                "trainable master-weight dtype could not be restored",
                taxonomy=FailureTaxonomy.GRADIENT_FAILURE,
                stage=StageMarker.adapter_attached,
            ),
            FailureTaxonomy.GRADIENT_FAILURE,
            StageMarker.adapter_attached,
        ),
        (
            TrainerEnvironmentError("the sealed attention kernel failed its runtime probe"),
            FailureTaxonomy.ENVIRONMENT_FAILURE,
            StageMarker.tokenizer_load,
        ),
        (
            TrainerError("attention policy deviation: the loaded model reports 'eager'"),
            FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            StageMarker.tokenizer_load,
        ),
    ],
)
def test_worker_loader_refusals_keep_the_sft_taxonomy_and_stream_their_stages(
    tmp_path, monkeypatch, raised, taxonomy, stage
):
    import corpus_studio.training.preference_worker as preference_worker

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(tmp_path, "preference")

    def _fake_run_preference(execution, *, dataset, output_dir=None, stage_callback=None):
        stage_callback("tokenizer_load", "loaded and verified the sealed tokenizer")
        stage_callback("precision_verified", "observed nf4 base storage")
        stage_callback("tokenizer_loadd", "a typo must surface")
        raise raised

    monkeypatch.setattr(preference_worker, "run_preference", _fake_run_preference)
    timeline: list[Any] = []

    result = execute_run(plan, build_lane_runner("preference"), run_id="run-map", sink=timeline.append)

    failure = result.manifest.failure
    assert failure is not None
    assert (failure.taxonomy, failure.stage) == (taxonomy, stage)
    assert str(raised) in failure.message
    events = _events(timeline)
    assert any(
        e.event_type == "stage" and e.stage == StageMarker.tokenizer_load
        and e.message == "loaded and verified the sealed tokenizer"
        for e in events
    )
    logs = [e.message for e in events if e.event_type == "log"]
    assert "precision_verified: observed nf4 base storage" in logs
    assert "unrecognized progress stage tokenizer_loadd: a typo must surface" in logs


def test_the_full_parameter_runner_takes_cpu_toy_only_from_the_seal(tmp_path, monkeypatch):
    import corpus_studio.training.full_finetune_trainer as full_finetune_trainer

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(
        tmp_path, "full_finetune", report=_report(readiness="cpu_toy_only"), allow_cpu_toy=True
    )
    received: dict[str, Any] = {}

    def _fake_run_full_finetune(execution, **kwargs):
        received.update(kwargs)
        raise full_finetune_trainer.FullFinetuneError("stopped after dispatch")

    monkeypatch.setattr(full_finetune_trainer, "run_full_finetune", _fake_run_full_finetune)

    result = execute_run(plan, build_lane_runner(required_runner_lane(plan)), run_id="run-toy")

    assert result.manifest.state == "failed"
    assert set(received) == {"dataset", "output_dir", "stage_callback"}  # no runner cpu_toy flag
    assert not hasattr(build_lane_runner("full_finetune"), "cpu_toy")
    assert callable(received["stage_callback"])


@pytest.mark.parametrize(
    ("lane", "module_name", "worker_name"),
    [
        ("reward", "reward_worker", "run_reward"),
        ("full_finetune", "full_finetune_trainer", "run_full_finetune"),
        ("rollout", "rollout_worker", "run_rollout"),
    ],
)
def test_every_newer_runner_maps_a_worker_loader_refusal(
    tmp_path, monkeypatch, lane, module_name, worker_name
):
    import importlib

    monkeypatch.chdir(tmp_path)
    plan = _sealed_plan(tmp_path, lane)

    def _fake_worker(execution, *, dataset, output_dir=None, stage_callback=None):
        stage_callback("model_load", "materialized the sealed model weights")
        raise ExecutionPlacementDeviation("PLACEMENT_DEVIATION: parameters outside cuda:0")

    monkeypatch.setattr(
        importlib.import_module(f"corpus_studio.training.{module_name}"), worker_name, _fake_worker
    )
    timeline: list[Any] = []
    if lane == "rollout":  # admitted at planning, not yet at execution: drive its runner directly
        with pytest.raises(RunnerFailure) as refused:
            RolloutRunner(memory_sampler=lambda: None).run(
                RunContext(plan, "run-x", timeline.append, CancelToken())
            )
        taxonomy, stage = refused.value.taxonomy, refused.value.stage
    else:
        result = execute_run(
            plan, build_lane_runner(required_runner_lane(plan)), run_id="run-x", sink=timeline.append
        )
        assert result.manifest.failure is not None
        taxonomy, stage = result.manifest.failure.taxonomy, result.manifest.failure.stage

    assert (taxonomy, stage) == (
        FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
        StageMarker.placement_deviation,
    )
    assert StageMarker.model_load in [e.stage for e in _events(timeline) if e.event_type == "stage"]
