"""Pure execution-seal tests; no torch, model downloads, or hardware required."""

from __future__ import annotations

from pathlib import Path

import pytest

from corpus_studio.platform.common import HashRef, Ref
from corpus_studio.platform.contracts import RunDispatchBody
from corpus_studio.platform.enums import AdapterMethod, LossImpl, QuantizationMode
from corpus_studio.platform.execution_config import (
    ExecutionConfigurationError,
    formatter_identity,
    local_input_binding,
    required_runner_lane,
    run_scoped_training_output,
    stable_directory_sha256,
    stable_file_bytes,
    verify_execution_inputs,
    verify_execution_objective,
    verify_runner_lane,
)
from corpus_studio.platform.planner import compute_plan_hash, run_plan_hash_payload
from corpus_studio.platform.runners import demo_training_plan
from corpus_studio.platform.supervisor import demo_run_plan
from corpus_studio.platform.worker_protocol import (
    build_worker_message,
    decode_worker_message,
    encode_worker_message,
    parse_worker_body,
)


def _execution():
    execution = demo_training_plan().resolved_execution
    assert execution is not None
    return execution


def test_runner_lane_is_derived_from_the_sealed_plan_only():
    plan = demo_training_plan()
    assert required_runner_lane(plan) == "cpu_toy"
    verify_runner_lane(plan, "cpu_toy")
    with pytest.raises(ExecutionConfigurationError, match="sealed lane"):
        verify_runner_lane(plan, "training")

    wrong_backend = plan.model_copy(update={"backend_ref": Ref(id="another-worker")})
    with pytest.raises(ExecutionConfigurationError, match="first-party"):
        required_runner_lane(wrong_backend)

    echo = demo_run_plan()
    assert required_runner_lane(echo) == "echo"
    with pytest.raises(ExecutionConfigurationError, match="no executable runner lane"):
        required_runner_lane(echo.model_copy(update={"backend_ref": Ref(id="unknown")}))


def test_removing_resolved_execution_cannot_turn_a_training_plan_into_echo():
    plan = demo_training_plan()
    draft = plan.model_copy(update={"resolved_execution": None})
    tampered = draft.model_copy(
        update={"plan_hash": compute_plan_hash(run_plan_hash_payload(draft))}
    )

    with pytest.raises(ExecutionConfigurationError, match="no executable runner lane"):
        required_runner_lane(tampered)


def test_run_scoped_output_is_derived_from_the_sealed_root():
    execution = _execution()
    path = run_scoped_training_output(execution, "run-123")
    assert path == Path(execution.output_dir) / "runs" / "run-123" / "artifacts" / "adapter"
    with pytest.raises(ExecutionConfigurationError, match="unsafe"):
        run_scoped_training_output(execution, "../escape")


def test_formatter_identity_is_implementation_bound_and_fail_closed(monkeypatch):
    formatter_id, digest = formatter_identity("trace")
    assert formatter_id.endswith("structured-trace-renderer-v1")
    assert len(digest) == 64
    with pytest.raises(ExecutionConfigurationError, match="no sealed formatter"):
        formatter_identity("unknown")

    import corpus_studio.platform.execution_config as module

    monkeypatch.setattr(module.inspect, "getsource", lambda _value: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(ExecutionConfigurationError, match="cannot inspect"):
        formatter_identity("instruction")


def test_stable_file_and_directory_bindings_cover_exact_bytes(tmp_path: Path):
    source = tmp_path / "dataset.jsonl"
    source.write_bytes(b'{"instruction":"a","output":"b"}\n')
    content, digest = stable_file_bytes(source)
    assert content.startswith(b"{") and len(digest) == 64

    file_binding = local_input_binding(
        kind="dataset", location=str(source), ref_id="dataset", directory=False
    )
    assert file_binding.content_sha256 == digest

    root = tmp_path / "model"
    (root / "nested").mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "nested" / "weights.safetensors").write_bytes(b"weights")
    ignored = root / ".git"
    ignored.mkdir()
    (ignored / "volatile").write_text("one", encoding="utf-8")
    directory_digest = stable_directory_sha256(root)
    (ignored / "volatile").write_text("two", encoding="utf-8")
    assert stable_directory_sha256(root) == directory_digest

    directory_binding = local_input_binding(
        kind="model", location=str(root), ref_id="model", directory=True
    )
    assert directory_binding.content_sha256 == directory_digest
    (root / "config.json").write_text('{"changed":true}', encoding="utf-8")
    assert stable_directory_sha256(root) != directory_digest


def test_stable_input_checks_reject_missing_empty_and_link_roots(tmp_path: Path):
    with pytest.raises(ExecutionConfigurationError, match="file does not exist"):
        stable_file_bytes(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExecutionConfigurationError, match="directory is empty"):
        stable_directory_sha256(empty)
    with pytest.raises(ExecutionConfigurationError, match="directory does not exist"):
        stable_directory_sha256(tmp_path / "missing-directory")

    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        return
    with pytest.raises(ExecutionConfigurationError, match="cannot be a link"):
        stable_file_bytes(link)


def test_execution_inputs_are_revalidated_against_current_local_bytes(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    config = _execution()
    binding = local_input_binding(
        kind="dataset", location=str(dataset), ref_id="dataset", directory=False
    )
    config = config.model_copy(
        update={"inputs": config.inputs.model_copy(update={"dataset": binding})}
    )
    verify_execution_inputs(config)
    dataset.write_text('{"instruction":"changed","output":"b"}\n', encoding="utf-8")
    with pytest.raises(ExecutionConfigurationError, match="bytes changed"):
        verify_execution_inputs(config)

    pinned_config = _execution()
    unpinned = pinned_config.inputs.model.model_copy(update={"resolved_revision": None})
    config = pinned_config.model_copy(
        update={"inputs": pinned_config.inputs.model_copy(update={"model": unpinned})}
    )
    with pytest.raises(ExecutionConfigurationError, match="not pinned"):
        verify_execution_inputs(config)


def test_execution_objective_is_bound_to_worker_semantics(monkeypatch):
    config = _execution()
    verify_execution_objective(config, task_type="sft")

    missing = config.model_copy(
        update={"objective_ref": Ref(id="missing", hash=HashRef(value="f" * 64))}
    )
    with pytest.raises(ExecutionConfigurationError, match="not in the current registry"):
        verify_execution_objective(missing, task_type="sft")

    stale = config.model_copy(
        update={
            "objective_ref": config.objective_ref.model_copy(
                update={"hash": HashRef(value="f" * 64)}
            )
        }
    )
    with pytest.raises(ExecutionConfigurationError, match="hash is stale"):
        verify_execution_objective(stale, task_type="sft")
    with pytest.raises(ExecutionConfigurationError, match="does not match the RunPlan task"):
        verify_execution_objective(config, task_type="pretraining")

    wrong_adapter = config.model_copy(
        update={"adapter": config.adapter.model_copy(update={"method": AdapterMethod.qlora})}
    )
    with pytest.raises(ExecutionConfigurationError, match="adapter method"):
        verify_execution_objective(wrong_adapter, task_type="sft")

    wrong_loss = config.model_copy(update={"loss_impl": LossImpl.dpo})
    with pytest.raises(ExecutionConfigurationError, match="loss implementation"):
        verify_execution_objective(wrong_loss, task_type="sft")

    wrong_quantization = config.model_copy(
        update={
            "precision": config.precision.model_copy(
                update={"quantized_storage_format": QuantizationMode.nf4}
            )
        }
    )
    with pytest.raises(ExecutionConfigurationError, match="unquantized"):
        verify_execution_objective(wrong_quantization, task_type="sft")

    wrong_format = config.model_copy(
        update={"data": config.data.model_copy(update={"dataset_format": "trace"})}
    )
    with pytest.raises(ExecutionConfigurationError, match="dataset format"):
        verify_execution_objective(wrong_format, task_type="sft")

    from corpus_studio.platform.objectives import get_objective

    objective = get_objective("lora")
    assert objective is not None
    without_adapter = objective.model_copy(
        update={
            "expected_artifacts": [
                item for item in objective.expected_artifacts if item.kind.value != "adapter"
            ]
        }
    )
    monkeypatch.setattr(
        "corpus_studio.platform.objectives.get_objective", lambda _objective_id: without_adapter
    )
    with pytest.raises(ExecutionConfigurationError, match="adapter artifact"):
        verify_execution_objective(config, task_type="sft")


def test_objective_seal_survives_worker_protocol_exclude_none_roundtrip():
    plan = demo_training_plan()
    execution = plan.resolved_execution
    assert execution is not None
    message = build_worker_message(
        "run_dispatch",
        RunDispatchBody(run_id="run-objective-seal", plan=plan),
        message_id="dispatch-objective-seal",
        direction="core_to_worker",
    )

    decoded = decode_worker_message(
        encode_worker_message(message), expected_direction="core_to_worker"
    )
    body = parse_worker_body(decoded)
    assert isinstance(body, RunDispatchBody)
    reloaded_execution = body.plan.resolved_execution
    assert reloaded_execution is not None
    assert reloaded_execution.objective_ref == execution.objective_ref
