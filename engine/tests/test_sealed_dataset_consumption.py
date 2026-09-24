"""The DPO, reward, full-parameter SFT and on-policy RL lanes consume only the sealed dataset bytes (#862),
proven through the real platform-run dispatch rather than a helper in isolation:

* in-process ``platform-run``: ``execute_run(plan, build_lane_runner(required_runner_lane(plan)))``;
* the ``--subprocess`` worker entrypoint: ``worker.run_worker`` over a real ``run_dispatch`` line;
* a genuinely spawned child: ``execute_run_subprocess`` with the default
  ``-P -m corpus_studio.platform.worker`` argv. That child test is deterministic with or without torch
  installed, because the refusal precedes every heavy import.

Plans come from the real planner over a temporary JSONL file. The heavy training stack is replaced by
fake modules whose tokenizer/model loaders and the sealed formatters are recorded on one timeline, so
each test can show what the worker reached and in which order. The rollout lane is not admitted at
execution yet (``on_policy_rl`` is contract_validated), so its runner is driven through a real
``RunContext`` directly."""

from __future__ import annotations

import importlib.machinery
import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from test_platform_planner import _REWARD_BRINGUP, _plan, _profile, _report
from test_reward_routing import _success as _reward_success

import corpus_studio.importers.jsonl_importer as jsonl_importer
import corpus_studio.platform.execution_config as execution_config
import corpus_studio.training.sealed_inputs as sealed_inputs
import corpus_studio.training.trainer as trainer_module
from corpus_studio.platform import supervisor
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import (
    required_runner_lane,
    run_scoped_training_output,
    stable_file_sha256,
)
from corpus_studio.platform.runners import RolloutRunner, build_lane_runner
from corpus_studio.platform.supervisor import CancelToken, RunContext, RunnerFailure, execute_run
from corpus_studio.training.sealed_inputs import VerifiedDataset

_LOADERS = frozenset(
    {
        "AutoTokenizer.from_pretrained",
        "AutoModelForCausalLM.from_pretrained",
        "AutoModelForSequenceClassification.from_pretrained",
        "PeftModel.from_pretrained",
    }
)


class _StopAtLoader(Exception):
    """Raised by a fake loader: the test only needs to see how far the worker got, never a real load."""


def _preference_rows(marker: str) -> list[dict[str, Any]]:
    return [{"prompt": f"p{index}", "chosen": marker, "rejected": "r"} for index in range(6)]


def _sft_rows(marker: str) -> list[dict[str, Any]]:
    return [{"instruction": f"i{index}", "output": marker} for index in range(6)]


def _prompt_rows(marker: str) -> list[dict[str, Any]]:
    return [{"messages": [{"role": "user", "content": f"{marker} {index}"}]} for index in range(6)]


_LANES: dict[str, tuple[dict[str, Any], Any]] = {
    "preference": ({"task_type": "preference", "objective_id": "dpo_qlora"}, _preference_rows),
    "reward": ({"task_type": "reward", "objective_id": "reward_model"}, _preference_rows),
    "full_finetune": (
        {"task_type": "sft", "adapter_method": "full_finetune", "export_format": "merged_safetensors"},
        _sft_rows,
    ),
    "rollout": (
        {
            "task_type": "grpo",
            "objective_id": "grpo",
            "reward_source_manifest": str(
                _REWARD_BRINGUP / "runs/run-reward-sealed-0001/RunManifest.json"
            ),
            "reward_source_plan": str(_REWARD_BRINGUP / "reward-bringup.RunPlan.json"),
        },
        _prompt_rows,
    ),
}
_ADMITTED_LANES = ("preference", "reward", "full_finetune")


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sealed_plan(tmp_path: Path, lane: str, *, content_sha256: str | None = None):
    plan_kw, rows_for = _LANES[lane]
    data = tmp_path / "data.jsonl"
    _write(data, rows_for("SEALED"))
    plan = _plan(
        _profile(cc_major=8),
        _report(),
        dataset_path=str(data),
        dataset_content_sha256=content_sha256 or stable_file_sha256(data),
        **plan_kw,
    )
    return plan, data, rows_for


def _execution(plan):
    return (
        plan.resolved_preference_execution
        or plan.resolved_reward_execution
        or plan.resolved_full_finetune_execution
        or plan.resolved_rollout_execution
    )


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_fake_training_stack(monkeypatch, timeline: list[tuple], *, stop_at_model: bool = True):
    """Fake torch/transformers/peft/datasets whose loaders, and the sealed formatters, record onto
    ``timeline``. The model loaders raise :class:`_StopAtLoader` unless ``stop_at_model`` is False."""

    class _Tokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **_kw):
            return "|".join(str(message["content"]) for message in messages)

        def __call__(self, _text, **_kw):
            return {"input_ids": [1, 2, 3]}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_a, **_k):
            timeline.append(("load", "AutoTokenizer.from_pretrained"))
            return _Tokenizer()

    class _ModelLoader:
        @classmethod
        def from_pretrained(cls, *_a, **_k):
            timeline.append(("load", f"{cls.__name__}.from_pretrained"))
            if stop_at_model:
                raise _StopAtLoader(f"{cls.__name__}.from_pretrained reached")
            return types.SimpleNamespace(config=types.SimpleNamespace(use_cache=True))

    class AutoModelForCausalLM(_ModelLoader):
        pass

    class AutoModelForSequenceClassification(_ModelLoader):
        pass

    class PeftModel(_ModelLoader):
        pass

    class Dataset:
        @staticmethod
        def from_list(_rows):
            raise _StopAtLoader("Dataset.from_list reached")

    transformers = _module(
        "transformers",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoModelForSequenceClassification=AutoModelForSequenceClassification,
        BitsAndBytesConfig=lambda **_k: None,
        Trainer=object,
        TrainerCallback=object,
        TrainingArguments=object,
        set_seed=lambda _seed: None,
    )
    peft = _module(
        "peft",
        LoraConfig=object,
        PeftModel=PeftModel,
        get_peft_model=object,
        get_peft_model_state_dict=object,
        prepare_model_for_kbit_training=object,
    )
    for name, module in (
        ("torch", _module("torch", bfloat16="bf16", float32="fp32")),
        ("transformers", transformers),
        ("peft", peft),
        ("datasets", _module("datasets", Dataset=Dataset)),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    real_pair = trainer_module.format_preference_pair
    real_example = trainer_module.format_example_text

    def _pair(row, tokenizer=None):
        timeline.append(("format", json.dumps(row)))
        return real_pair(row, tokenizer)

    def _example(row, dataset_format, tokenizer=None):
        timeline.append(("format", json.dumps(row)))
        return real_example(row, dataset_format, None)

    def _prompt(row, tokenizer=None):
        timeline.append(("format", json.dumps(row)))
        return "prompt"

    monkeypatch.setattr(trainer_module, "format_preference_pair", _pair)
    monkeypatch.setattr(trainer_module, "format_example_text", _example)
    monkeypatch.setattr(trainer_module, "format_rollout_prompt", _prompt)


def _dispatch(plan, lane: str, run_id: str, timeline: list[tuple]):
    """Run ``plan`` through the lane's real runner and return (state, taxonomy, stage, message)."""

    def _sink(event) -> None:
        timeline.append(("event", event))

    if lane == "rollout":
        ctx = RunContext(plan, run_id, _sink, CancelToken())
        try:
            RolloutRunner().run(ctx)
        except RunnerFailure as exc:
            return "failed", exc.taxonomy, exc.stage, str(exc)
        except _StopAtLoader as exc:
            return "failed", FailureTaxonomy.FAIL, None, str(exc)
        return "succeeded", None, None, ""  # pragma: no cover - the fake loaders never let it finish
    result = execute_run(
        plan, build_lane_runner(required_runner_lane(plan)), run_id=run_id, sink=_sink
    )
    failure = result.manifest.failure
    if failure is None:  # pragma: no cover - the fake loaders never let a run finish
        return result.manifest.state, None, None, ""
    return result.manifest.state, failure.taxonomy, failure.stage, failure.message


def _verified_events(timeline: list[tuple]) -> list[Any]:
    return [
        item[1]
        for item in timeline
        if item[0] == "event"
        and item[1].stage == StageMarker.dataset_verification
        and item[1].payload is not None
    ]


def _heavy_calls(timeline: list[tuple]) -> list[tuple]:
    return [item for item in timeline if item[0] in {"load", "format"}]


# --- refusals: a changed or unusable dataset never reaches a loader, formatter or output directory ------


@pytest.mark.parametrize("lane", sorted(_LANES))
@pytest.mark.parametrize(
    "case", ["post_plan", "mismatched_seal", "mid_read", "missing", "link", "sealed_malformed"]
)
def test_non_sft_lanes_refuse_a_changed_dataset_before_any_loader(tmp_path, monkeypatch, lane, case):
    monkeypatch.chdir(tmp_path)
    if case == "mismatched_seal":
        plan, data, rows_for = _sealed_plan(tmp_path, lane, content_sha256="e" * 64)
    elif case == "sealed_malformed":
        # The seal names these exact bytes, so only the parse of the verified bytes can refuse them.
        malformed = tmp_path / "malformed.jsonl"
        malformed.write_text('{"prompt": "p"}\n{not json\n', encoding="utf-8")
        plan, data, rows_for = _sealed_plan(
            tmp_path, lane, content_sha256=stable_file_sha256(malformed)
        )
        data.write_bytes(malformed.read_bytes())
    else:
        plan, data, rows_for = _sealed_plan(tmp_path, lane)
    if case == "post_plan":
        _write(data, rows_for("TAMPERED"))
    elif case == "missing":
        data.unlink()
    elif case == "link":
        target = tmp_path / "real.jsonl"
        data.rename(target)
        try:
            data.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are not supported on this filesystem")
    elif case == "mid_read":
        real_stable_file_bytes = sealed_inputs.stable_file_bytes

        def _append_during_read(path, *, progress_callback=None):
            appended: list[int] = []

            def _callback(completed: int, total: int) -> None:
                if not appended:
                    appended.append(completed)
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rows_for("EVIL")[0]) + "\n")
                if progress_callback is not None:
                    progress_callback(completed, total)

            return real_stable_file_bytes(path, progress_callback=_callback)

        monkeypatch.setattr(sealed_inputs, "stable_file_bytes", _append_during_read)
    timeline: list[tuple] = []
    _install_fake_training_stack(monkeypatch, timeline)

    state, taxonomy, stage, message = _dispatch(plan, lane, f"run-{lane}", timeline)

    assert state == "failed"
    assert taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION, message
    assert stage == StageMarker.dataset_verification
    expected = {
        "post_plan": "dataset bytes changed after the execution configuration was sealed",
        "mismatched_seal": "dataset bytes changed after the execution configuration was sealed",
        "mid_read": "changed while hashing",
        "missing": "does not exist",
        "link": "cannot be a link",
        "sealed_malformed": "sealed dataset is invalid: line 2: Invalid JSON",
    }[case]
    assert expected in message
    assert _heavy_calls(timeline) == []
    assert _verified_events(timeline) == []
    execution = _execution(plan)
    assert not run_scoped_training_output(execution, f"run-{lane}").exists()
    assert not Path(execution.output_dir).exists()


# --- same bytes: the sealed rows reach the real worker body, after verification --------------------------


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_same_bytes_reach_the_real_worker_body_after_verification(tmp_path, monkeypatch, lane):
    monkeypatch.chdir(tmp_path)
    plan, data, _rows_for = _sealed_plan(tmp_path, lane)
    timeline: list[tuple] = []
    # Full-parameter SFT formats rows only after its model loads, so let that load "succeed" and stop at
    # the dataset build instead; the other lanes format before any model load.
    _install_fake_training_stack(monkeypatch, timeline, stop_at_model=(lane != "full_finetune"))

    state, _taxonomy, _stage, message = _dispatch(plan, lane, f"run-{lane}", timeline)

    assert state == "failed" and "reached" in message  # stopped by a fake loader, not by verification
    verified = _verified_events(timeline)
    assert len(verified) == 1
    assert verified[0].payload == {
        "content_sha256": stable_file_sha256(data),
        "byte_count": data.stat().st_size,
        "row_count": 6,
        "execution_configuration_hash": _execution(plan).configuration_hash,
    }
    assert verified[0].payload["content_sha256"] == _execution(plan).inputs.dataset.content_sha256
    formatted = [item[1] for item in timeline if item[0] == "format"]
    assert formatted and all("SEALED" in row for row in formatted)
    verification_index = next(
        index for index, item in enumerate(timeline) if item[0] == "event" and item[1] is verified[0]
    )
    first_heavy_index = next(index for index, item in enumerate(timeline) if item[0] in {"load", "format"})
    assert verification_index < first_heavy_index
    if lane == "full_finetune":
        loads = [item[1] for item in timeline if item[0] == "load"]
        assert loads[0] == "AutoModelForCausalLM.from_pretrained"


@pytest.mark.parametrize("lane", _ADMITTED_LANES)
def test_dataset_is_read_exactly_once_per_run(tmp_path, monkeypatch, lane):
    monkeypatch.chdir(tmp_path)
    plan, data, _rows_for = _sealed_plan(tmp_path, lane)
    target = str(data)
    stable_reads: list[str] = []
    real_stable_file_read = execution_config._stable_file_read

    def _counting_stable_read(path, **kwargs):
        if str(path) == target:
            stable_reads.append(str(path))
        return real_stable_file_read(path, **kwargs)

    def _refuse_path_reread(real):
        def _reader(path, *args, **kwargs):
            if str(path) == target:
                pytest.fail("the sealed dataset path was reopened by a mutable-path reader")
            return real(path, *args, **kwargs)

        return _reader

    monkeypatch.setattr(execution_config, "_stable_file_read", _counting_stable_read)
    monkeypatch.setattr(jsonl_importer, "read_jsonl", _refuse_path_reread(jsonl_importer.read_jsonl))
    monkeypatch.setattr(jsonl_importer, "iter_jsonl", _refuse_path_reread(jsonl_importer.iter_jsonl))
    timeline: list[tuple] = []
    _install_fake_training_stack(monkeypatch, timeline, stop_at_model=(lane != "full_finetune"))

    state, _taxonomy, _stage, message = _dispatch(plan, lane, f"run-{lane}", timeline)

    assert state == "failed" and "reached" in message
    assert stable_reads == [target]  # one stable read + hash + capture; no second full-corpus pass
    assert len(_verified_events(timeline)) == 1


def test_same_bytes_run_is_admitted_with_the_consumed_digest_on_its_event_stream(
    tmp_path, monkeypatch
):
    import corpus_studio.training.reward_worker as reward_worker

    monkeypatch.chdir(tmp_path)
    plan, data, rows_for = _sealed_plan(tmp_path, "reward")
    execution = plan.resolved_reward_execution
    sealed_digest = execution.inputs.dataset.content_sha256

    def _fake_run_reward(execution, *, dataset, output_dir=None):
        assert isinstance(dataset, VerifiedDataset)
        assert dataset.content_sha256 == sealed_digest
        assert list(dataset.rows) == rows_for("SEALED")
        Path(output_dir).mkdir(parents=True)
        return types.SimpleNamespace(output_dir=output_dir, success_evidence=_reward_success())

    monkeypatch.setattr(reward_worker, "run_reward", _fake_run_reward)
    monkeypatch.setattr(supervisor, "_reload_verify_adapter", lambda *_a, **_k: (True, None))
    records = tmp_path / "records"

    result = execute_run(
        plan,
        build_lane_runner(required_runner_lane(plan)),
        run_id="run-reward-admitted",
        out_dir=records,
    )

    assert result.manifest.state == "succeeded", result.manifest.failure
    persisted = [
        json.loads(line)
        for line in (records / "runs/run-reward-admitted/RunEvents.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    payloads = [
        event["payload"]
        for event in persisted
        if event.get("stage") == "dataset_verification" and event.get("payload")
    ]
    assert payloads == [
        {
            "content_sha256": sealed_digest,
            "byte_count": data.stat().st_size,
            "row_count": 6,
            "execution_configuration_hash": execution.configuration_hash,
        }
    ]


# --- the subprocess worker entrypoint and a genuinely spawned child -----------------------------------


@pytest.mark.parametrize("lane", _ADMITTED_LANES)
def test_worker_entrypoint_refuses_a_changed_dataset_before_loaders(tmp_path, monkeypatch, lane):
    from corpus_studio.platform.subprocess_supervisor import _dispatch_line
    from corpus_studio.platform.worker import run_worker

    monkeypatch.chdir(tmp_path)
    plan, data, rows_for = _sealed_plan(tmp_path, lane)
    _write(data, rows_for("TAMPERED"))
    timeline: list[tuple] = []
    _install_fake_training_stack(monkeypatch, timeline)
    out = io.StringIO()

    rc = run_worker(
        _dispatch_line(plan, f"run-w-{lane}", 30),
        runner_name=required_runner_lane(plan),
        backend_id=plan.backend_ref.id,
        environment_ref=plan.environment_ref,
        out=out,
    )

    messages = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    terminal = messages[-1]["body"]
    assert rc == 0
    assert terminal["outcome"] == "UNSUPPORTED_CONFIGURATION"
    assert terminal["failure"]["stage"] == "dataset_verification"
    assert "dataset bytes changed after the execution configuration was sealed" in (
        terminal["failure"]["message"]
    )
    assert _heavy_calls(timeline) == []


@pytest.mark.parametrize("lane", _ADMITTED_LANES)
def test_platform_run_subprocess_refuses_a_changed_dataset_before_importing_torch(
    tmp_path, monkeypatch, lane
):
    from corpus_studio.platform.subprocess_supervisor import execute_run_subprocess

    monkeypatch.chdir(tmp_path)
    plan, data, rows_for = _sealed_plan(tmp_path, lane)
    _write(data, rows_for("TAMPERED"))

    result = execute_run_subprocess(
        plan, run_id=f"run-child-{lane}", runner_name="auto", silence_timeout_s=60
    )

    failure = result.manifest.failure
    assert result.manifest.state == "failed"
    assert failure is not None
    assert failure.taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION, failure.message
    assert failure.stage == StageMarker.dataset_verification
    assert "dataset bytes changed after the execution configuration was sealed" in failure.message
    assert not run_scoped_training_output(_execution(plan), f"run-child-{lane}").exists()


# --- the worker-side guard: rows must be bound to the worker's own seal ---------------------------------


def _worker(lane: str):
    if lane == "preference":
        from corpus_studio.training.preference_worker import PreferenceWorkerError, run_preference

        return run_preference, PreferenceWorkerError
    if lane == "reward":
        from corpus_studio.training.reward_worker import RewardWorkerError, run_reward

        return run_reward, RewardWorkerError
    if lane == "rollout":
        from corpus_studio.training.rollout_worker import RolloutWorkerError, run_rollout

        return run_rollout, RolloutWorkerError
    from corpus_studio.training.full_finetune_trainer import FullFinetuneError, run_full_finetune

    return run_full_finetune, FullFinetuneError


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_worker_refuses_rows_not_bound_to_its_seal(tmp_path, monkeypatch, lane):
    plan, data, rows_for = _sealed_plan(tmp_path, lane)
    execution = _execution(plan)
    run_worker_fn, worker_error = _worker(lane)
    timeline: list[tuple] = []
    _install_fake_training_stack(monkeypatch, timeline)
    rows = tuple(rows_for("FORGED"))
    forged = VerifiedDataset(
        rows=rows, content_sha256="0" * 64, byte_count=1, location=str(data)
    )
    empty = VerifiedDataset(
        rows=(),
        content_sha256=execution.inputs.dataset.content_sha256,
        byte_count=0,
        location=execution.inputs.dataset.location,
    )
    output_dir = str(tmp_path / "out")

    with pytest.raises(worker_error, match="does not belong to this sealed execution"):
        run_worker_fn(execution, dataset=forged, output_dir=output_dir)
    with pytest.raises(worker_error, match="requires rows from the verified sealed dataset read"):
        run_worker_fn(execution, dataset=list(rows), output_dir=output_dir)
    with pytest.raises(worker_error, match="dataset is empty"):
        run_worker_fn(execution, dataset=empty, output_dir=output_dir)
    assert _heavy_calls(timeline) == []
    assert not Path(output_dir).exists()
