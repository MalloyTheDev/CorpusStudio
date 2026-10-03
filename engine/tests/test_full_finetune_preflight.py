"""Full-parameter SFT honors the sealed no-truncation policy (#861).

The full-parameter worker shares the adapter SFT lane's full-content preflight
(``trainer.preflight_sft_dataset``): every verified row is formatted with the bound tokenizer, tokenized
ONCE at full length, and measured by the token-coverage ledger before the kernel probe or any weights
load. Under the default ``refuse`` policy an over-length or unrenderable row anywhere is refused; a
sealed lossy policy cuts explicitly and records deterministic counts bound to the execution hash. The
rows that train are exactly the measured ids.

Every seal comes from the real planner and is re-validated against its own contract before being
re-sealed (``_reseal``). The worker and runner tests drive the REAL ``run_full_finetune`` and
``FullFinetuneRunner`` against the deterministic fake stack in ``_sealed_loader_fakes``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from _sealed_loader_fakes import FakeStack
from pydantic import ValidationError
from test_platform_planner import _plan, _profile, _report
from test_sealed_loader import B40, TOKENIZER, _sealed, _Stages
from test_sealed_loader_admission import _events, _reseal

import corpus_studio.training.full_finetune_trainer as full_finetune_trainer
from corpus_studio.platform.enums import FailureTaxonomy, StageMarker
from corpus_studio.platform.execution_config import (
    formatter_identity,
    required_runner_lane,
    run_scoped_training_output,
    stable_file_sha256,
)
from corpus_studio.platform.runners import build_lane_runner
from corpus_studio.platform.supervisor import execute_run
from corpus_studio.training.full_finetune_trainer import (
    FullFinetuneDataRefusal,
    FullFinetuneError,
    build_full_finetune_rows,
    full_finetune_truncation_permitted,
    pad_sft_row,
    prepare_full_finetune_dataset,
    run_full_finetune,
)
from corpus_studio.training.sealed_inputs import read_verified_dataset
from corpus_studio.training.trainer import preflight_sft_dataset

_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
_FULL_LENGTH = {"add_special_tokens": True, "truncation": False}


class _Tokenizer:
    """A tokenizer that honors ``add_special_tokens``/``truncation`` the way Hugging Face does, records
    every call, and renders chat messages with an expanding template (role markers around content)."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(
        self,
        ids_for_text: Callable[[str], Sequence[int]],
        *,
        bos: int | None = None,
        eos: int | None = None,
        chat_template: str | None = None,
        fail_on_call: int | None = None,
        template_error: str | None = None,
    ) -> None:
        self.ids_for_text = ids_for_text
        self.bos = bos
        self.eos = eos
        self.chat_template = chat_template
        self.fail_on_call = fail_on_call
        self.template_error = template_error
        self.events: list[tuple[Any, ...]] = []
        self.template_calls: list[Any] = []

    @property
    def calls(self) -> list[tuple[Any, ...]]:
        return [event for event in self.events if event[0] == "tokenize"]

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        self.template_calls.append(messages)
        if self.template_error is not None:
            raise RuntimeError(self.template_error)
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.events.append(("tokenize", text, dict(kwargs)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("tokenizer wedged")
        ids = list(self.ids_for_text(text))
        if kwargs.get("add_special_tokens", True):
            ids = [self.bos] * (self.bos is not None) + ids + [self.eos] * (self.eos is not None)
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids}


def _long_or_short(text: str) -> list[int]:
    return list(range(1, 11)) if "LONG" in text else [1, 2]


def _execution(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    seq_len: int,
    allow: bool = False,
    sequence_allows: bool | None = None,
    fmt: str = "instruction",
) -> Any:
    """A planner-made full-parameter seal re-sealed over ``rows`` with the given sequence and policy."""
    execution = _sealed(tmp_path, "full_finetune")
    data = Path(execution.inputs.dataset.location)
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = stable_file_sha256(data)

    def _mutate(payload: dict[str, Any]) -> None:
        payload["sequence"]["max_sequence_len"] = seq_len
        payload["sequence"]["truncation_allowed"] = allow if sequence_allows is None else sequence_allows
        payload["data"]["truncation_policy"] = "allow" if allow else "refuse"
        batching = payload["batching"]
        batching["supervised_token_accumulation_target"] = (
            seq_len * batching["micro_batch_size"] * batching["fallback_grad_accumulation_steps"]
        )
        payload["inputs"]["dataset"]["content_sha256"] = digest
        payload["inputs"]["dataset"]["ref"]["hash"]["value"] = digest
        if fmt == "chat":
            formatter_id, formatter_sha256 = formatter_identity("chat")
            payload["data"].update(
                dataset_format="chat",
                formatter_id=formatter_id,
                formatter_sha256=formatter_sha256,
                chat_template_sha256=hashlib.sha256(_TEMPLATE.encode("utf-8")).hexdigest(),
            )

    return _reseal(execution, _mutate)


def _instruction_rows(long_at: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(3)]
    if long_at is not None:
        rows[long_at] = {"instruction": "LONG", "output": "LONG"}
    return rows


_CHAT_ROW = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
_POSITIONS = pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "final"])


# --- the shared preflight core ------------------------------------------------------------------------------


def test_the_shared_preflight_keeps_the_exact_untruncated_ids_it_measured(capsys):
    rows = [
        {"instruction": "LONG", "output": "LONG"},
        {"instruction": "", "output": ""},
        {"instruction": "q", "output": "a"},
    ]
    measured: list[str] = []

    def _encode(text: str) -> tuple[int, ...]:
        measured.append(text)
        return (1, 2, 3, 4, 5, 6) if "LONG" in text else (7,)

    class _NoDefaultCall:
        def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("an explicit encode replaces the default tokenizer call")

    preflight = preflight_sft_dataset(
        rows,
        dataset_format="instruction",
        sequence_len=4,
        truncation_allowed=True,
        tokenizer=_NoDefaultCall(),
        encode=_encode,
        keep_token_ids=True,
    )

    assert preflight.token_ids == [[1, 2, 3, 4, 5, 6], [7]]  # never cut at sequence_len
    assert preflight.texts == [measured[0], "", measured[1]]  # index-aligned with the rows
    assert preflight.unrenderable_rows == 1
    assert (preflight.ledger.n_examples, preflight.ledger.dropped_tokens) == (2, 2)
    assert preflight.report.n_truncated == 1
    assert "no renderable text" in capsys.readouterr().err


def test_the_shared_preflight_measures_with_the_single_argument_call_by_default():
    class _TextOnlyTokenizer:  # the adapter SFT fakes (and the HF default call) take the text alone
        def __call__(self, text: str) -> dict[str, list[str]]:
            return {"input_ids": text.split()}

    preflight = preflight_sft_dataset(
        _instruction_rows(),
        dataset_format="instruction",
        sequence_len=100,
        truncation_allowed=False,
        tokenizer=_TextOnlyTokenizer(),
    )

    assert preflight.token_ids is None
    assert preflight.unrenderable_rows == 0
    assert preflight.ledger.is_lossless and preflight.ledger.n_examples == 3


# --- the sealed policy and the row builder --------------------------------------------------------------


def test_truncation_needs_both_the_data_policy_and_the_sequence_flag(tmp_path):
    rows = [{"instruction": "LONG", "output": "LONG"}]
    refuse = _execution(tmp_path, rows, seq_len=4)
    sequence_only = _execution(tmp_path, rows, seq_len=4, sequence_allows=True)  # a valid seal
    allowed = _execution(tmp_path, rows, seq_len=4, allow=True)
    unvalidated = allowed.model_copy(
        update={"sequence": allowed.sequence.model_copy(update={"truncation_allowed": False})}
    )

    permitted = [
        full_finetune_truncation_permitted(execution)
        for execution in (refuse, sequence_only, allowed, unvalidated)
    ]

    assert permitted == [False, False, True, False]
    with pytest.raises(FullFinetuneDataRefusal, match="supervised"):
        prepare_full_finetune_dataset(sequence_only, rows, _Tokenizer(_long_or_short))
    with pytest.raises(ValidationError, match="would silently truncate"):
        _execution(tmp_path, rows, seq_len=4, allow=True, sequence_allows=False)


def test_the_row_builder_refuses_overlength_without_a_lossy_policy_and_names_the_row():
    with pytest.raises(FullFinetuneDataRefusal, match="rendered row 2: an SFT row has 5 tokens"):
        build_full_finetune_rows([[1], [1, 2], [1, 2, 3, 4, 5]], 4, 0, truncation_permitted=False)


def test_the_row_builder_cuts_explicitly_only_under_a_lossy_policy():
    rows = build_full_finetune_rows([[1, 2, 3, 4, 5], [9]], 4, 0, truncation_permitted=True)

    assert rows == [
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [9, 0, 0, 0], "labels": [9, -100, -100, -100], "attention_mask": [1, 0, 0, 0]},
    ]


def test_padding_keeps_a_trailing_eos_that_equals_the_pad_id_supervised():
    assert pad_sft_row([101, 7, 8, 0], 4, 0) == {
        "input_ids": [101, 7, 8, 0],
        "labels": [101, 7, 8, 0],
        "attention_mask": [1, 1, 1, 1],
    }
    assert pad_sft_row([101, 7, 0], 4, 0) == {
        "input_ids": [101, 7, 0, 0],
        "labels": [101, 7, 0, -100],
        "attention_mask": [1, 1, 1, 0],
    }


# --- the data preparation -----------------------------------------------------------------------------------


@_POSITIONS
def test_an_overlength_row_anywhere_is_refused_after_every_row_is_measured(tmp_path, position):
    rows = _instruction_rows(long_at=position)
    execution = _execution(tmp_path, rows, seq_len=4)
    tokenizer = _Tokenizer(_long_or_short)
    stages: list[tuple[str, str]] = []
    records: list[Any] = []

    with pytest.raises(FullFinetuneDataRefusal, match=r"6 supervised .* across 1 example") as refused:
        prepare_full_finetune_dataset(
            execution,
            rows,
            tokenizer,
            stage_callback=lambda name, message: stages.append((name, message)),
            coverage_callback=records.append,
        )

    assert str(refused.value).isascii()
    assert len(tokenizer.calls) == 3  # every row was measured, at full length
    assert all(kwargs == _FULL_LENGTH for _event, _text, kwargs in tokenizer.calls)
    assert records == []
    analysis = [message for name, message in stages if name == "truncation_analysis"]
    assert analysis and not any(message.startswith("verified") for message in analysis)


def test_exactly_at_limit_rows_train_every_measured_id_including_special_tokens(tmp_path):
    rows = [{"instruction": "FULL", "output": "x"}, {"instruction": "q", "output": "a"}]
    execution = _execution(tmp_path, rows, seq_len=5)
    # The tokenizer adds BOS=101 and EOS=0 itself; EOS equals the pad id.
    tokenizer = _Tokenizer(lambda text: [7, 8, 9] if "FULL" in text else [7], bos=101, eos=0)

    prepared = prepare_full_finetune_dataset(execution, rows, tokenizer)

    assert prepared.rows == [
        {"input_ids": [101, 7, 8, 9, 0], "labels": [101, 7, 8, 9, 0], "attention_mask": [1] * 5},
        {
            "input_ids": [101, 7, 0, 0, 0],
            "labels": [101, 7, 0, -100, -100],
            "attention_mask": [1, 1, 1, 0, 0],
        },
    ]
    ledger = prepared.coverage.ledger
    assert (ledger.input_tokens_total, ledger.retained_tokens, ledger.dropped_tokens) == (8, 8, 0)
    assert ledger.supervised_dropped == 0 and not prepared.coverage.truncation_permitted


def test_one_token_past_the_limit_is_refused(tmp_path):
    rows = [{"instruction": "FULL", "output": "x"}]
    execution = _execution(tmp_path, rows, seq_len=4)

    with pytest.raises(FullFinetuneDataRefusal, match=r"1 supervised .* across 1 example"):
        prepare_full_finetune_dataset(
            execution, rows, _Tokenizer(lambda _text: [7, 8, 9], bos=101, eos=0)
        )


def test_allowed_truncation_is_explicit_and_recorded_deterministically(tmp_path, capsys):
    rows = [{"instruction": "LONG", "output": "LONG"}, {"instruction": "q", "output": "a"}]
    execution = _execution(tmp_path, rows, seq_len=4, allow=True)
    stages: list[tuple[str, str]] = []
    records: list[Any] = []

    first = prepare_full_finetune_dataset(
        execution,
        rows,
        _Tokenizer(_long_or_short),
        stage_callback=lambda name, message: stages.append((name, message)),
        coverage_callback=records.append,
    )
    second = prepare_full_finetune_dataset(
        execution, rows, _Tokenizer(_long_or_short), coverage_callback=records.append
    )

    assert first.rows == second.rows
    assert first.rows[0] == {
        "input_ids": [1, 2, 3, 4],
        "labels": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
    }
    assert first.rows[1]["labels"] == [1, 2, -100, -100]
    assert records == [first.coverage, second.coverage]
    evidence = records[0].evidence()
    assert evidence == records[1].evidence()
    assert json.loads(json.dumps(evidence)) == evidence
    ledger = evidence["ledger"]
    assert evidence["execution_configuration_hash"] == execution.configuration_hash
    assert (evidence["truncation_policy"], evidence["sealed_rows"], evidence["unrenderable_rows"]) == (
        "allow",
        2,
        0,
    )
    assert (
        ledger["input_tokens_total"],
        ledger["retained_tokens"],
        ledger["dropped_tokens"],
        ledger["supervised_dropped"],
        ledger["boundary_severances"],
    ) == (12, 6, 6, 6, 1)
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert evidence["ledger_sha256"] == hashlib.sha256(canonical).hexdigest()
    summary = records[0].summary()
    assert summary == records[1].summary() and summary.isascii()
    for part in (
        f"token coverage for execution {execution.configuration_hash}:",
        "policy=allow",
        "dropped_tokens=6",
        "severed_examples=1",
        f"ledger_sha256={evidence['ledger_sha256']}",
    ):
        assert part in summary
    # The coverage callback owns the record; the stage stream carries only the shared preflight stages.
    assert not any(message.startswith("token coverage") for _name, message in stages)
    err = capsys.readouterr().err
    assert "REFUSED" in err and "TRUNCATION" in err  # the adapter SFT lane's own lossy notice


def test_chat_template_expansion_is_measured_with_the_bound_tokenizer(tmp_path):
    rows = [_CHAT_ROW]
    tokenizer = _Tokenizer(lambda text: [ord(char) for char in text])  # one id per character
    expansion = "<|user|>hi<|end|><|assistant|>yo<|end|>"

    with pytest.raises(FullFinetuneDataRefusal, match="supervised"):
        prepare_full_finetune_dataset(
            _execution(tmp_path, rows, seq_len=8, fmt="chat"), rows, tokenizer
        )
    prepared = prepare_full_finetune_dataset(
        _execution(tmp_path, rows, seq_len=64, fmt="chat"), rows, tokenizer
    )

    assert len("hiyo") <= 8 < len(expansion)  # the raw content fits; its expansion does not
    assert tokenizer.template_calls == [_CHAT_ROW["messages"]] * 2
    assert [text for _event, text, _kwargs in tokenizer.calls] == [expansion] * 2
    assert prepared.rows[0]["input_ids"][: len(expansion)] == [ord(char) for char in expansion]


def test_a_chat_template_failure_is_a_data_refusal(tmp_path):
    rows = [_CHAT_ROW]
    tokenizer = _Tokenizer(_long_or_short, template_error="bad template")

    with pytest.raises(FullFinetuneDataRefusal, match="the tokenizer chat template failed: bad template"):
        prepare_full_finetune_dataset(
            _execution(tmp_path, rows, seq_len=64, fmt="chat"), rows, tokenizer
        )
    assert tokenizer.calls == []


def test_an_unrenderable_row_is_refused_under_the_default_policy(tmp_path):
    rows = [{"instruction": "q", "output": "a"}, {"instruction": "", "output": ""}]

    with pytest.raises(FullFinetuneDataRefusal, match=r"1 of 2 row\(s\) produced no renderable text"):
        prepare_full_finetune_dataset(
            _execution(tmp_path, rows, seq_len=4), rows, _Tokenizer(_long_or_short)
        )


def test_an_unrenderable_row_is_dropped_and_counted_under_a_sealed_lossy_policy(tmp_path, capsys):
    rows = [{"instruction": "q", "output": "a"}, {"instruction": "", "output": ""}]
    stages: list[tuple[str, str]] = []

    prepared = prepare_full_finetune_dataset(
        _execution(tmp_path, rows, seq_len=4, allow=True),
        rows,
        _Tokenizer(_long_or_short),
        stage_callback=lambda name, message: stages.append((name, message)),
    )

    assert len(prepared.rows) == 1
    assert (prepared.coverage.sealed_rows, prepared.coverage.unrenderable_rows) == (2, 1)
    # Without a coverage callback the record is streamed as the final truncation_analysis message.
    assert stages[-1][0] == "truncation_analysis"
    for part in ("policy=allow", "rows=2", "examples=1", "unrenderable_rows=1"):
        assert part in stages[-1][1]
    assert "no renderable text" in capsys.readouterr().err


def test_a_dataset_with_no_renderable_row_is_refused_even_under_a_lossy_policy(tmp_path):
    rows = [{"instruction": "", "output": ""}]

    with pytest.raises(FullFinetuneDataRefusal, match="rendered no trainable rows"):
        prepare_full_finetune_dataset(
            _execution(tmp_path, rows, seq_len=4, allow=True), rows, _Tokenizer(_long_or_short)
        )


def test_a_row_that_tokenizes_to_nothing_is_refused_not_trained_as_padding(tmp_path):
    rows = [{"instruction": "q", "output": "a"}, {"instruction": "EMPTY", "output": "x"}]
    tokenizer = _Tokenizer(lambda text: [] if "EMPTY" in text else [1, 2])

    with pytest.raises(FullFinetuneDataRefusal, match="rendered row 1: an SFT row tokenized to zero"):
        prepare_full_finetune_dataset(_execution(tmp_path, rows, seq_len=4), rows, tokenizer)


def test_a_tokenizer_failure_is_a_data_refusal_without_a_coverage_record(tmp_path):
    rows = [{"instruction": f"q{i}", "output": "a"} for i in range(5)]
    stages: list[tuple[str, str]] = []
    records: list[Any] = []

    with pytest.raises(FullFinetuneDataRefusal, match="truncation analysis failed: tokenizer wedged"):
        prepare_full_finetune_dataset(
            _execution(tmp_path, rows, seq_len=4),
            rows,
            _Tokenizer(_long_or_short, fail_on_call=3),
            stage_callback=lambda name, message: stages.append((name, message)),
            coverage_callback=records.append,
        )
    assert records == []
    assert not any(message.startswith("verified") for _name, message in stages)


# --- the real worker: the preflight precedes the kernel probe and every weight --------------------------


class _StopAtDataset(Exception):
    """Raised by the fake ``Dataset.from_list`` with the rows the worker would train."""


def _worker_stack(monkeypatch: pytest.MonkeyPatch, tokenizer: _Tokenizer) -> FakeStack:
    stack = FakeStack(monkeypatch)
    tokenizer.events = stack.events  # tokenizer calls land on the same timeline as every load

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(location: str, **kwargs: Any) -> _Tokenizer:
            stack.events.append(("load", "AutoTokenizer", location, kwargs))
            return tokenizer

    class Dataset:
        @staticmethod
        def from_list(rows: list[dict[str, Any]]) -> Any:
            stack.events.append(("dataset_from_list", len(rows)))
            raise _StopAtDataset(rows)

    monkeypatch.setattr(stack.transformers, "AutoTokenizer", AutoTokenizer)
    monkeypatch.setattr(stack.transformers, "Trainer", object, raising=False)
    monkeypatch.setattr(stack.transformers, "TrainingArguments", object, raising=False)
    monkeypatch.setattr(sys.modules["datasets"], "Dataset", Dataset)
    return stack


def _kinds(stack: FakeStack) -> list[str]:
    return [event[1] if event[0] == "load" else event[0] for event in stack.events]


@_POSITIONS
def test_the_worker_refuses_before_the_kernel_probe_or_any_weights(tmp_path, monkeypatch, position):
    rows = _instruction_rows(long_at=position)
    execution = _execution(tmp_path, rows, seq_len=4)
    stack = _worker_stack(monkeypatch, _Tokenizer(_long_or_short))
    stages = _Stages()

    with pytest.raises(FullFinetuneDataRefusal, match="6 supervised"):
        run_full_finetune(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
            stage_callback=stages,
        )

    kinds = _kinds(stack)
    assert kinds.count("tokenize") == 3
    assert [load[0] for load in stack.loads()] == ["AutoTokenizer"]
    assert not {"sdp_toggle", "probe_randn", "probe_sdpa", "dataset_from_list"} & set(kinds)
    assert stages.names[-1] == "truncation_analysis"
    assert "attention_policy_applied" not in stages.names and "model_load" not in stages.names
    assert not (tmp_path / "out").exists()


def test_the_worker_trains_the_measured_ids_and_loads_weights_only_after_the_preflight(
    tmp_path, monkeypatch
):
    rows = [{"instruction": "FULL", "output": "x"}, {"instruction": "q", "output": "a"}]
    execution = _execution(tmp_path, rows, seq_len=5)
    stack = _worker_stack(
        monkeypatch,
        _Tokenizer(lambda text: [7, 8, 9] if "FULL" in text else [7], bos=101, eos=0),
    )
    stages = _Stages()

    def _record(coverage: Any) -> None:
        stack.events.append(("coverage", coverage))

    with pytest.raises(_StopAtDataset) as stopped:
        run_full_finetune(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            stage_callback=stages,
            coverage_callback=_record,
        )

    assert stopped.value.args[0] == [
        {"input_ids": [101, 7, 8, 9, 0], "labels": [101, 7, 8, 9, 0], "attention_mask": [1] * 5},
        {
            "input_ids": [101, 7, 0, 0, 0],
            "labels": [101, 7, 0, -100, -100],
            "attention_mask": [1, 1, 1, 0, 0],
        },
    ]
    kinds = _kinds(stack)
    last_tokenize = max(index for index, kind in enumerate(kinds) if kind == "tokenize")
    assert kinds.index("AutoTokenizer") < kinds.index("tokenize")
    assert last_tokenize < kinds.index("coverage") < kinds.index("sdp_toggle")
    assert kinds.index("sdp_toggle") < kinds.index("AutoModelForCausalLM")
    assert all(event[2] == _FULL_LENGTH for event in stack.events if event[0] == "tokenize")
    first_seen = list(dict.fromkeys(stages.names))
    assert first_seen[:5] == [
        "tokenizer_load",
        "dataset_formatting",
        "truncation_analysis",
        "attention_policy_applied",
        "model_load",
    ]


def test_the_worker_measures_chat_expansion_with_the_pinned_tokenizer_it_verified(
    tmp_path, monkeypatch
):
    rows = [_CHAT_ROW]
    execution = _execution(tmp_path, rows, seq_len=8, fmt="chat")
    tokenizer = _Tokenizer(lambda text: [ord(char) for char in text], chat_template=_TEMPLATE)
    stack = _worker_stack(monkeypatch, tokenizer)

    with pytest.raises(FullFinetuneDataRefusal, match="supervised"):
        run_full_finetune(execution, dataset=read_verified_dataset(execution.inputs.dataset))

    assert stack.loads() == [("AutoTokenizer", TOKENIZER, {"trust_remote_code": False, "revision": B40})]
    assert tokenizer.template_calls == [_CHAT_ROW["messages"]]

    drifted = _Tokenizer(lambda text: [ord(char) for char in text], chat_template=_TEMPLATE + " ")
    _worker_stack(monkeypatch, drifted)
    with pytest.raises(Exception, match="chat template changed after planning"):
        run_full_finetune(execution, dataset=read_verified_dataset(execution.inputs.dataset))
    assert drifted.template_calls == [] and drifted.calls == []  # the digest is checked first


# --- the runner: classification and structured evidence ------------------------------------------------


def _full_finetune_plan(
    tmp_path: Path, rows: list[dict[str, Any]], *, sequence_len: int, allow: bool = False
) -> Any:
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return _plan(
        _profile(cc_major=8),
        _report(),
        dataset_path=str(data),
        dataset_content_sha256=stable_file_sha256(data),
        task_type="sft",
        adapter_method="full_finetune",
        export_format="merged_safetensors",
        sequence_len=sequence_len,
        truncation_allowed=allow,
    )


def _long_20_or_short(text: str) -> list[int]:
    return list(range(1, 21)) if "LONG" in text else [1, 2]


def test_the_runner_refuses_an_overlength_dataset_before_any_weights_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = _full_finetune_plan(tmp_path, _instruction_rows(long_at=2), sequence_len=8)
    stack = _worker_stack(monkeypatch, _Tokenizer(_long_20_or_short))
    timeline: list[Any] = []

    result = execute_run(
        plan, build_lane_runner(required_runner_lane(plan)), run_id="run-ff", sink=timeline.append
    )

    failure = result.manifest.failure
    assert result.manifest.state == "failed" and failure is not None
    assert (failure.taxonomy, failure.stage) == (
        FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
        StageMarker.truncation_analysis,
    )
    assert "12 supervised" in failure.message
    assert failure.remediation is not None and "--allow-truncation" in failure.remediation
    assert failure.remediation.isascii()
    assert [load[0] for load in stack.loads()] == ["AutoTokenizer"]
    stage_events = [event for event in _events(timeline) if event.event_type == "stage"]
    assert StageMarker.model_load not in [event.stage for event in stage_events]
    assert not [event for event in stage_events if event.payload and "ledger" in event.payload]
    execution = plan.resolved_full_finetune_execution
    assert not run_scoped_training_output(execution, "run-ff").exists()


@pytest.mark.parametrize(("allow", "dropped"), [(False, 0), (True, 12)], ids=["refuse", "allow"])
def test_the_runner_records_token_coverage_as_structured_evidence_before_weights_load(
    tmp_path, monkeypatch, allow, dropped
):
    monkeypatch.chdir(tmp_path)
    rows = [{"instruction": "LONG", "output": "LONG"}, {"instruction": "q", "output": "a"}]
    plan = _full_finetune_plan(tmp_path, rows, sequence_len=8 if allow else 20, allow=allow)
    _worker_stack(monkeypatch, _Tokenizer(_long_20_or_short))
    records = tmp_path / "records"

    execute_run(
        plan, build_lane_runner(required_runner_lane(plan)), run_id="run-ff", out_dir=records
    )

    persisted = [
        json.loads(line)
        for line in (records / "runs/run-ff/RunEvents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    coverage = [
        index
        for index, event in enumerate(persisted)
        if event.get("stage") == "truncation_analysis" and event.get("payload")
    ]
    assert len(coverage) == 1
    event = persisted[coverage[0]]
    execution = plan.resolved_full_finetune_execution
    assert event["payload"]["execution_configuration_hash"] == execution.configuration_hash
    assert event["payload"]["truncation_policy"] == ("allow" if allow else "refuse")
    assert event["payload"]["ledger"]["dropped_tokens"] == dropped
    assert event["message"].startswith(f"token coverage for execution {execution.configuration_hash}")
    model_load = next(
        index for index, item in enumerate(persisted) if item.get("stage") == "model_load"
    )
    assert coverage[0] < model_load


@pytest.mark.parametrize(
    ("error", "taxonomy", "stage"),
    [
        (
            FullFinetuneDataRefusal("an unrenderable row"),
            FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
            StageMarker.dataset_formatting,
        ),
        (FullFinetuneError("update failed"), FailureTaxonomy.UPDATE_FAILURE, StageMarker.optimizer_step),
    ],
    ids=["data-refusal", "update-failure"],
)
def test_the_runner_attributes_a_data_refusal_to_the_reached_stage_and_keeps_update_failures(
    tmp_path, monkeypatch, error, taxonomy, stage
):
    monkeypatch.chdir(tmp_path)
    plan = _full_finetune_plan(tmp_path, _instruction_rows(), sequence_len=8)

    def _fake_worker(execution, *, dataset, output_dir=None, stage_callback=None, coverage_callback=None):
        stage_callback("tokenizer_load", "loaded and verified the sealed tokenizer")
        stage_callback("dataset_formatting", "formatted all 3 dataset rows")
        raise error

    monkeypatch.setattr(full_finetune_trainer, "run_full_finetune", _fake_worker)

    result = execute_run(plan, build_lane_runner(required_runner_lane(plan)), run_id="run-ff")

    failure = result.manifest.failure
    assert failure is not None
    assert (failure.taxonomy, failure.stage) == (taxonomy, stage)


def test_the_full_parameter_worker_module_stays_torch_free_at_import():
    heavy = ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "safetensors")
    code = (
        "import sys\n"
        "import corpus_studio.training.full_finetune_trainer\n"
        f"print(sorted(name for name in {heavy!r} if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]"


# --- remediation honesty: only a policy refusal points at a lossy policy -------------------------------


def _chat_template_failure(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = [_CHAT_ROW]
    tokenizer = _Tokenizer(_long_or_short, template_error="bad template")
    return _execution(tmp_path, rows, seq_len=64, fmt="chat"), rows, tokenizer


def _tokenizer_failure(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = [{"instruction": f"q{i}", "output": "a"} for i in range(3)]
    return _execution(tmp_path, rows, seq_len=4), rows, _Tokenizer(_long_or_short, fail_on_call=2)


def _empty_ids(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = [{"instruction": "EMPTY", "output": "x"}]
    tokenizer = _Tokenizer(lambda text: [] if "EMPTY" in text else [1, 2])
    return _execution(tmp_path, rows, seq_len=4), rows, tokenizer


def _nothing_renders(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = [{"instruction": "", "output": ""}]
    return _execution(tmp_path, rows, seq_len=4, allow=True), rows, _Tokenizer(_long_or_short)


def _overlength(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = _instruction_rows(long_at=1)
    return _execution(tmp_path, rows, seq_len=4), rows, _Tokenizer(_long_or_short)


def _unrenderable(tmp_path: Path) -> tuple[Any, list[dict[str, Any]], _Tokenizer]:
    rows = [{"instruction": "q", "output": "a"}, {"instruction": "", "output": ""}]
    return _execution(tmp_path, rows, seq_len=4), rows, _Tokenizer(_long_or_short)


@pytest.mark.parametrize(
    ("case", "policy_refusal"),
    [
        (_overlength, True),
        (_unrenderable, True),
        (_chat_template_failure, False),
        (_tokenizer_failure, False),
        (_empty_ids, False),
        (_nothing_renders, False),
    ],
    ids=["overlength", "unrenderable", "template", "tokenizer", "empty-ids", "nothing-renders"],
)
def test_only_the_no_truncation_policy_marks_a_refusal_a_policy_refusal(tmp_path, case, policy_refusal):
    execution, rows, tokenizer = case(tmp_path)

    with pytest.raises(FullFinetuneDataRefusal) as refused:
        prepare_full_finetune_dataset(execution, rows, tokenizer)

    assert refused.value.policy_refusal is policy_refusal


def test_the_row_builder_keeps_the_policy_classification_of_its_refusal():
    with pytest.raises(FullFinetuneDataRefusal) as overlength:
        build_full_finetune_rows([[1, 2, 3, 4, 5]], 4, 0, truncation_permitted=False)
    with pytest.raises(FullFinetuneDataRefusal) as empty:
        build_full_finetune_rows([[1], []], 4, 0, truncation_permitted=True)

    assert overlength.value.policy_refusal is True
    assert empty.value.policy_refusal is False
    assert str(empty.value).startswith("rendered row 1: ")


@pytest.mark.parametrize(
    ("error", "suggests_allow_truncation"),
    [
        (FullFinetuneDataRefusal("7 supervised tokens would be dropped", policy_refusal=True), True),
        (FullFinetuneDataRefusal("the tokenizer chat template failed: bad template"), False),
    ],
    ids=["policy", "formatter"],
)
def test_the_runner_suggests_a_lossy_policy_only_for_a_policy_refusal(
    tmp_path, monkeypatch, error, suggests_allow_truncation
):
    monkeypatch.chdir(tmp_path)
    plan = _full_finetune_plan(tmp_path, _instruction_rows(), sequence_len=8)

    def _fake_worker(execution, *, dataset, output_dir=None, stage_callback=None, coverage_callback=None):
        stage_callback("dataset_formatting", "formatting 3 sealed dataset rows")
        raise error

    monkeypatch.setattr(full_finetune_trainer, "run_full_finetune", _fake_worker)

    result = execute_run(plan, build_lane_runner(required_runner_lane(plan)), run_id="run-ff")

    failure = result.manifest.failure
    assert failure is not None and failure.remediation is not None
    assert (failure.taxonomy, failure.stage) == (
        FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
        StageMarker.dataset_formatting,
    )
    assert ("--allow-truncation" in failure.remediation) is suggests_allow_truncation
    assert failure.remediation.isascii()


def test_a_real_tokenizer_failure_gets_a_remediation_without_a_lossy_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = _full_finetune_plan(tmp_path, _instruction_rows(), sequence_len=8)
    stack = _worker_stack(monkeypatch, _Tokenizer(_long_20_or_short, fail_on_call=2))

    result = execute_run(plan, build_lane_runner(required_runner_lane(plan)), run_id="run-ff")

    failure = result.manifest.failure
    assert failure is not None and failure.remediation is not None
    assert (failure.taxonomy, failure.stage) == (
        FailureTaxonomy.UNSUPPORTED_CONFIGURATION,
        StageMarker.truncation_analysis,
    )
    assert "truncation analysis failed: tokenizer wedged" in failure.message
    assert "--allow-truncation" not in failure.remediation
    assert [load[0] for load in stack.loads()] == ["AutoTokenizer"]


def test_the_lossy_truncation_notice_on_stderr_is_ascii(tmp_path, capsys):
    rows = [{"instruction": "LONG", "output": "LONG"}, {"instruction": "q", "output": "a"}]

    prepare_full_finetune_dataset(
        _execution(tmp_path, rows, seq_len=4, allow=True), rows, _Tokenizer(_long_or_short)
    )

    err = capsys.readouterr().err
    assert "will be CUT - the end" in err
    assert err.isascii()
