import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.tokenization import estimate as estimate_mod
from corpus_studio.tokenization.estimate import estimate_tokens
from corpus_studio.training.estimators import (
    build_training_token_budget,
    estimate_row_tokens,
    estimate_token_budget,
)

runner = CliRunner()


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding(list(range(len(text.split()))))


def test_budget_uses_the_target_model_tokenizer_when_available(monkeypatch):
    # When the target model's tokenizer loads (optional `tokenizers` extra), the
    # budget's method reports it, so the count is exact for that model.
    estimate_mod._hf_tokenizer_cache.clear()
    monkeypatch.setattr(estimate_mod, "_load_hf_tokenizer", lambda model_id: _FakeTokenizer())

    budget = build_training_token_budget(
        [{"instruction": "a b c", "output": "d e"}], sequence_len=2048, model_id="fake/model"
    )

    assert budget.method == "hf:fake/model"


def test_budget_falls_back_without_a_model_tokenizer(monkeypatch):
    estimate_mod._hf_tokenizer_cache.clear()
    monkeypatch.setattr(estimate_mod, "_load_hf_tokenizer", lambda model_id: None)

    budget = build_training_token_budget(
        [{"instruction": "a b c", "output": "d e"}], sequence_len=2048, model_id="fake/model"
    )

    assert budget.method == "heuristic"  # no tokenizer + no tiktoken -> heuristic


def test_chat_row_counts_the_template_overhead():
    row = {
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is recursion?"},
            {"role": "assistant", "content": "A function calling itself."},
        ]
    }
    # Raw content-only count (what the old flat extraction produced) vs the chat-aware
    # count, which must be higher by the per-message + per-conversation overhead.
    content_only = sum(estimate_tokens(m["content"]) for m in row["messages"])
    chat_aware = estimate_row_tokens(row)

    assert chat_aware > content_only
    assert chat_aware == content_only + 4 * 3 + 3  # 3 messages * 4 + 3 per conversation


def test_non_chat_row_is_unchanged():
    row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    # A non-chat row keeps the flat text-extraction estimate (no chat overhead applied).
    assert estimate_row_tokens(row) == estimate_tokens("Explain variables. A variable stores a value.")


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_budget_counts_and_truncation():
    rows = [
        {"instruction": "q", "output": "word " * 100},  # well over 50 tokens
        {"instruction": "q", "output": "short"},
    ]
    budget = build_training_token_budget(rows, sequence_len=50)
    assert budget.example_count == 2
    assert budget.max_tokens_in_example >= 100
    assert budget.examples_over_sequence_len == 1
    assert budget.tokens_per_epoch <= budget.estimated_tokens
    assert budget.tokens_per_epoch >= 50
    assert budget.method in {"heuristic", "tiktoken"}


def test_build_budget_empty_rows():
    budget = build_training_token_budget([], sequence_len=128)
    assert budget.example_count == 0
    assert budget.estimated_tokens == 0
    assert budget.sequence_len == 128


def test_estimate_token_budget_uses_shared_estimator():
    estimate = estimate_token_budget(["abcd", "abcdefgh"])
    assert estimate.example_count == 2
    assert estimate.estimated_tokens == 3  # 1 + 2 via the shared estimator


def test_cli_training_config_reports_token_budget(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    _write(
        dataset,
        [
            {"instruction": "explain recursion", "output": "word " * 60},
            {"instruction": "explain loops", "output": "a short answer"},
        ],
    )
    out = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        [
            "training-config",
            str(dataset),
            "instruction",
            "--output-path",
            str(out),
            "--base-model",
            "some-base-model",
            "--sequence-len",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "token_budget" in payload
    budget = payload["token_budget"]
    assert budget["example_count"] == 2
    assert budget["sequence_len"] == 50
    assert budget["examples_over_sequence_len"] >= 1
    assert any("truncated" in warning for warning in payload["warnings"])
