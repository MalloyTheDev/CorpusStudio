import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.arena.models import ArenaPrompt, build_arena_report
from corpus_studio.arena.runner import load_prompt_suite, responses_for_prompt, run_arena
from corpus_studio.cli import app
from corpus_studio.model_backends.base import BackendGenerateResponse

runner = CliRunner()


class EchoBackend:
    """Echoes the model name + whether a system prompt was supplied."""

    def __init__(self, model: str):
        self.model = model

    def generate(self, request):
        tag = "sys" if request.messages else "plain"
        text = "" if self.model == "silent" else f"{self.model}:{tag}"
        return BackendGenerateResponse(text=text, model_name=self.model)


PROMPTS = [
    ArenaPrompt(id="p1", prompt="Explain recursion."),
    ArenaPrompt(id="p2", prompt="Say hi.", system="You are terse."),
]


def test_run_arena_collects_every_model_response():
    report = run_arena(PROMPTS, [("a", EchoBackend("a")), ("b", EchoBackend("b"))])
    assert report.prompt_count == 2
    assert report.models == ["a", "b"]
    assert len(report.responses) == 4  # 2 prompts x 2 models
    assert {s.model: s.response_count for s in report.model_summaries} == {"a": 2, "b": 2}


def test_system_prompt_routed_as_messages():
    report = run_arena([PROMPTS[1]], [("a", EchoBackend("a"))])
    assert report.responses[0].text == "a:sys"


def test_plain_prompt_routed_as_prompt():
    report = run_arena([PROMPTS[0]], [("a", EchoBackend("a"))])
    assert report.responses[0].text == "a:plain"


def test_limit_caps_prompts():
    report = run_arena(PROMPTS, [("a", EchoBackend("a"))], limit=1)
    assert report.prompt_count == 1
    assert len(report.responses) == 1


def test_empty_responses_are_counted():
    report = run_arena([PROMPTS[0]], [("silent", EchoBackend("silent"))])
    assert report.model_summaries[0].empty_response_count == 1


def test_responses_for_prompt_keys_by_model():
    report = run_arena(PROMPTS, [("a", EchoBackend("a")), ("b", EchoBackend("b"))])
    assert responses_for_prompt(report, "p1") == {"a": "a:plain", "b": "b:plain"}


def test_build_arena_report_is_pure():
    from corpus_studio.arena.models import ArenaResponse

    responses = [ArenaResponse(prompt_id="p1", model="a", text="hi")]
    report = build_arena_report([PROMPTS[0]], ["a"], responses)
    assert report.model_summaries[0].response_count == 1


# --- prompt suite loading ----------------------------------------------------

def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_prompt_suite_assigns_ids_and_skips_empty(tmp_path: Path):
    src = tmp_path / "suite.jsonl"
    _write(src, [{"prompt": "one"}, {"prompt": "  "}, {"id": "custom", "prompt": "three", "system": "S"}])
    prompts = load_prompt_suite(src)
    assert [p.id for p in prompts] == ["prompt-1", "custom"]
    assert prompts[1].system == "S"


# --- CLI (backend monkeypatched, no network) ---------------------------------

def test_cli_arena_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "suite.jsonl"
    _write(src, [{"prompt": "Explain recursion."}, {"prompt": "Say hi."}])
    monkeypatch.setattr(cli, "_build_backend", lambda **kwargs: EchoBackend(kwargs["model"]))

    result = runner.invoke(
        app, ["arena-run", str(src), "--model", "alpha", "--model", "beta"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["prompt_count"] == 2
    assert report["models"] == ["alpha", "beta"]
    assert len(report["responses"]) == 4
    assert report["generated_at"]


def test_cli_arena_run_rejects_empty_suite(tmp_path: Path):
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    result = runner.invoke(app, ["arena-run", str(src), "--model", "alpha"])
    assert result.exit_code == 1
