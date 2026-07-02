import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.arena.judge import judge_arena, parse_judgment
from corpus_studio.arena.models import ArenaPrompt, build_arena_report
from corpus_studio.arena.runner import load_prompt_suite, responses_for_prompt, run_arena
from corpus_studio.cli import app
from corpus_studio.model_backends.base import BackendGenerateResponse
from corpus_studio.providers.policy import ProviderPolicyError, resolve_policy

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


# --- judging -----------------------------------------------------------------

class JudgeBackend:
    """Judge that always prefers model 'a' with fixed scores."""

    def __init__(self, text: str | None = None):
        self._text = text

    def generate(self, request):
        text = self._text if self._text is not None else json.dumps(
            {"scores": {"a": 90, "b": 40}, "winner": "a", "rationale": "clearer"}
        )
        return BackendGenerateResponse(text=text, model_name="judge")


def test_parse_judgment_validates_against_candidates():
    judgment = parse_judgment(
        "p1",
        {"a": "x", "b": "y"},
        json.dumps({"scores": {"a": 80, "b": 60, "ghost": 99}, "winner": "a", "rationale": "ok"}),
    )
    assert judgment.parsed is True
    assert judgment.winner == "a"
    assert judgment.scores == {"a": 80.0, "b": 60.0}  # unknown 'ghost' dropped


def test_parse_judgment_falls_back_to_top_score_on_bad_winner():
    judgment = parse_judgment(
        "p1", {"a": "x", "b": "y"}, json.dumps({"scores": {"a": 30, "b": 70}, "winner": "nope"})
    )
    assert judgment.winner == "b"


def test_parse_judgment_marks_unparseable():
    judgment = parse_judgment("p1", {"a": "x"}, "not json at all")
    assert judgment.parsed is False
    assert judgment.winner == ""


def test_judge_arena_aggregates_wins_and_scores():
    report = run_arena(PROMPTS, [("a", EchoBackend("a")), ("b", EchoBackend("b"))])
    judged = judge_arena(report, JudgeBackend(), "judge")
    assert judged.judge_model == "judge"
    assert len(judged.judgments) == 2
    summary = {s.model: s for s in judged.model_summaries}
    assert summary["a"].win_count == 2  # judge always picks 'a'
    assert summary["a"].average_judge_score == 90.0
    assert summary["b"].win_count == 0


def test_judge_arena_blocks_non_evaluator_policy():
    report = run_arena([PROMPTS[0]], [("a", EchoBackend("a"))])
    # A policy with the evaluator role blocked cannot judge.
    from corpus_studio.providers.policy import ProviderPolicy, ProviderRole

    policy = ProviderPolicy(provider_id="x", blocked_roles=[ProviderRole.EVALUATOR])
    with pytest.raises(ProviderPolicyError):
        judge_arena(report, JudgeBackend(), "judge", policy=policy)


def test_evaluator_only_provider_may_judge():
    # OpenAI is evaluator-only, which is exactly what judging needs.
    report = run_arena([PROMPTS[0]], [("a", EchoBackend("a"))])
    judged = judge_arena(report, JudgeBackend(), "gpt-4o", policy=resolve_policy("openai"))
    assert judged.judgments[0].winner == "a"


def test_save_and_load_arena_report_roundtrip(tmp_path: Path):
    from corpus_studio.arena.storage import (
        list_arena_reports,
        load_arena_report,
        save_arena_report,
    )

    report = run_arena(PROMPTS, [("a", EchoBackend("a"))], generated_at="2026-07-02T00:00:00Z")
    path = save_arena_report(tmp_path, report, "my suite")
    assert path.name == "my_suite.json"

    reloaded = load_arena_report(path)
    assert reloaded.prompt_count == report.prompt_count
    assert reloaded.generated_at == "2026-07-02T00:00:00Z"
    assert [p.name for p in list_arena_reports(tmp_path)] == ["my_suite.json"]


def test_cli_arena_run_saves_to_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "chat_suite.jsonl"
    _write(src, [{"prompt": "hi"}])
    monkeypatch.setattr(cli, "_build_backend", lambda **kwargs: EchoBackend(kwargs["model"]))

    result = runner.invoke(
        app, ["arena-run", str(src), "--model", "a", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "arena_reports" / "chat_suite.json").exists()


def test_cli_arena_run_with_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "suite.jsonl"
    _write(src, [{"prompt": "Explain recursion."}])

    def fake_build(**kwargs):
        return JudgeBackend() if kwargs["model"] == "judge" else EchoBackend(kwargs["model"])

    monkeypatch.setattr(cli, "_build_backend", fake_build)
    result = runner.invoke(
        app,
        ["arena-run", str(src), "--model", "a", "--model", "b", "--judge-model", "judge"],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["judge_model"] == "judge"
    assert report["judgments"][0]["winner"] == "a"
