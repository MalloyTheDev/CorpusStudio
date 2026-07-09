"""Close-the-loop eval handoff: turn a finished run into serve→eval→link→gate steps."""

from corpus_studio.training.eval_handoff import build_eval_handoff
from corpus_studio.training.run_registry import (
    FAILED,
    RUNNING,
    SUCCEEDED,
    TrainingRunRecord,
)


def _run(**overrides) -> TrainingRunRecord:
    fields = dict(
        run_id="20260709-r1",
        created_at="2026-07-09T00:00:00Z",
        updated_at="2026-07-09T00:00:00Z",
        status=SUCCEEDED,
        base_model="mistralai/Mistral-7B",
        output_dir="/proj/out/run1",
    )
    fields.update(overrides)
    return TrainingRunRecord(**fields)


def test_succeeded_run_is_ready_with_the_four_ordered_steps():
    plan = build_eval_handoff(
        _run(),
        project_dir="/proj",
        eval_dataset_path="/proj/heldout.jsonl",
        schema_id="instruction",
        served_model="mistral-tuned",
    )
    assert plan.ready is True
    titles = [step.title for step in plan.steps]
    assert len(plan.steps) == 4
    # Ordered: serve (manual) → eval → link → gate.
    assert "Serve" in titles[0]
    assert plan.steps[0].command == ""  # serving is external, no command
    commands = " ".join(step.command for step in plan.steps)
    assert "eval-run" in commands
    assert "training-run-update" in commands
    assert "training-run-gate" in commands


def test_commands_are_grounded_in_the_run_and_inputs():
    plan = build_eval_handoff(
        _run(),
        project_dir="/proj",
        eval_dataset_path="/proj/heldout.jsonl",
        schema_id="instruction",
        served_model="mistral-tuned",
    )
    eval_cmd = plan.steps[1].command
    assert '"/proj/heldout.jsonl"' in eval_cmd
    assert "instruction" in eval_cmd
    assert "--model mistral-tuned" in eval_cmd
    assert "--backend ollama" in eval_cmd
    # The after-eval report lands at a stable per-run path, reused by link + gate.
    assert plan.after_eval_path == "eval_reports/after-20260709-r1.json"
    assert plan.after_eval_path in eval_cmd
    link_cmd = plan.steps[2].command
    assert "--run-id 20260709-r1" in link_cmd
    assert "--after-eval-model mistral-tuned" in link_cmd
    # The produced weights + base model surface in the guidance.
    assert "/proj/out/run1" in plan.steps[0].detail
    assert "mistralai/Mistral-7B" in plan.steps[2].detail


def test_openai_compatible_backend_includes_the_base_url_flag():
    plan = build_eval_handoff(
        _run(),
        project_dir="/proj",
        eval_dataset_path="/proj/heldout.jsonl",
        schema_id="instruction",
        backend="openai-compatible",
        base_url="http://localhost:8000/v1",
        served_model="tuned",
    )
    eval_cmd = plan.steps[1].command
    assert "--backend openai-compatible" in eval_cmd
    assert '--base-url "http://localhost:8000/v1"' in eval_cmd


def test_missing_inputs_render_as_labelled_placeholders():
    plan = build_eval_handoff(_run(), project_dir="/proj")
    eval_cmd = plan.steps[1].command
    assert "<your-served-model>" in eval_cmd
    assert "<held-out-dataset.jsonl>" in eval_cmd
    assert "<schema-id>" in eval_cmd


def test_running_run_is_not_ready_and_has_no_steps():
    plan = build_eval_handoff(_run(status=RUNNING), project_dir="/proj")
    assert plan.ready is False
    assert plan.steps == []
    assert "running" in plan.note
    assert SUCCEEDED in plan.note


def test_failed_run_is_not_ready():
    plan = build_eval_handoff(_run(status=FAILED), project_dir="/proj")
    assert plan.ready is False
    assert plan.steps == []
