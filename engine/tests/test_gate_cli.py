import json
from pathlib import Path

from typer.testing import CliRunner

from corpus_studio.cli import app

runner = CliRunner()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_provider_policy_lists_defaults():
    result = runner.invoke(app, ["provider-policy"])
    assert result.exit_code == 0, result.output
    providers = json.loads(result.output)["providers"]
    assert providers["openai"]["outputs_trainable"] is False
    assert "trainable_output_generator" in providers["openai"]["blocked_roles"]
    assert providers["ollama"]["provider_kind"] == "local"


def test_provider_approve_rejects_frontier_provider(tmp_path: Path):
    result = runner.invoke(
        app,
        ["provider-approve", "--provider", "openai", "--project-dir", str(tmp_path), "--model", "gpt-4o"],
    )
    assert result.exit_code == 2


def test_provider_approve_and_revoke_ollama(tmp_path: Path):
    approve = runner.invoke(
        app,
        ["provider-approve", "--provider", "ollama", "--project-dir", str(tmp_path), "--model", "llama3"],
    )
    assert approve.exit_code == 0, approve.output
    payload = json.loads(approve.output)
    assert payload["approved_key"] == "ollama/model:llama3"
    assert payload["can_generate_trainable"] is True
    assert payload["requires_human_review"] is True

    # Effective policy now reflects approval.
    policy = runner.invoke(
        app,
        ["provider-policy", "--provider", "ollama", "--model", "llama3", "--project-dir", str(tmp_path)],
    )
    assert json.loads(policy.output)["user_approved_generation"] is True

    revoke = runner.invoke(
        app,
        ["provider-approve", "--provider", "ollama", "--project-dir", str(tmp_path), "--model", "llama3", "--revoke"],
    )
    assert json.loads(revoke.output)["revoked"] is True


def test_gate_run_dataset_passes_clean(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    _write(
        src,
        [
            {"instruction": "Explain recursion clearly.", "output": "A function calls itself on subproblems until a base case."},
            {"instruction": "Explain binary search.", "output": "It halves a sorted range each step to locate a value."},
        ],
    )
    result = runner.invoke(app, ["gate-run", str(src), "instruction"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["scope"] == "dataset"
    assert report["overall_status"] == "pass"


def test_gate_run_export_blocks_on_secret_and_writes_report(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    _write(src, [{"instruction": "leak", "output": "use AKIAIOSFODNN7EXAMPLE now"}])
    result = runner.invoke(
        app, ["gate-run", str(src), "instruction", "--scope", "export", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["scope"] == "export"
    assert report["overall_status"] == "block"
    written = list((tmp_path / "gate_reports").glob("export-*.json"))
    assert len(written) == 1


def test_gate_run_distinct_targets_do_not_collide(tmp_path: Path):
    rows = [{"instruction": "Explain recursion.", "output": "A function calls itself on subproblems clearly."}]
    for name in ("a.jsonl", "b.jsonl"):
        src = tmp_path / name
        _write(src, rows)
        runner.invoke(app, ["gate-run", str(src), "instruction", "--project-dir", str(tmp_path)])
    reports = list((tmp_path / "gate_reports").glob("dataset-*.json"))
    assert len(reports) == 2  # a.jsonl and b.jsonl each kept a distinct report
