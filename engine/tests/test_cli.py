import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import corpus_studio.cli as cli
from corpus_studio.cli import app
from corpus_studio.model_backends.base import BackendGenerateResponse, ModelBackendConfig
from corpus_studio.providers.overrides import approve_generation


runner = CliRunner()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_schemas_command_lists_builtin_schemas():
    result = runner.invoke(app, ["schemas"])

    assert result.exit_code == 0
    schemas = json.loads(result.output)
    assert "instruction" in {schema["id"] for schema in schemas}


def test_new_project_command_creates_project_files(tmp_path: Path):
    result = runner.invoke(
        app,
        ["new-project", "demo_project", "Demo Project", "instruction", "--root", str(tmp_path)],
    )

    assert result.exit_code == 0
    project_path = tmp_path / "demo_project" / "project.json"
    assert project_path.exists()
    assert (tmp_path / "demo_project" / "examples.jsonl").exists()
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["split_settings"] == {
        "train_ratio": 0.9,
        "validation_ratio": 0.05,
        "seed": 42,
    }


def test_export_command_rejects_invalid_rows(tmp_path: Path):
    input_path = tmp_path / "invalid.jsonl"
    output_path = tmp_path / "export.jsonl"
    write_rows(input_path, [{"instruction": "Explain variables."}])

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 1
    assert "Missing required field: output" in result.output
    assert not output_path.exists()


def test_export_command_rejects_wrong_field_type(tmp_path: Path):
    input_path = tmp_path / "invalid_type.jsonl"
    output_path = tmp_path / "export.jsonl"
    write_rows(
        input_path,
        [{"instruction": "Explain variables.", "output": "A value.", "tags": "bad"}],
    )

    result = runner.invoke(app, ["export", str(input_path), str(output_path), "instruction"])

    assert result.exit_code == 1
    assert "Expected list." in result.output
    assert not output_path.exists()


def test_split_command_writes_train_validation_and_test_files(tmp_path: Path):
    input_path = tmp_path / "instruction.jsonl"
    output_dir = tmp_path / "splits"
    write_rows(
        input_path,
        [
            {"instruction": f"Explain item {index}.", "output": f"Item {index} explanation."}
            for index in range(20)
        ],
    )

    result = runner.invoke(app, ["split", str(input_path), str(output_dir), "instruction"])

    assert result.exit_code == 0
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_split_command_accepts_custom_ratios_and_seed(tmp_path: Path):
    input_path = tmp_path / "instruction.jsonl"
    output_dir = tmp_path / "custom_splits"
    write_rows(
        input_path,
        [
            {"instruction": f"Explain item {index}.", "output": f"Item {index} explanation."}
            for index in range(20)
        ],
    )

    result = runner.invoke(
        app,
        [
            "split",
            str(input_path),
            str(output_dir),
            "instruction",
            "--train-ratio",
            "0.8",
            "--validation-ratio",
            "0.1",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["train"] == 16
    assert report["validation"] == 2
    assert report["test"] == 2
    assert report["train_ratio"] == 0.8
    assert report["validation_ratio"] == 0.1
    assert report["test_ratio"] == pytest.approx(0.1)
    assert report["seed"] == 123
    assert report["warnings"] == []


def test_split_command_warns_for_tiny_validation_and_test_splits(tmp_path: Path):
    input_path = tmp_path / "tiny_instruction.jsonl"
    output_dir = tmp_path / "tiny_splits"
    write_rows(
        input_path,
        [
            {"instruction": f"Explain item {index}.", "output": f"Item {index} explanation."}
            for index in range(3)
        ],
    )

    result = runner.invoke(app, ["split", str(input_path), str(output_dir), "instruction"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    # 3 rows at 0.9/0.05: train gets 2, and validation is guaranteed a row rather than being
    # silently floored to empty (item 14) — which leaves the test split empty here instead.
    assert report["validation"] == 1
    assert report["test"] == 0
    assert report["warnings"] == [
        "Validation split has only 1 row. Add examples or adjust split ratios before relying on scores.",
        "Test split has no rows. Add examples or adjust split ratios before using it.",
    ]
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def test_split_command_rejects_ratios_without_test_split(tmp_path: Path):
    input_path = tmp_path / "instruction.jsonl"
    output_dir = tmp_path / "bad_splits"
    write_rows(
        input_path,
        [{"instruction": "Explain variables.", "output": "A variable stores a value."}],
    )

    result = runner.invoke(
        app,
        [
            "split",
            str(input_path),
            str(output_dir),
            "instruction",
            "--train-ratio",
            "0.95",
            "--validation-ratio",
            "0.05",
        ],
    )

    assert result.exit_code == 1
    assert not output_dir.exists()


def test_quality_command_reports_duplicates(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    duplicate_row = {"instruction": "Explain variables.", "output": "A variable stores a value."}
    write_rows(input_path, [duplicate_row, duplicate_row])

    result = runner.invoke(app, ["quality", str(input_path)])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["example_count"] == 2
    assert report["duplicate_exact_count"] == 1
    assert report["duplicate_normalized_count"] == 1
    assert report["low_information_count"] == 0


def test_quality_command_reports_normalized_duplicates_and_low_information_rows(tmp_path: Path):
    input_path = tmp_path / "quality_rows.jsonl"
    write_rows(
        input_path,
        [
            {"instruction": "Explain variables.", "output": "A variable stores a value."},
            {"instruction": " explain VARIABLES ", "output": "A variable stores a value."},
            {"instruction": "Hi", "output": "Ok"},
        ],
    )

    result = runner.invoke(app, ["quality", str(input_path)])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["example_count"] == 3
    assert report["duplicate_exact_count"] == 0
    assert report["duplicate_normalized_count"] == 1
    assert report["low_information_count"] == 1
    assert report["low_information_token_threshold"] == 5


def test_quality_command_reports_dataset_wide_synthetic_patterns(tmp_path: Path):
    input_path = tmp_path / "synthetic_rows.jsonl"
    write_rows(
        input_path,
        [
            {
                "instruction": f"Certainly, here is a helpful example about loops {index}.",
                "output": "This answer repeats a synthetic opening and closing pattern.",
            }
            for index in range(3)
        ],
    )

    result = runner.invoke(app, ["quality", str(input_path)])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["example_count"] == 3
    assert report["synthetic_pattern_count"] >= 2
    assert any(
        "generic synthetic phrase" in warning
        for warning in report["synthetic_pattern_warnings"]
    )
    assert any(
        "repeated opening" in warning
        for warning in report["synthetic_pattern_warnings"]
    )
    assert any(
        issue["kind"] == "generic_phrase"
        and issue["severity"] == "medium"
        and "Rewrite these rows" in issue["suggestion"]
        for issue in report["synthetic_pattern_issues"]
    )
    assert any(
        issue["kind"] == "repeated_opening"
        and issue["severity"] == "high"
        and issue["row_numbers"] == [1, 2, 3]
        for issue in report["synthetic_pattern_issues"]
    )


def test_import_preview_command_reports_failed_rows(tmp_path: Path):
    input_path = tmp_path / "mixed.jsonl"
    write_rows(
        input_path,
        [
            {"instruction": "Explain variables.", "output": "A variable stores a value."},
            {"instruction": "Missing output."},
        ],
    )

    result = runner.invoke(app, ["import-preview", str(input_path), "instruction"])

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["valid"] is False
    assert report["accepted_rows"] == 1
    assert report["rejected_rows"] == 1
    assert report["failed_rows"][0]["row_number"] == 2


def test_eval_run_command_uses_backend_and_writes_report_without_real_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "instruction.jsonl"
    output_path = tmp_path / "evaluation_report.json"
    write_rows(
        input_path,
        [
            {
                "instruction": "Explain variables.",
                "input": "",
                "output": "A variable stores a value.",
                "tags": ["basics"],
            }
        ],
    )

    class FakeBackend:
        def generate(self, request):
            return BackendGenerateResponse(
                text="A variable stores a value.",
                model_name="fake-model",
            )

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    result = runner.invoke(
        app,
        [
            "eval-run",
            str(input_path),
            "instruction",
            "--model",
            "fake-model",
            "--backend",
            "ollama",
            "--base-url",
            "http://localhost:11434",
            "--output-path",
            str(output_path),
            "--limit",
            "1",
            "--score-threshold",
            "55",
            "--timeout-seconds",
            "33",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    written_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["examples_tested"] == 1
    assert report["failed_examples"] == 0
    assert report["tag_summary"] == [
        {
            "tag": "basics",
            "examples": 1,
            "failed_examples": 0,
            "average_score": 100.0,
        }
    ]
    assert report["failure_reason_summary"] == []
    assert report["score_band_summary"] == [
        {
            "band": "85-100",
            "examples": 1,
            "failed_examples": 0,
            "average_score": 100.0,
        }
    ]
    assert report["run_settings"] == {
        "dataset_path": str(input_path),
        "schema_id": "instruction",
        "backend": "ollama",
        "base_url": "http://localhost:11434",
        "model": "fake-model",
        "limit": 1,
        "score_threshold": 55.0,
        "timeout_seconds": 33,
    }
    assert written_report == report


def test_eval_run_with_judge_model_uses_llm_judge_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "instruction.jsonl"
    write_rows(
        input_path,
        [{"instruction": "Explain variables.", "input": "", "output": "A variable stores a value."}],
    )

    class AnswerBackend:
        def generate(self, request):
            return BackendGenerateResponse(text="A named box holding a value.", model_name="fake-model")

    class JudgeBackend:
        def generate(self, request):
            return BackendGenerateResponse(
                text='{"score": 88, "rationale": "equivalent meaning"}', model_name="judge-model"
            )

    # The judge and the model-under-test resolve to different fake backends by model name.
    monkeypatch.setattr(
        cli,
        "_build_backend",
        lambda **kw: JudgeBackend() if kw.get("model") == "judge-model" else AnswerBackend(),
    )

    result = runner.invoke(
        app,
        [
            "eval-run", str(input_path), "instruction",
            "--model", "fake-model",
            "--judge-model", "judge-model",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["metric"] == "llm_judge"
    assert report["average_score"] == 88.0
    assert report["results"][0]["rationale"] == "equivalent meaning"


def test_ai_assist_command_uses_backend_and_returns_review_required_suggestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    input_path = tmp_path / "draft.jsonl"
    output_path = tmp_path / "ai_assist.json"
    suggested_jsonl = json.dumps(
        {
            "instruction": "Explain variables clearly.",
            "input": "",
            "output": "A variable stores a value that code can read or update.",
            "tags": ["basics"],
        }
    )
    write_rows(
        input_path,
        [
            {
                "instruction": "Explain variables.",
                "input": "",
                "output": "A variable stores a value.",
                "tags": "bad",
            }
        ],
    )

    class FakeBackend:
        def generate(self, request):
            assert "untrusted user data" in request.prompt
            return BackendGenerateResponse(
                text=json.dumps(
                    {
                        "summary": "Suggested clearer wording and a valid tag list.",
                        "suggested_jsonl": suggested_jsonl,
                        "tags": ["basics"],
                        "warnings": [],
                    }
                ),
                model_name="fake-model",
            )

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    # rewrite-output is a trainable-generating action; v0.6 requires the local
    # model to be explicitly generation-approved in the project overrides.
    approve_generation(input_path.parent, "ollama", model_id="fake-model")

    result = runner.invoke(
        app,
        [
            "ai-assist",
            str(input_path),
            "instruction",
            "--action",
            "rewrite-output",
            "--model",
            "fake-model",
            "--backend",
            "ollama",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["review_required"] is True
    assert payload["review_state"] == "review_required"
    assert "Expected list." in payload["warnings"][0]
    assert payload["validation_errors"] == []
    assert json.loads(payload["suggested_jsonl"])["tags"] == ["basics"]
    assert written_payload == payload


def test_backend_health_command_reports_reachable_backend_without_real_network(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeBackend:
        config = ModelBackendConfig(
            provider_name="ollama",
            base_url="http://localhost:11434",
            model_name="fake-model",
        )

        def list_models(self):
            return ["fake-model", "other-model"]

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    result = runner.invoke(
        app,
        [
            "backend-health",
            "--model",
            "fake-model",
            "--backend",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["reachable"] is True
    assert payload["model_available"] is True
    assert payload["available_models"] == ["fake-model", "other-model"]


def test_backend_health_command_returns_clean_failure_without_real_network(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeBackend:
        config = ModelBackendConfig(
            provider_name="ollama",
            base_url="http://localhost:11434",
            model_name="missing-model",
        )

        def list_models(self):
            raise TimeoutError("timed out")

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    result = runner.invoke(
        app,
        [
            "backend-health",
            "--model",
            "missing-model",
            "--backend",
            "ollama",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["reachable"] is False
    assert payload["error"] == "timed out"


def test_model_list_command_reports_available_models_without_real_network(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeBackend:
        config = ModelBackendConfig(
            provider_name="ollama",
            base_url="http://localhost:11434",
            model_name="",
        )

        def list_models(self):
            return ["llama3.1:8b", "qwen2.5-coder:7b"]

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    result = runner.invoke(
        app,
        [
            "model-list",
            "--backend",
            "ollama",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["provider_name"] == "ollama"
    assert payload["reachable"] is True
    assert payload["models"] == ["llama3.1:8b", "qwen2.5-coder:7b"]


def test_model_list_command_returns_clean_failure_without_real_network(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeBackend:
        config = ModelBackendConfig(
            provider_name="ollama",
            base_url="http://localhost:11434",
            model_name="",
        )

        def list_models(self):
            raise TimeoutError("timed out")

    monkeypatch.setattr(cli, "_build_backend", lambda **_: FakeBackend())

    result = runner.invoke(
        app,
        [
            "model-list",
            "--backend",
            "ollama",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["reachable"] is False
    assert payload["models"] == []
    assert payload["error"] == "timed out"


def test_training_config_command_writes_config_without_training_dependencies(tmp_path: Path):
    input_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output_path = tmp_path / "training" / "axolotl.yaml"
    write_rows(
        input_path,
        [
            {
                "instruction": "Explain variables.",
                "input": "",
                "output": "A variable stores a value.",
            }
        ],
    )
    write_rows(
        validation_path,
        [
            {
                "instruction": "Explain functions.",
                "input": "",
                "output": "A function groups reusable logic.",
            }
        ],
    )

    result = runner.invoke(
        app,
        [
            "training-config",
            str(input_path),
            "instruction",
            "--output-path",
            str(output_path),
            "--base-model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--target",
            "axolotl",
            "--format",
            "instruction",
            "--eval-dataset-path",
            str(validation_path),
            "--sequence-len",
            "2048",
            "--lora-r",
            "8",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    written_config = output_path.read_text(encoding="utf-8")
    assert payload["target"] == "axolotl_yaml"
    assert payload["training_launcher_implemented"] is True
    assert payload["config"]["sequence_len"] == 2048
    assert payload["config"]["lora_r"] == 8
    # A reproducible-by-default seed is emitted into the config (pinned by the run's
    # provenance manifest via the config hash).
    assert payload["config"]["seed"] == 42
    assert "seed: 42" in written_config
    assert 'base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"' in written_config
    assert "exports the config only" in payload["warnings"][0]


def test_training_config_command_rejects_invalid_lora_values(tmp_path: Path):
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "training" / "bad.yaml"
    write_rows(
        input_path,
        [{"instruction": "Explain variables.", "output": "A variable stores a value."}],
    )

    result = runner.invoke(
        app,
        [
            "training-config",
            str(input_path),
            "instruction",
            "--output-path",
            str(output_path),
            "--base-model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--lora-r",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert not output_path.exists()
