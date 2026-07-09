"""Reproducibility manifest for a training run."""

from pathlib import Path

from corpus_studio import __version__
from corpus_studio.training.provenance import build_run_provenance
from corpus_studio.training.run_registry import TrainingRunRecord


def _project(tmp_path: Path, rows: str = '{"instruction":"a","output":"b"}\n') -> Path:
    (tmp_path / "examples.jsonl").write_text(rows, encoding="utf-8")
    return tmp_path


def _config(tmp_path: Path, text: str = "base_model: x\n") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _make(tmp_path: Path, name: str, rows: str) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "examples.jsonl").write_text(rows, encoding="utf-8")
    return directory


def test_manifest_captures_dataset_config_and_environment(tmp_path: Path):
    project = _project(tmp_path, '{"instruction":"a","output":"b"}\n{"instruction":"c","output":"d"}\n')
    config = _config(tmp_path)

    manifest = build_run_provenance(project, config)

    assert manifest.dataset_fingerprint  # a real fingerprint
    assert manifest.dataset_row_count == 2
    assert manifest.config_sha256 and len(manifest.config_sha256) == 64
    assert manifest.engine_version == __version__
    assert manifest.platform and manifest.python_version


def test_fingerprint_is_deterministic_for_the_same_data(tmp_path: Path):
    rows = '{"instruction":"a","output":"b"}\n'
    config = _config(tmp_path)
    a = build_run_provenance(_make(tmp_path, "a", rows), config)
    b = build_run_provenance(_make(tmp_path, "b", rows), config)
    assert a.dataset_fingerprint == b.dataset_fingerprint


def test_missing_dataset_or_config_leaves_fields_null_not_raising(tmp_path: Path):
    manifest = build_run_provenance(tmp_path, tmp_path / "nope.yaml")  # no examples.jsonl, no config

    assert manifest.dataset_fingerprint is None
    assert manifest.dataset_row_count == 0
    assert manifest.config_sha256 is None
    assert manifest.engine_version == __version__  # environment is still captured


def test_record_roundtrips_with_and_without_provenance(tmp_path: Path):
    manifest = build_run_provenance(_project(tmp_path), _config(tmp_path))
    record = TrainingRunRecord(run_id="r1", created_at="t", updated_at="t", provenance=manifest)

    restored = TrainingRunRecord.model_validate_json(record.model_dump_json())
    assert restored.provenance is not None
    assert restored.provenance.config_sha256 == manifest.config_sha256

    # Pre-manifest records (no provenance field) still load, as None.
    legacy = TrainingRunRecord.model_validate({"run_id": "r", "created_at": "t", "updated_at": "t"})
    assert legacy.provenance is None
