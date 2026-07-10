"""Reproducibility manifest for a training run."""

from pathlib import Path

from corpus_studio import __version__
from corpus_studio.training.config_templates import (
    build_lora_config_template,
    render_training_config,
)
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


def _write_rendered_config(tmp_path: Path, name: str, seed: int) -> Path:
    template = build_lora_config_template(
        base_model="m", dataset_path="d.jsonl", eval_dataset_path=None,
        dataset_format="instruction", seed=seed,
    )
    path = tmp_path / name
    path.write_text(render_training_config(template), encoding="utf-8")
    return path


def test_config_sha256_pins_the_seed(tmp_path: Path):
    # The manifest's claim: the config emits a seed and the config SHA-256 hashes it WITH the
    # config, so the seed is pinned. Changing only the seed must change config_sha256; the same
    # seed must reproduce it. (Guards the docstring against drifting back to "seed not pinned".)
    project = _project(tmp_path)

    sha_42 = build_run_provenance(project, _write_rendered_config(tmp_path, "c42.yaml", 42)).config_sha256
    sha_42_again = build_run_provenance(project, _write_rendered_config(tmp_path, "c42b.yaml", 42)).config_sha256
    sha_99 = build_run_provenance(project, _write_rendered_config(tmp_path, "c99.yaml", 99)).config_sha256

    assert sha_42 and len(sha_42) == 64
    assert sha_42 == sha_42_again  # same seed → identical config hash (deterministic)
    assert sha_42 != sha_99  # a different seed → a different pinned config hash


def test_record_roundtrips_with_and_without_provenance(tmp_path: Path):
    manifest = build_run_provenance(_project(tmp_path), _config(tmp_path))
    record = TrainingRunRecord(run_id="r1", created_at="t", updated_at="t", provenance=manifest)

    restored = TrainingRunRecord.model_validate_json(record.model_dump_json())
    assert restored.provenance is not None
    assert restored.provenance.config_sha256 == manifest.config_sha256

    # Pre-manifest records (no provenance field) still load, as None.
    legacy = TrainingRunRecord.model_validate({"run_id": "r", "created_at": "t", "updated_at": "t"})
    assert legacy.provenance is None
