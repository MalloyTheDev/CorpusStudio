"""Hugging Face Hub import (read-only, public) — all offline via a fake opener."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

from typer.testing import CliRunner

import corpus_studio.importers.hf_hub as hf
from corpus_studio.cli import app
from corpus_studio.importers.hf_hub import (
    fetch_rows,
    inspect_dataset,
    map_rows,
    suggest_mapping,
)
from corpus_studio.schemas.registry import load_builtin_schema

runner = CliRunner()

IS_VALID = {"preview": True, "viewer": True}
META = {"gated": False, "cardData": {"license": "apache-2.0"}}
SPLITS = {"splits": [{"dataset": "acme/set", "config": "default", "split": "train"}]}
ROWS = {
    "num_rows_total": 2,
    "features": [{"name": n} for n in ("instruction", "input", "output", "text")],
    "rows": [
        {"row": {"instruction": "Q1", "input": "", "output": "A1", "text": "t1"}},
        {"row": {"instruction": "Q2", "input": "i2", "output": "A2", "text": "t2"}},
    ],
}


class _Resp:
    def __init__(self, body: bytes):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._b


def _make_opener(meta=META, rows_by_offset=None):
    rows_by_offset = rows_by_offset or {0: ROWS}

    def opener(request, timeout=None):
        url = request.full_url
        if "/rows?" in url:
            offset = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["offset"][0])
            return _Resp(json.dumps(rows_by_offset[offset]).encode("utf-8"))
        if "/is-valid" in url:
            return _Resp(json.dumps(IS_VALID).encode("utf-8"))
        if "/api/datasets/" in url:
            return _Resp(json.dumps(meta).encode("utf-8"))
        if "/splits" in url:
            return _Resp(json.dumps(SPLITS).encode("utf-8"))
        raise AssertionError(f"unexpected url: {url}")

    return opener


# --- unit: client + mapping --------------------------------------------------


def test_inspect_dataset_reports_splits_license_and_columns():
    info = inspect_dataset("acme/set", opener=_make_opener())
    assert info.viewable is True
    assert info.gated is False
    assert info.license == "apache-2.0"
    assert [(c.config, c.split) for c in info.configs_splits] == [("default", "train")]
    assert info.sample_columns == ["instruction", "input", "output", "text"]
    assert "NOT assumed to be licensed for training" in info.license_note


def test_inspect_distinguishes_failed_license_lookup_from_absent():
    def opener(request, timeout=None):
        url = request.full_url
        if "/api/datasets/" in url:
            # A cold / rate-limited hub API can return a non-JSON body; that
            # ValueError is non-transient (not retried), so this stays fast.
            raise ValueError("metadata body was not JSON")
        return _make_opener()(request, timeout)

    info = inspect_dataset("acme/set", opener=opener)
    assert info.license is None
    # Honest: a failed lookup is NOT the same as "the dataset declares no license".
    assert "could not be verified" in info.license_note
    assert "not declared" not in info.license_note
    assert "NOT assumed to be licensed for training" in info.license_note  # caveat still shown


def test_fetch_rows_extracts_row_dicts_and_columns():
    page = fetch_rows("acme/set", "default", "train", limit=2, opener=_make_opener())
    assert page.num_rows_total == 2
    assert page.columns == ["instruction", "input", "output", "text"]
    assert page.rows[0]["instruction"] == "Q1"
    assert page.rows[1]["input"] == "i2"


def test_fetch_rows_paginates_across_the_page_cap(monkeypatch):
    # Force a 1-row page cap so a 3-row fetch must paginate.
    monkeypatch.setattr(hf, "_MAX_PAGE", 1)
    pages = {
        0: {"num_rows_total": 3, "features": [{"name": "text"}], "rows": [{"row": {"text": "a"}}]},
        1: {"num_rows_total": 3, "features": [{"name": "text"}], "rows": [{"row": {"text": "b"}}]},
        2: {"num_rows_total": 3, "features": [{"name": "text"}], "rows": [{"row": {"text": "c"}}]},
    }
    page = fetch_rows("acme/set", "default", "train", limit=3, opener=_make_opener(rows_by_offset=pages))
    assert [r["text"] for r in page.rows] == ["a", "b", "c"]


def test_fetch_rows_paginates_when_num_rows_total_absent(monkeypatch):
    # The datasets-server omits num_rows_total for some datasets. It defaults to 0, which
    # previously made pagination stop after the first page and silently truncate. End-of-data
    # must come from the pages themselves (empty/short page), not from that field.
    monkeypatch.setattr(hf, "_MAX_PAGE", 1)
    pages = {
        0: {"features": [{"name": "text"}], "rows": [{"row": {"text": "a"}}]},
        1: {"features": [{"name": "text"}], "rows": [{"row": {"text": "b"}}]},
        2: {"features": [{"name": "text"}], "rows": []},  # end of split
    }
    page = fetch_rows("acme/set", "default", "train", limit=10, opener=_make_opener(rows_by_offset=pages))
    assert [r["text"] for r in page.rows] == ["a", "b"]  # both pages, not just the first


def test_suggest_mapping_matches_schema_fields_by_name():
    schema = load_builtin_schema("instruction")
    mapping = suggest_mapping(["instruction", "input", "output", "text"], schema)
    # tags has no matching column; text is not a schema field.
    assert mapping == {"instruction": "instruction", "input": "input", "output": "output"}


def test_map_rows_projects_only_mapped_fields():
    rows = [{"instruction": "Q", "input": "", "output": "A", "text": "t"}]
    mapped = map_rows(rows, {"instruction": "instruction", "output": "output"})
    assert mapped == [{"instruction": "Q", "output": "A"}]  # input/text dropped


def test_map_rows_flattens_nested_values_to_json_strings():
    # HF columns can hold nested structs/lists (e.g. an image dict or a chat array). A staging
    # row must stay a flat, inspectable object, so nested values are rendered as JSON strings.
    rows = [{"instruction": "Q", "meta": {"a": 1}, "turns": ["x", "y"]}]
    mapped = map_rows(rows, {"instruction": "instruction", "input": "meta", "output": "turns"})
    assert mapped[0]["instruction"] == "Q"          # scalar preserved
    assert mapped[0]["input"] == '{"a": 1}'          # nested dict -> JSON string
    assert mapped[0]["output"] == '["x", "y"]'       # nested list -> JSON string


# --- CLI ----------------------------------------------------------------------


def test_hf_import_writes_staging_and_reports_license(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hf, "urlopen", _make_opener())
    out = tmp_path / "hf_staging.jsonl"
    result = runner.invoke(
        app, ["hf-import", "acme/set", "--out", str(out), "--schema", "instruction"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["fetched_rows"] == 2
    assert payload["license"] == "apache-2.0"
    assert payload["mapping"]["instruction"] == "instruction"
    assert "text" in payload["unused_columns"]
    assert "tags" in payload["unmapped_schema_fields"]

    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["instruction"] == "Q1"
    assert "text" not in lines[0]  # unmapped column not written


def test_hf_import_map_override_wins_over_autodetect(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hf, "urlopen", _make_opener())
    out = tmp_path / "s.jsonl"
    result = runner.invoke(
        app,
        ["hf-import", "acme/set", "--out", str(out), "--schema", "instruction",
         "--map", "output=text"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["mapping"]["output"] == "text"


def test_hf_import_refuses_to_write_examples_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hf, "urlopen", _make_opener())
    result = runner.invoke(
        app,
        ["hf-import", "acme/set", "--out", str(tmp_path / "examples.jsonl"),
         "--schema", "instruction"],
    )
    assert result.exit_code == 2
    assert "examples.jsonl" in result.output


def test_hf_import_refuses_case_variant_examples_jsonl(tmp_path: Path):
    # The guard is case-insensitive so `--out Examples.jsonl` can't clobber the dataset on
    # a case-insensitive filesystem (Windows/macOS). It fires BEFORE any network call, so no
    # opener stub is needed.
    result = runner.invoke(
        app,
        ["hf-import", "acme/set", "--out", str(tmp_path / "Examples.JSONL"),
         "--schema", "instruction"],
    )
    assert result.exit_code == 2
    assert "examples.jsonl" in result.output
    assert not (tmp_path / "Examples.JSONL").exists()


def test_hf_import_refuses_limit_over_cap(tmp_path: Path):
    # A huge --limit would buffer the whole staging page in memory; the import cap refuses it.
    # The check fires before any network call (no opener stub needed).
    from corpus_studio.importers.hf_hub import MAX_IMPORT_ROWS

    result = runner.invoke(
        app,
        ["hf-import", "acme/set", "--out", str(tmp_path / "s.jsonl"),
         "--schema", "instruction", "--limit", str(MAX_IMPORT_ROWS + 1)],
    )
    assert result.exit_code == 1
    assert "cap" in result.output.lower()
    assert not (tmp_path / "s.jsonl").exists()


def test_hf_import_refuses_gated_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(hf, "urlopen", _make_opener(meta={"gated": True, "cardData": {}}))
    result = runner.invoke(
        app,
        ["hf-import", "acme/set", "--out", str(tmp_path / "s.jsonl"), "--schema", "instruction"],
    )
    assert result.exit_code == 2
    assert "gated" in result.output.lower()


def test_hf_inspect_command_emits_json(monkeypatch):
    monkeypatch.setattr(hf, "urlopen", _make_opener())
    result = runner.invoke(app, ["hf-inspect", "acme/set"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["license"] == "apache-2.0"
    assert payload["configs_splits"][0]["split"] == "train"
