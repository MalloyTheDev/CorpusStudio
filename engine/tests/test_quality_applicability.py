"""The quality/debt layer must never report a verdict for a shape it could not assess.

The quality signals are TEXT signals. On an object-detection row - normalized boxes plus a
provenance record - they have nothing to read, yet they used to return a confident verdict: a
detection corpus with a missing image, a class_id/class_name mismatch and an out-of-frame box came
back grade D whose single finding was the provenance metadata being CORRECTLY consistent. That is
false assurance twice over, and these tests pin both halves of the fix:

* applicability - every signal declares the field roles it needs, roles come from the schema's
  declared ``FieldType`` (never guessed from values), and a signal with nothing to read reports
  ``not_applicable`` instead of a zero. ``dataset-debt`` then WITHHOLDS the letter grade;
* structural repetition - a field that is constant across the dataset, or path-shaped, is required
  to repeat, so templating detection must not read it. This half needs no schema.

The existing text-corpus behaviour must be unchanged by both: ``examples/wbg/data/wbg_clean_522.jsonl``
still grades D on 62 synthetic-pattern issues, and that verdict is correct for a text corpus.
"""

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.quality.applicability import (
    MIN_ASSESSED_CONTENT_SHARE,
    SCHEMA_TYPE_ROLES,
    SIGNAL_REQUIRED_ROLES,
    measure_applicability,
    schema_field_roles,
)
from corpus_studio.quality.basic_quality import (
    _synthetic_pattern_text,
    build_basic_quality_report,
    structurally_repeating_leaves,
)
from corpus_studio.reporting.debt_report import build_debt_report, render_debt_report_markdown
from corpus_studio.schemas.base import DatasetSchema, FieldType
from corpus_studio.schemas.registry import load_builtin_schema

from tests.test_detection_schema_validation import DETECTION_SCHEMA

runner = CliRunner()

WBG_CORPUS = Path(__file__).resolve().parents[2] / "examples" / "wbg" / "data" / "wbg_clean_522.jsonl"

_TEXT_SIGNALS = {
    "normalized_duplicates",
    "low_information",
    "synthetic_patterns",
    "token_length_outliers",
}


def _detection_schema() -> DatasetSchema:
    return DatasetSchema.model_validate(DETECTION_SCHEMA)


def _detection_rows() -> list[dict[str, Any]]:
    """Three rows that are schema-VALID but genuinely broken, as a real lane would look: distinct
    images and geometry, with the provenance columns constant across the lane (which the corpus
    policy requires). The defects are a missing image, a class_id/class_name mismatch, and a box
    that overflows the frame."""
    base = {
        "source_dataset": "open_images_v7",
        "source_version": "v7",
        "source_url": "https://storage.googleapis.com/openimages/web/index.html",
        "retrieved_at": "2026-10-03T00:00:00Z",
        "declared_dataset_license": "CC BY 4.0 (annotations)",
        "commercial_use_status": "legal_review_required",
        "redistribution_status": "prohibited",
        "lane": "commercial_candidate",
        "provenance_confidence": "B",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "annotation_status": "imported",
        "human_reviewed": False,
        "notes": "Per-image license must be verified individually.",
    }
    defects = [
        ("missing-image", [{"class_id": 0, "class_name": "person", "x_center": 0.11,
                            "y_center": 0.23, "width": 0.09, "height": 0.13}]),
        ("label-mismatch", [{"class_id": 3, "class_name": "rifle", "x_center": 0.31,
                             "y_center": 0.41, "width": 0.12, "height": 0.17}]),
        ("box-overflow", [{"class_id": 6, "class_name": "knife", "x_center": 0.95,
                           "y_center": 0.52, "width": 0.4, "height": 0.21}]),
    ]
    rows = []
    for index, (name, annotations) in enumerate(defects):
        row = dict(base)
        row["image"] = f"data/commercial_candidate/openimages_v7/images/train/img{index:04d}.jpg"
        row["source_image_id"] = name
        row["annotations"] = annotations
        rows.append(row)
    return rows


# --- roles come from the schema, never from the values -------------------------------

def test_every_field_type_has_a_role():
    # A new FieldType must be given a role deliberately, not silently fall back to non-prose.
    containers = {"list", "object"}  # recursed into, never leaves themselves
    assert set(get_args(FieldType)) - containers == set(SCHEMA_TYPE_ROLES)


def test_string_is_not_prose_but_text_is():
    # FieldType separates `string` (a short scalar: id, label, hash, URL, enum member) from the
    # prose types. Every builtin schema uses it that way, and the distinction is what lets a
    # detection schema be told apart from a text corpus.
    assert SCHEMA_TYPE_ROLES["string"] == "categorical"
    for prose in ("text", "markdown", "code", "messages"):
        assert SCHEMA_TYPE_ROLES[prose] == "free_text"


def test_leaf_roles_recurse_into_lists_of_objects():
    roles = schema_field_roles(_detection_schema())
    assert roles["annotations[].class_id"] == "numeric"
    assert roles["annotations[].x_center"] == "numeric"
    assert roles["annotations[].class_name"] == "categorical"
    assert roles["image"] == "path"
    assert roles["notes"] == "free_text"
    assert "annotations" not in roles  # the container itself is not a leaf


# --- a non-text schema: the text signals report not_applicable ----------------------

def test_detection_schema_marks_the_text_signals_not_applicable():
    applicability = measure_applicability(_detection_rows(), _detection_schema())
    assert applicability.role_source == "schema"
    assert applicability.assessed_content_share < MIN_ASSESSED_CONTENT_SHARE
    assert set(applicability.not_applicable_signals) == _TEXT_SIGNALS
    for signal in _TEXT_SIGNALS:
        assert not applicability.applies(signal)


def test_any_shape_signals_still_apply_to_a_detection_schema():
    # An entirely empty row and a byte-identical duplicate are debt in ANY schema, and a pasted
    # credential can land in any string field, so these are never withheld.
    applicability = measure_applicability(_detection_rows(), _detection_schema())
    for signal in ("empty_rows", "exact_duplicates", "secrets", "personal_data"):
        assert applicability.applies(signal)
        assert signal in applicability.applicable_signals


@pytest.mark.parametrize("schema_id", ["instruction", "chat", "preference", "raw_text",
                                       "image_caption", "classification", "code", "evaluation"])
def test_builtin_text_schemas_keep_the_text_signals(schema_id: str):
    # A field-COUNT ratio would wrongly withhold raw_text (one prose field among five) and
    # image_caption (one among four); the measured content share does not.
    schema = load_builtin_schema(schema_id)
    rows = [schema.example] if schema.example else []
    applicability = measure_applicability(rows, schema)
    assert applicability.assessed_content_share is not None
    assert applicability.assessed_content_share >= MIN_ASSESSED_CONTENT_SHARE
    assert not _TEXT_SIGNALS & set(applicability.not_applicable_signals), applicability.reason


def test_a_schema_with_no_categorical_field_cannot_be_category_imbalanced():
    # preference is all prose (prompt/chosen/rejected/reason), so there is no low-cardinality
    # field to be imbalanced. That signal is honestly not_applicable rather than a zero, and it
    # is not grade-bearing, so the letter grade still stands.
    schema = load_builtin_schema("preference")
    applicability = measure_applicability([schema.example] if schema.example else [], schema)
    assert applicability.not_applicable_signals == ["category_imbalance"]
    report = build_debt_report(
        build_basic_quality_report([schema.example] if schema.example else [], schema)
    )
    assert report.grade is not None
    assert report.not_assessed == ["category_imbalance"]


def test_no_schema_reports_unmeasured_rather_than_assuming():
    applicability = measure_applicability(_detection_rows(), None)
    assert applicability.role_source == "unmeasured"
    assert applicability.measured is False
    assert applicability.assessed_content_share is None
    assert applicability.applicable_signals == sorted(SIGNAL_REQUIRED_ROLES)
    assert "--schema" in applicability.reason


def test_not_applicable_signals_are_not_computed():
    # The count must be a *skipped* zero, named as skipped - never a computed zero that reads as a
    # pass. The detection rows repeat their provenance columns in every row.
    report = build_basic_quality_report(_detection_rows(), _detection_schema())
    assert report.synthetic_pattern_count == 0
    assert report.synthetic_pattern_issues == []
    assert report.token_length_outlier_count == 0
    assert report.duplicate_normalized_count == 0
    assert report.applicability is not None
    assert set(report.applicability.not_applicable_signals) == _TEXT_SIGNALS


# --- the debt ledger withholds the grade --------------------------------------------

def test_detection_corpus_gets_no_letter_grade():
    report = build_debt_report(build_basic_quality_report(_detection_rows(), _detection_schema()))
    assert report.grade is None
    assert report.graded is False
    assert report.clean is False  # a withheld grade is never "clean"
    assert "no applicable text signals for this schema" in report.grade_reason
    assert sorted(report.not_assessed) == sorted(_TEXT_SIGNALS)
    assert not any(item.category in _TEXT_SIGNALS for item in report.items)


def test_withheld_grade_renders_as_withheld_not_as_a_letter():
    markdown = render_debt_report_markdown(
        build_debt_report(build_basic_quality_report(_detection_rows(), _detection_schema()))
    )
    assert markdown.startswith("# Dataset Debt - Grade withheld")
    assert "No grade:" in markdown
    assert "Not assessed on this shape:" in markdown
    for letter in ("Grade A", "Grade B", "Grade C", "Grade D", "Grade F"):
        assert letter not in markdown


def test_clean_without_a_measured_shape_is_not_reported_as_a_pass():
    # The same rows with no schema: the text signals now find nothing (the provenance false
    # positive is gone), and "found nothing" must not be published as a clean bill of health.
    report = build_debt_report(build_basic_quality_report(_detection_rows()))
    assert report.items == []
    assert report.grade is None
    assert report.clean is False
    assert "not assessed" in report.grade_reason


# --- the provenance false positive ---------------------------------------------------

def test_constant_and_path_shaped_fields_are_treated_as_structurally_repeating():
    leaves = structurally_repeating_leaves(_detection_rows())
    for constant in ("source_dataset", "source_version", "lane", "declared_dataset_license",
                     "notes", "sha256", "human_reviewed"):
        assert constant in leaves, constant
    assert "image" in leaves       # path-shaped: its directory prefix is shared by construction
    assert "source_url" in leaves  # URL-shaped, and constant across the lane
    assert "source_image_id" not in leaves  # genuinely varies


def test_templating_detection_does_not_fire_on_repeated_provenance():
    # The reported defect was: repeated opening 'open_images_v7 v7 ... https storage', severity
    # high, remediation "reduce templated repetition" - i.e. the provenance columns flagged for
    # being correctly consistent. A stable source_dataset/source_version/source_url across a lane
    # is mandatory, not debt.
    report = build_basic_quality_report(_detection_rows())
    assert report.synthetic_pattern_count == 0
    assert report.synthetic_pattern_warnings == []


def test_a_single_row_excludes_nothing():
    # Constancy only carries information across at least two rows (and one row can never reach a
    # repetition threshold anyway).
    assert structurally_repeating_leaves(_detection_rows()[:1]) == frozenset()


def test_a_field_null_in_some_rows_and_a_list_in_others_keeps_its_list_contents():
    # Regression: such a field occupies BOTH leaf paths ("tags" for the null, "tags[]" for the
    # elements). Excluding it on the scalar path must not discard the list's contents, or the
    # text inside the list would silently stop being read.
    rows: list[dict[str, Any]] = [{"tags": None, "output": "first"}]
    rows += [{"tags": [f"varying tag {n}"], "output": "second"} for n in range(3)]
    leaves = structurally_repeating_leaves(rows)
    assert "tags" in leaves        # the scalar path is constant (always null)
    assert "tags[]" not in leaves  # the element path varies, so it is kept
    kept = _synthetic_pattern_text(rows[1], leaves)
    assert "varying" in kept and "tag" in kept


# --- the existing text-corpus behaviour is unchanged ---------------------------------

@pytest.mark.skipif(not WBG_CORPUS.exists(), reason="the WBG example corpus is not present")
def test_wbg_text_corpus_verdict_is_unchanged():
    # The non-goal, pinned on the real corpus: 522 chat rows, grade D on 62 synthetic-pattern
    # issues plus 12 token-length outliers. That verdict is correct for a text corpus and neither
    # the applicability gate nor the structural-repetition fix may move it.
    rows = [json.loads(line) for line in WBG_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 522
    quality = build_basic_quality_report(rows)
    assert quality.synthetic_pattern_count == 62
    assert quality.token_length_outlier_count == 12
    report = build_debt_report(quality)
    assert report.grade == "D"
    assert {item.category for item in report.items} == {
        "synthetic_patterns", "token_length_outliers"
    }
    # Supplying the chat schema is a SHARPER read of the same corpus, not a different verdict:
    # a chat corpus is text-bearing so nothing is withheld, and the templating signal now reads
    # the conversations rather than the `meta` sidecar that the schema does not declare. The
    # findings move (67 openings, in the user turns) and the grade does not.
    with_schema = build_basic_quality_report(rows, load_builtin_schema("chat"))
    assert with_schema.applicability is not None
    assert with_schema.applicability.not_applicable_signals == []
    assert with_schema.synthetic_pattern_count == 67
    assert all(
        "quests" not in issue.pattern for issue in with_schema.synthetic_pattern_issues
    ), "the undeclared meta sidecar must not drive the templating verdict"
    assert build_debt_report(with_schema).grade == "D"


def test_a_text_corpus_keeps_grading_on_its_own_findings():
    rows = [{"instruction": "A", "output": "1"}, {"instruction": "A", "output": "1"}]
    report = build_debt_report(
        build_basic_quality_report(rows, load_builtin_schema("instruction"))
    )
    assert report.grade == "D"
    assert any(item.category == "exact_duplicates" for item in report.items)
    assert report.not_assessed == []


def test_an_identical_prose_field_in_every_row_is_still_flagged():
    # The sharper half of the fix. An assistant output copy-pasted into every row is the
    # archetype of templated data, NOT structural repetition - so where the schema declares the
    # field as prose it stays in, even though it is constant.
    rows = [
        {"instruction": f"Tell me about topic {n} in detail",
         "output": "In conclusion the answer resolves the task as always expected here"}
        for n in range(6)
    ]
    report = build_basic_quality_report(rows, load_builtin_schema("instruction"))
    assert report.applicability is not None
    assert not _TEXT_SIGNALS & set(report.applicability.not_applicable_signals)
    assert report.synthetic_pattern_count > 0
    assert build_debt_report(report).grade in {"C", "D", "F"}


def test_a_declared_prose_field_is_read_but_metadata_is_not():
    # Role-based pruning: the templating verdict comes from the prose the schema declares, and
    # an undeclared metadata sidecar is not read at all (it repeats by construction).
    rows = [
        {
            "instruction": "Repeat this same opening phrase for every single row here",
            "output": f"a distinct answer {n} with enough tokens to be measured properly",
            "meta": {"module": "quests", "teacher": "some-model:cloud"},
        }
        for n in range(5)
    ]
    report = build_basic_quality_report(rows, load_builtin_schema("instruction"))
    patterns = [issue.pattern for issue in report.synthetic_pattern_issues]
    assert patterns, report
    assert any("repeat this same opening" in pattern for pattern in patterns), patterns
    assert not any("quests" in pattern for pattern in patterns), patterns


# --- the CLI surface -----------------------------------------------------------------

def _detection_project(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    assert runner.invoke(
        app, ["new-project", "det", "Detection", "image_caption", "--root", str(root)]
    ).exit_code == 0
    schemas = root / "det" / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "weapon_detection.schema.json").write_text(
        json.dumps(DETECTION_SCHEMA), encoding="utf-8"
    )
    return root / "det"


def test_cli_dataset_debt_with_a_non_text_schema_withholds_the_grade(tmp_path: Path):
    project_dir = _detection_project(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "".join(json.dumps(row) + "\n" for row in _detection_rows()), encoding="utf-8"
    )
    result = runner.invoke(
        app,
        ["dataset-debt", str(data), "--schema", "weapon_detection",
         "--project-dir", str(project_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["grade"] is None
    assert "no applicable text signals" in payload["grade_reason"]
    assert sorted(payload["not_assessed"]) == sorted(_TEXT_SIGNALS)


def test_cli_quality_with_a_non_text_schema_names_what_it_could_not_assess(tmp_path: Path):
    project_dir = _detection_project(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "".join(json.dumps(row) + "\n" for row in _detection_rows()), encoding="utf-8"
    )
    result = runner.invoke(
        app,
        ["quality", str(data), "--schema", "weapon_detection", "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0, result.output
    applicability = json.loads(result.stdout)["applicability"]
    assert applicability["role_source"] == "schema"
    assert sorted(applicability["not_applicable_signals"]) == sorted(_TEXT_SIGNALS)
    assert "could not assess" in applicability["reason"]


def test_cli_report_commands_reject_an_unknown_schema(tmp_path: Path):
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps({"instruction": "a", "output": "b"}) + "\n", encoding="utf-8")
    for command in ("quality", "dataset-debt"):
        result = runner.invoke(app, [command, str(data), "--schema", "no_such_schema"])
        assert result.exit_code == 1, command
        assert "Unknown schema" in result.stderr, command


def test_gate_run_on_a_non_text_schema_reports_what_quality_could_not_assess(tmp_path: Path):
    # The gate resolves the schema, so its quality gate inherits the applicability verdict instead
    # of passing quality silently on a shape it cannot read.
    project_dir = _detection_project(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text(
        "".join(json.dumps(row) + "\n" for row in _detection_rows()), encoding="utf-8"
    )
    result = runner.invoke(
        app, ["gate-run", str(data), "weapon_detection", "--project-dir", str(project_dir)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    schema_gate = [r for r in report["results"] if r["gate_id"] == "schema"][0]
    assert schema_gate["status"] == "pass"  # the rows ARE schema-valid; the defects are semantic


def test_dataset_card_names_the_signals_it_could_not_assess():
    # The card holds the resolved schema, so its quality warnings inherit the applicability. An
    # empty warning list on a non-text dataset would read as a clean bill of health.
    from corpus_studio.reporting.dataset_card import build_dataset_card

    card = build_dataset_card(
        project_id="det",
        project_name="Detection",
        schema=_detection_schema(),
        rows=_detection_rows(),
    )
    assert card.quality.applicability is not None
    assert set(card.quality.applicability.not_applicable_signals) == _TEXT_SIGNALS
    skipped = [w for w in card.warnings if "do not apply to this schema's shape" in w]
    assert len(skipped) == 1, card.warnings
    for signal in _TEXT_SIGNALS:
        assert signal in skipped[0]


def test_dataset_card_for_a_text_corpus_has_no_not_applicable_warning():
    from corpus_studio.reporting.dataset_card import build_dataset_card

    schema = load_builtin_schema("instruction")
    card = build_dataset_card(
        project_id="t",
        project_name="Text",
        schema=schema,
        rows=[{"instruction": f"do task {n}", "output": f"result {n} is complete"} for n in range(4)],
    )
    assert not [w for w in card.warnings if "do not apply to this schema's shape" in w]
