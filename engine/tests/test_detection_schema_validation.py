"""A non-text (object-detection) schema, validated end to end through the CLI.

The driver is a weapon/face detection corpus: one row per image carrying normalized YOLO boxes plus
a mandatory provenance record. Nothing here needs a new ``FieldType`` - the compositional
``list`` + ``item_type: object`` + ``item_fields`` path already models a box - so these tests pin
that the SCHEMA-DRIVEN claim holds for a shape that is not a text corpus:

* the structural rules (bounds, enums, required, types) reject the defects they should, nested
  inside a list of objects;
* every failure NAMES the field path the validator built, including the list index, in both the
  plain and the --json output. On a 500k-row corpus "Value must be <= 1.0." alone cannot be triaged;
* ``gate-run`` resolves a PROJECT-LOCAL schema, so the gate pipeline is usable with a custom schema
  rather than failing as an unknown id;
* the opt-in path-existence check fires only when the schema declares it AND a root is supplied, and
  it treats the path value as untrusted input.

Two defect classes this shape also needs are NOT covered here, because both need a declarative
cross-field rule surface that does not exist yet: a class_id / class_name pair that disagrees
(#953) and a box whose centre plus size escapes the unit frame while every individual value stays
inside [0, 1] (#954). Rows carrying either defect validate clean today; both issues carry the
reproduction.
"""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from corpus_studio.cli import app

runner = CliRunner()

CLASS_NAMES = [
    "person", "face", "hand", "handgun", "rifle", "shotgun", "knife", "machete",
    "axe", "blunt_object", "phone", "bottle", "stapler", "tool", "umbrella",
    "remote_control", "toy_or_replica_weapon", "other_hard_negative",
]

DETECTION_SCHEMA: dict[str, Any] = {
    "id": "weapon_detection",
    "name": "Weapon and Face Detection Dataset",
    "version": "0.1.0",
    "description": "One image per row with its normalized YOLO boxes and a provenance record.",
    "fields": [
        {"name": "image", "type": "image_path", "required": True},
        {
            "name": "annotations",
            "type": "list",
            "item_type": "object",
            "required": False,
            "description": "Normalized YOLO boxes. An empty list is a legitimate hard negative.",
            "item_fields": [
                {"name": "class_id", "type": "integer", "required": True,
                 "minimum": 0, "maximum": 17},
                {"name": "class_name", "type": "string", "required": True, "enum": CLASS_NAMES},
                {"name": "source_label", "type": "string", "required": False},
                {"name": "x_center", "type": "float", "required": True,
                 "minimum": 0.0, "maximum": 1.0},
                {"name": "y_center", "type": "float", "required": True,
                 "minimum": 0.0, "maximum": 1.0},
                {"name": "width", "type": "float", "required": True,
                 "minimum": 0.0, "maximum": 1.0},
                {"name": "height", "type": "float", "required": True,
                 "minimum": 0.0, "maximum": 1.0},
            ],
        },
        {"name": "source_dataset", "type": "string", "required": True},
        {"name": "source_version", "type": "string", "required": True},
        {"name": "source_image_id", "type": "string", "required": True},
        {"name": "source_url", "type": "string", "required": True},
        {"name": "retrieved_at", "type": "string", "required": True},
        {"name": "declared_dataset_license", "type": "string", "required": True},
        {"name": "commercial_use_status", "type": "string", "required": True,
         "enum": ["permitted", "prohibited", "legal_review_required", "unknown"]},
        {"name": "redistribution_status", "type": "string", "required": True,
         "enum": ["permitted", "prohibited", "unknown"]},
        {"name": "lane", "type": "string", "required": True,
         "enum": ["commercial_candidate", "research_only", "quarantine"]},
        {"name": "provenance_confidence", "type": "string", "required": True,
         "enum": ["A", "B", "C", "D"]},
        {"name": "sha256", "type": "string", "required": True},
        {"name": "annotation_status", "type": "string", "required": True,
         "enum": ["imported", "reviewed", "corrected", "rejected"]},
        {"name": "human_reviewed", "type": "boolean", "required": True},
        {"name": "notes", "type": "text", "required": False},
    ],
}

_BOXES = [
    {"class_id": 0, "class_name": "person", "source_label": "Person",
     "x_center": 0.5123, "y_center": 0.4817, "width": 0.224, "height": 0.661},
    {"class_id": 3, "class_name": "handgun", "source_label": "Handgun",
     "x_center": 0.6042, "y_center": 0.5533, "width": 0.0585, "height": 0.041},
]


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": "data/commercial_candidate/openimages_v7/images/train/0a1b2c3d.jpg",
        "annotations": json.loads(json.dumps(_BOXES)),
        "source_dataset": "open_images_v7",
        "source_version": "v7",
        "source_image_id": "0a1b2c3d4e5f6071",
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
    row.update(overrides)
    return row


def _box(index: int, **overrides: Any) -> list[dict[str, Any]]:
    boxes = json.loads(json.dumps(_BOXES))
    boxes[index].update(overrides)
    return boxes


def _project(tmp_path: Path, schema: dict[str, Any] | None = None) -> Path:
    """A project whose schemas/ holds the detection schema, as a project-local schema."""
    root = tmp_path / "projects"
    result = runner.invoke(
        app, ["new-project", "det", "Detection", "image_caption", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    project_dir = root / "det"
    schemas = project_dir / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    payload = schema or DETECTION_SCHEMA
    (schemas / f"{payload['id']}.schema.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return project_dir


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _validate(project_dir: Path, data: Path, *extra: str, schema_id: str = "weapon_detection"):
    return runner.invoke(
        app, ["schema-validate", str(project_dir), schema_id, "--data", str(data), *extra]
    )


# --- the schema resolves and is well-formed -----------------------------------------

def test_project_local_detection_schema_resolves(tmp_path: Path):
    project_dir = _project(tmp_path)
    result = runner.invoke(app, ["schema-validate", str(project_dir), "weapon_detection"])
    assert result.exit_code == 0, result.output
    assert "(project)" in result.stdout
    assert f"{len(DETECTION_SCHEMA['fields'])} field(s)" in result.stdout


# --- positive cases -----------------------------------------------------------------

def test_valid_multi_box_row_and_hard_negative_pass(tmp_path: Path):
    project_dir = _project(tmp_path)
    rows = [_row(), _row(source_image_id="hard-negative", annotations=[])]
    result = _validate(project_dir, _jsonl(tmp_path / "rows.jsonl", rows))
    assert result.exit_code == 0, result.output
    assert "all rows valid" in result.stdout


# --- negative cases: each defect, its field path, and a non-zero exit ----------------

def test_each_structural_defect_is_rejected_with_its_field_path(tmp_path: Path):
    project_dir = _project(tmp_path)
    cases = [
        ("coordinate above 1", _row(annotations=_box(1, x_center=1.4)),
         "annotations[2].x_center", "Value must be <= 1.0."),
        ("negative width", _row(annotations=_box(1, width=-0.1)),
         "annotations[2].width", "Value must be >= 0.0."),
        ("class_id out of range", _row(annotations=_box(1, class_id=99)),
         "annotations[2].class_id", "Value must be <= 17.0."),
        ("class_name outside the enum", _row(annotations=_box(1, class_name="bazooka")),
         "annotations[2].class_name", "Value must be one of:"),
        ("bad provenance enum", _row(commercial_use_status="probably fine"),
         "commercial_use_status", "Value must be one of:"),
        ("non-boolean human_reviewed", _row(human_reviewed="yes"),
         "human_reviewed", "Expected boolean."),
        ("missing required provenance field", {k: v for k, v in _row().items() if k != "sha256"},
         "sha256", "Missing required field: sha256"),
    ]
    for label, row, field, message in cases:
        result = _validate(project_dir, _jsonl(tmp_path / "one.jsonl", [row]), "--json")
        assert result.exit_code == 1, f"{label}: expected a non-zero exit\n{result.output}"
        errors = json.loads(result.stdout)["row_errors"]
        assert len(errors) == 1, f"{label}: {errors}"
        assert errors[0]["field"] == field, f"{label}: {errors[0]}"
        assert message in errors[0]["message"], f"{label}: {errors[0]}"


def test_defect_free_rows_still_pass_alongside_defective_ones(tmp_path: Path):
    # The probe set: rows 1-2 valid, 3-9 defective. The exit code is driven by the failures, and
    # every failure is attributed to its own row.
    project_dir = _project(tmp_path)
    rows = [
        _row(source_image_id="valid"),
        _row(source_image_id="hard-negative", annotations=[]),
        _row(annotations=_box(1, x_center=1.4)),
        _row(annotations=_box(1, width=-0.1)),
        _row(annotations=_box(1, class_id=99)),
        _row(annotations=_box(1, class_name="bazooka")),
        _row(commercial_use_status="probably fine"),
        _row(human_reviewed="yes"),
        {k: v for k, v in _row().items() if k != "sha256"},
    ]
    result = _validate(project_dir, _jsonl(tmp_path / "probe.jsonl", rows), "--json")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["rows_checked"] == 9
    assert payload["row_error_count"] == 7
    assert sorted({error["row_number"] for error in payload["row_errors"]}) == [3, 4, 5, 6, 7, 8, 9]


# --- regression: the field path reaches BOTH outputs --------------------------------

def test_plain_output_names_the_failing_field_path_with_its_list_index(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row(), _row(annotations=_box(1, x_center=1.4))])
    result = _validate(project_dir, data)
    assert result.exit_code == 1
    # Previously: "row 2: Value must be <= 1.0." - which box, and which coordinate?
    assert "row 2: annotations[2].x_center: Value must be <= 1.0." in result.stderr


def test_json_output_carries_the_field_path(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row(annotations=_box(0, y_center=2.0))])
    result = _validate(project_dir, data, "--json")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["row_errors"] == [
        {"row_number": 1, "field": "annotations[1].y_center", "message": "Value must be <= 1.0."}
    ]


# --- regression: gate-run resolves a project-local schema ---------------------------

def test_gate_run_resolves_a_project_local_schema(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row(), _row(source_image_id="b", annotations=[])])
    result = runner.invoke(
        app,
        ["gate-run", str(data), "weapon_detection", "--scope", "dataset",
         "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "Unknown schema" not in result.output
    report = json.loads(result.stdout)
    schema_gate = [r for r in report["results"] if r["gate_id"] == "schema"][0]
    assert schema_gate["status"] == "pass", schema_gate


def test_gate_run_project_local_schema_blocks_on_a_real_defect(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row(annotations=_box(1, x_center=1.4))])
    result = runner.invoke(
        app,
        ["gate-run", str(data), "weapon_detection", "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["overall_status"] == "block"
    schema_gate = [r for r in report["results"] if r["gate_id"] == "schema"][0]
    assert schema_gate["status"] == "block"


def test_gate_run_still_rejects_a_genuinely_unknown_schema(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row()])
    result = runner.invoke(
        app, ["gate-run", str(data), "no_such_schema", "--project-dir", str(project_dir)]
    )
    assert result.exit_code == 1
    assert "Unknown schema" in result.stderr


def test_gate_run_export_scope_also_resolves_a_project_local_schema(tmp_path: Path):
    project_dir = _project(tmp_path)
    data = _jsonl(tmp_path / "rows.jsonl", [_row()])
    result = runner.invoke(
        app,
        ["gate-run", str(data), "weapon_detection", "--scope", "export",
         "--project-dir", str(project_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "Unknown schema" not in result.output
