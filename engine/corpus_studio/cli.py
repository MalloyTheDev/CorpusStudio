from pathlib import Path
import json
import os
import sqlite3
import sys
from typing import Optional

import typer
from pydantic import ValidationError

from corpus_studio.ai_assist.assistant import run_ai_assist
from corpus_studio.evaluation.evaluator import (
    EvaluationRunConfig,
    extract_evaluation_examples,
    run_evaluation,
)
from corpus_studio.exporters.jsonl_exporter import export_jsonl, write_jsonl
from corpus_studio.exporters.preference_exporter import (
    analyze_preference_pairs,
    drop_degenerate_pairs,
    export_preference,
)
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.importers.jsonl_preview import preview_jsonl_import
from corpus_studio.model_backends.base import BackendHealthReport, BackendModelListReport
from corpus_studio.model_backends.ollama import OllamaBackend, default_ollama_config
from corpus_studio.model_backends.openai_compatible import (
    OpenAICompatibleBackend,
    default_openai_compatible_config,
)
from corpus_studio.quality.basic_quality import build_basic_quality_report
from corpus_studio.reporting.dataset_card import (
    DatasetCardEvaluation,
    DatasetCardSplits,
    build_dataset_card,
    render_dataset_card_markdown,
)
from corpus_studio.evaluation.reports import EvaluationReport
from corpus_studio.schemas.registry import list_builtin_schemas, load_builtin_schema, repository_root
from corpus_studio.splitters.leakage import detect_split_leakage
from corpus_studio.splitters.random_splitter import random_split
from corpus_studio.storage.index import (
    default_index_path,
    index_single_project,
    list_projects_from_root,
    rebuild_index,
)
from corpus_studio.storage.project import DatasetProject, create_project
from corpus_studio.training.compatibility import training_compatibility_warnings
from corpus_studio.training.config_templates import (
    build_lora_config_template,
    normalize_training_config_target,
    render_training_config,
)
from corpus_studio.validators.basic_validator import validate_jsonl_file
from corpus_studio.validators.results import ValidationReport

app = typer.Typer(help="Corpus Studio dataset engine CLI.")


def _repo_relative_path(env_name: str, fallback: Path) -> Path:
    configured = os.environ.get(env_name)
    path = Path(configured) if configured else repository_root() / fallback
    if not path.is_absolute():
        path = repository_root() / path
    return path


def _index_enabled() -> bool:
    """Whether the optional SQLite project index should be kept in sync on writes."""
    return os.environ.get("CORPUS_STUDIO_USE_INDEX", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _exit_if_invalid(report: ValidationReport) -> None:
    if not report.valid:
        typer.echo(report.model_dump_json(indent=2))
        raise typer.Exit(code=1)


def _build_split_warnings(validation_count: int, test_count: int) -> list[str]:
    warnings: list[str] = []
    for split_name, count in (
        ("Validation", validation_count),
        ("Test", test_count),
    ):
        if count == 0:
            warnings.append(
                f"{split_name} split has no rows. Add examples or adjust split ratios before using it."
            )
        elif count == 1:
            warnings.append(
                f"{split_name} split has only 1 row. Add examples or adjust split ratios before relying on scores."
            )

    return warnings


def _build_preference_warnings(issues, drop_degenerate: bool, dropped_count: int) -> list[str]:
    warnings: list[str] = []
    if issues.identical:
        warnings.append(
            f"{issues.identical} pair(s) have identical chosen and rejected text "
            "(zero training signal)."
        )
    empty = issues.empty_chosen + issues.empty_rejected
    if empty:
        warnings.append(f"{empty} pair(s) have an empty chosen or rejected side.")
    if issues.low_contrast:
        warnings.append(
            f"{issues.low_contrast} pair(s) are low-contrast (chosen and rejected "
            "are very similar)."
        )
    if drop_degenerate and dropped_count:
        warnings.append(
            f"Dropped {dropped_count} degenerate pair(s) before export (--drop-degenerate)."
        )
    elif issues.degenerate and not drop_degenerate:
        warnings.append(
            f"{issues.degenerate} degenerate pair(s) were exported; re-run with "
            "--drop-degenerate to exclude them."
        )
    return warnings


@app.command("schemas")
def schemas():
    """List built-in dataset schemas."""
    schema_rows = [schema.model_dump() for schema in list_builtin_schemas()]
    typer.echo(json.dumps(schema_rows, indent=2))


@app.command()
def validate(path: Path, schema: str):
    """Validate a JSONL file against a built-in schema."""
    result = validate_jsonl_file(path, schema)
    typer.echo(result.model_dump_json(indent=2))
    if not result.valid:
        raise typer.Exit(code=1)


@app.command("new-project")
def new_project(project_id: str, name: str, schema: str, root: Optional[Path] = None):
    """Create a local dataset project folder."""
    try:
        load_builtin_schema(schema)
    except (ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    project_root = root or _repo_relative_path("CORPUS_STUDIO_DATA_DIR", Path("data") / "projects")
    project = DatasetProject(id=project_id, name=name, schema_id=schema)

    try:
        path = create_project(project_root, project)
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if _index_enabled():
        try:
            index_single_project(project_root, path)
        except (sqlite3.Error, OSError):
            pass  # the index is an optional cache; never fail project creation

    typer.echo(str(path))


@app.command("project-list")
def project_list(
    root: Optional[Path] = typer.Option(
        None, "--root", help="Projects root. Defaults to the data dir."
    ),
    schema: Optional[str] = typer.Option(None, "--schema", help="Filter by schema id."),
    name_contains: Optional[str] = typer.Option(
        None, "--name-contains", help="Filter by a case-insensitive name substring."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Rebuild the index from disk before listing."
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index-path", help="Override the SQLite index path."
    ),
):
    """List local dataset projects using the optional SQLite index.

    The index is built from project.json files on first use and rebuilt on
    demand; the JSON/JSONL files remain the source of truth.
    """
    projects_root = root or _repo_relative_path(
        "CORPUS_STUDIO_DATA_DIR", Path("data") / "projects"
    )
    entries = list_projects_from_root(
        projects_root,
        db_path=index_path,
        schema_id=schema,
        name_contains=name_contains,
        rebuild=rebuild,
    )
    typer.echo(
        json.dumps(
            {
                "projects_root": str(projects_root),
                "index_path": str(index_path or default_index_path(projects_root)),
                "count": len(entries),
                "projects": [entry.model_dump() for entry in entries],
            },
            indent=2,
        )
    )


@app.command("project-index-rebuild")
def project_index_rebuild(
    root: Optional[Path] = typer.Option(
        None, "--root", help="Projects root. Defaults to the data dir."
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index-path", help="Override the SQLite index path."
    ),
):
    """Rebuild the optional SQLite project index from project.json files on disk."""
    projects_root = root or _repo_relative_path(
        "CORPUS_STUDIO_DATA_DIR", Path("data") / "projects"
    )
    resolved_index = index_path or default_index_path(projects_root)
    count = rebuild_index(projects_root, resolved_index)
    typer.echo(
        json.dumps(
            {
                "projects_root": str(projects_root),
                "index_path": str(resolved_index),
                "indexed": count,
            },
            indent=2,
        )
    )


@app.command()
def quality(path: Path):
    """Build a basic quality report for a JSONL file."""
    rows = list(read_jsonl(path))
    report = build_basic_quality_report(rows)
    typer.echo(report.model_dump_json(indent=2))


@app.command("import-preview")
def import_preview(path: Path, schema: str):
    """Preview a JSONL import and report accepted/rejected rows."""
    try:
        load_builtin_schema(schema)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = preview_jsonl_import(path, schema)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def split(
    input_path: Path,
    output_dir: Path,
    schema: str,
    train_ratio: float = 0.9,
    validation_ratio: float = 0.05,
    seed: int = 42,
):
    """Validate and split JSONL into train, validation, and test files."""
    if train_ratio <= 0 or validation_ratio < 0 or train_ratio + validation_ratio >= 1:
        typer.echo("Split ratios must leave room for a test split.", err=True)
        raise typer.Exit(code=1)

    test_ratio = 1 - train_ratio - validation_ratio
    report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(report)

    rows = list(read_jsonl(input_path))
    split_result = random_split(
        rows,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    write_jsonl(split_result.train, output_dir / "train.jsonl")
    write_jsonl(split_result.validation, output_dir / "validation.jsonl")
    write_jsonl(split_result.test, output_dir / "test.jsonl")

    leakage = detect_split_leakage(
        split_result.train,
        split_result.validation,
        split_result.test,
    )
    warnings = _build_split_warnings(
        len(split_result.validation),
        len(split_result.test),
    )
    if leakage.leaked_group_count > 0:
        warnings.append(
            f"{leakage.rows_shared_across_splits} row(s) in "
            f"{leakage.leaked_group_count} duplicate group(s) are shared across "
            "splits (train/test leakage); dedupe before training to avoid "
            "inflated evaluation scores."
        )

    typer.echo(
        json.dumps(
            {
                "train": len(split_result.train),
                "validation": len(split_result.validation),
                "test": len(split_result.test),
                "output_dir": str(output_dir),
                "train_ratio": train_ratio,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "seed": seed,
                "rows_shared_across_splits": leakage.rows_shared_across_splits,
                "leakage": leakage.model_dump(),
                "warnings": warnings,
            },
            indent=2,
        )
    )


@app.command("eval-run")
def eval_run(
    input_path: Path,
    schema: str,
    model: str = typer.Option(..., "--model", help="Model name to run."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    output_path: Optional[Path] = typer.Option(None, "--output-path", help="Write report JSON."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum examples to run."),
    score_threshold: float = typer.Option(70.0, "--score-threshold"),
    timeout_seconds: int = typer.Option(120, "--timeout-seconds"),
):
    """Run an Evaluation Lab MVP pass against a local model backend."""

    validation_report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(validation_report)

    rows = list(read_jsonl(input_path))
    try:
        examples = extract_evaluation_examples(rows, schema)
        backend_client = _build_backend(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = run_evaluation(
        EvaluationRunConfig(
            dataset=input_path.stem,
            model=model,
            schema_id=schema,
            dataset_path=str(input_path),
            backend=backend,
            base_url=base_url,
            limit=limit,
            score_threshold=score_threshold,
            timeout_seconds=timeout_seconds,
        ),
        examples,
        backend_client,
        limit=limit,
    )
    payload = report.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    typer.echo(payload)


@app.command("ai-assist")
def ai_assist(
    input_path: Path,
    schema: str,
    action: str = typer.Option("review", "--action", help="AI Assist action to run."),
    model: str = typer.Option(..., "--model", help="Model name to run."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    output_path: Optional[Path] = typer.Option(None, "--output-path", help="Write result JSON."),
    user_instruction: Optional[str] = typer.Option(None, "--instruction", help="Reviewer guidance."),
    timeout_seconds: int = typer.Option(120, "--timeout-seconds"),
):
    """Run AI Assist Lab on draft rows and return review-only suggestions."""

    try:
        load_builtin_schema(schema)
        rows = list(read_jsonl(input_path))
        backend_client = _build_backend(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        result = run_ai_assist(
            schema_id=schema,
            action=action,
            rows=rows,
            backend=backend_client,
            model=model,
            user_instruction=user_instruction,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    payload = result.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    typer.echo(payload)


@app.command("backend-health")
def backend_health(
    model: str = typer.Option(..., "--model", help="Model name to check."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    timeout_seconds: int = typer.Option(10, "--timeout-seconds"),
):
    """Check whether a configured model backend is reachable."""

    try:
        backend_client = _build_backend(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = _build_backend_health_report(backend_client)
    typer.echo(report.model_dump_json(indent=2))
    if not report.reachable:
        raise typer.Exit(code=1)


@app.command("model-list")
def model_list(
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    timeout_seconds: int = typer.Option(10, "--timeout-seconds"),
):
    """List models available from a configured local model backend."""

    try:
        backend_client = _build_backend(
            backend=backend,
            model="",
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = _build_backend_model_list_report(backend_client)
    typer.echo(report.model_dump_json(indent=2))
    if not report.reachable:
        raise typer.Exit(code=1)


@app.command("training-config")
def training_config(
    input_path: Path,
    schema: str,
    output_path: Path = typer.Option(..., "--output-path", help="Write rendered config file."),
    base_model: str = typer.Option(..., "--base-model", help="Base model identifier."),
    target: str = typer.Option("axolotl_yaml", "--target", help="Training config target."),
    dataset_format: Optional[str] = typer.Option(
        None,
        "--format",
        help="Training data format label. Defaults to the schema id.",
    ),
    eval_dataset_path: Optional[Path] = typer.Option(
        None,
        "--eval-dataset-path",
        help="Optional validation/eval JSONL path.",
    ),
    sequence_len: int = typer.Option(4096, "--sequence-len"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    micro_batch_size: int = typer.Option(1, "--micro-batch-size"),
    gradient_accumulation_steps: int = typer.Option(8, "--gradient-accumulation-steps"),
    learning_rate: float = typer.Option(0.0002, "--learning-rate"),
):
    """Generate an inspectable Training Lab config without launching training."""

    validation_report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(validation_report)

    if eval_dataset_path is not None:
        eval_report = validate_jsonl_file(eval_dataset_path, schema)
        _exit_if_invalid(eval_report)

    try:
        normalized_target = normalize_training_config_target(target)
        template = build_lora_config_template(
            base_model=base_model,
            dataset_path=str(input_path),
            eval_dataset_path=str(eval_dataset_path) if eval_dataset_path is not None else None,
            dataset_format=dataset_format or schema,
            target=normalized_target,
            sequence_len=sequence_len,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    config_text = render_training_config(template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_text, encoding="utf-8")

    compatibility_warnings = training_compatibility_warnings(
        schema_id=schema,
        dataset_format=dataset_format or schema,
        target=normalized_target,
    )

    warnings = [
        "Training config export only; Corpus Studio does not launch training yet.",
        "Review dataset rights, eval readiness, compute budget, and target tool docs before training.",
    ]
    if eval_dataset_path is None:
        warnings.append("No validation dataset path was provided; generate splits before training.")
    warnings.extend(compatibility_warnings)

    typer.echo(
        json.dumps(
            {
                "target": normalized_target,
                "output_path": str(output_path),
                "training_launcher_implemented": False,
                "config": template.to_training_dict(),
                "config_text": config_text,
                "warnings": warnings,
                "compatibility_warnings": compatibility_warnings,
            },
            indent=2,
        )
    )


@app.command("training-compat")
def training_compat(
    schema: str = typer.Option(..., "--schema", help="Dataset schema id."),
    target: str = typer.Option("axolotl_yaml", "--target", help="Training config target."),
    dataset_format: Optional[str] = typer.Option(
        None, "--format", help="Training data format label. Defaults to the schema id."
    ),
):
    """Report training-config compatibility warnings without generating a config."""
    try:
        load_builtin_schema(schema)
        normalized_target = normalize_training_config_target(target)
    except (ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    warnings = training_compatibility_warnings(
        schema_id=schema,
        dataset_format=dataset_format or schema,
        target=normalized_target,
    )
    typer.echo(
        json.dumps(
            {
                "schema": schema,
                "target": normalized_target,
                "format": dataset_format or schema,
                "compatible": len(warnings) == 0,
                "warnings": warnings,
            },
            indent=2,
        )
    )


@app.command("preference-export")
def preference_export(
    input_path: Path,
    output_path: Path = typer.Option(..., "--output-path", help="Write the reshaped JSONL."),
    export_format: str = typer.Option("dpo", "--format", help="Target format: dpo, kto, or reward."),
    drop_degenerate: bool = typer.Option(
        False,
        "--drop-degenerate",
        help="Exclude empty or identical chosen/rejected pairs before export.",
    ),
):
    """Export preference rows into a trainer-ready format (DPO/KTO/reward)."""
    report = validate_jsonl_file(input_path, "preference")
    _exit_if_invalid(report)

    rows = list(read_jsonl(input_path))
    pair_issues = analyze_preference_pairs(rows)

    export_rows = drop_degenerate_pairs(rows) if drop_degenerate else rows
    dropped = len(rows) - len(export_rows)

    try:
        exported = export_preference(export_rows, export_format)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    write_jsonl(exported, output_path)
    typer.echo(
        json.dumps(
            {
                "format": export_format.strip().lower(),
                "input_rows": len(rows),
                "exported_source_rows": len(export_rows),
                "output_rows": len(exported),
                "dropped_degenerate": dropped,
                "pair_issues": pair_issues.model_dump(),
                "output_path": str(output_path),
                "warnings": _build_preference_warnings(pair_issues, drop_degenerate, dropped),
            },
            indent=2,
        )
    )


@app.command("dataset-card")
def dataset_card(
    project_dir: Path,
    output_path: Optional[Path] = typer.Option(
        None, "--output-path", help="Write the rendered Markdown card."
    ),
    schema: Optional[str] = typer.Option(
        None, "--schema", help="Schema id override. Defaults to the project schema."
    ),
    export_dir: Optional[Path] = typer.Option(
        None,
        "--export-dir",
        help="Export directory holding splits/ and evaluation/ for this project.",
    ),
):
    """Build an inspectable dataset card from a project's existing artifacts."""

    project_file = project_dir / "project.json"
    if not project_file.exists():
        typer.echo(f"Project metadata was not found: {project_file}", err=True)
        raise typer.Exit(code=1)

    try:
        metadata = json.loads(project_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"Project metadata is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    schema_id = schema or metadata.get("schema_id")
    if not schema_id:
        typer.echo("Project metadata is missing a schema id.", err=True)
        raise typer.Exit(code=1)

    try:
        dataset_schema = load_builtin_schema(schema_id)
    except (ValueError, ValidationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    examples_path = project_dir / "examples.jsonl"
    rows = list(read_jsonl(examples_path)) if examples_path.exists() else []

    resolved_export_dir = export_dir or (
        _repo_relative_path("CORPUS_STUDIO_EXPORT_DIR", Path("exports")) / project_dir.name
    )
    splits = _load_split_counts(resolved_export_dir)
    evaluation = _load_latest_evaluation_summary(resolved_export_dir)

    card = build_dataset_card(
        project_id=metadata.get("id", project_dir.name),
        project_name=metadata.get("name", project_dir.name),
        schema=dataset_schema,
        rows=rows,
        created_at=metadata.get("created_at"),
        updated_at=metadata.get("updated_at"),
        splits=splits,
        evaluation=evaluation,
    )
    markdown = render_dataset_card_markdown(card)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    typer.echo(
        json.dumps(
            {
                "output_path": str(output_path) if output_path is not None else None,
                "markdown": markdown,
                "warnings": card.warnings,
                "card": card.model_dump(),
            },
            indent=2,
        )
    )


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_split_counts(export_dir: Path) -> Optional[DatasetCardSplits]:
    split_dir = export_dir / "splits"
    train_path = split_dir / "train.jsonl"
    validation_path = split_dir / "validation.jsonl"
    test_path = split_dir / "test.jsonl"
    if not any(path.exists() for path in (train_path, validation_path, test_path)):
        return None

    return DatasetCardSplits(
        train=_count_jsonl_rows(train_path),
        validation=_count_jsonl_rows(validation_path),
        test=_count_jsonl_rows(test_path),
    )


def _load_latest_evaluation_summary(export_dir: Path) -> Optional[DatasetCardEvaluation]:
    evaluation_dir = export_dir / "evaluation"
    if not evaluation_dir.is_dir():
        return None

    reports = sorted(
        evaluation_dir.glob("*_evaluation_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for report_path in reports:
        try:
            report = EvaluationReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError, OSError):
            continue
        return DatasetCardEvaluation.from_report(report, source_report=report_path.name)

    return None


@app.command()
def export(input_path: Path, output_path: Path, schema: str):
    """Validate and export a JSONL file."""
    report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(report)

    export_jsonl(input_path, output_path)
    typer.echo(f"Exported {input_path} -> {output_path}")


def _build_backend(
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    timeout_seconds: int,
):
    normalized_backend = backend.replace("_", "-").lower()
    if normalized_backend == "ollama":
        config = default_ollama_config(model)
        if base_url is not None:
            config = config.model_copy(update={"base_url": base_url})
        config = config.model_copy(update={"timeout_seconds": timeout_seconds})
        return OllamaBackend(config)

    if normalized_backend in {"openai-compatible", "lm-studio"}:
        config = default_openai_compatible_config(
            model,
            base_url=base_url or "http://localhost:1234/v1",
            api_key=api_key,
        ).model_copy(update={"timeout_seconds": timeout_seconds})
        return OpenAICompatibleBackend(config)

    raise ValueError("Unsupported model backend. Use ollama or openai-compatible.")


def _build_backend_health_report(backend_client) -> BackendHealthReport:
    config = backend_client.config
    try:
        available_models = list(backend_client.list_models())
    except Exception as exc:  # noqa: BLE001 - provider adapters raise varied network exceptions.
        return BackendHealthReport(
            provider_name=config.provider_name,
            base_url=config.base_url,
            model_name=config.model_name,
            reachable=False,
            error=str(exc),
        )

    return BackendHealthReport(
        provider_name=config.provider_name,
        base_url=config.base_url,
        model_name=config.model_name,
        reachable=True,
        model_available=config.model_name in available_models,
        available_models=available_models,
    )


def _build_backend_model_list_report(backend_client) -> BackendModelListReport:
    config = backend_client.config
    try:
        available_models = list(backend_client.list_models())
    except Exception as exc:  # noqa: BLE001 - provider adapters raise varied network exceptions.
        return BackendModelListReport(
            provider_name=config.provider_name,
            base_url=config.base_url,
            reachable=False,
            error=str(exc),
        )

    return BackendModelListReport(
        provider_name=config.provider_name,
        base_url=config.base_url,
        reachable=True,
        models=available_models,
    )


def _ensure_utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr so non-ASCII JSON output never dies on the
    Windows console/pipe code page (cp1252). Safe no-op where a stream does not
    support reconfigure (e.g. some test capture buffers)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


if __name__ == "__main__":
    _ensure_utf8_stdio()
    app()
