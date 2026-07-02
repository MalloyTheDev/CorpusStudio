from pathlib import Path
import json
import os
import sqlite3
import sys
from typing import Optional

import typer
from pydantic import ValidationError

from corpus_studio.ai_assist.assistant import run_ai_assist
from corpus_studio.arena.judge import judge_arena
from corpus_studio.arena.runner import load_prompt_suite, run_arena
from corpus_studio.arena.storage import save_arena_report
from corpus_studio.gates.runner import (
    run_dataset_gates,
    run_export_gates,
    save_gate_report,
)
from corpus_studio.providers.overrides import (
    approve_generation,
    load_overrides,
    revoke_generation,
)
from corpus_studio.providers.policy import (
    DEFAULT_PROVIDER_POLICIES,
    ProviderPolicyError,
    ProviderRole,
    infer_provider_id,
    resolve_policy,
)
from corpus_studio.evaluation.benchmark import build_benchmark_report
from corpus_studio.evaluation.evaluator import (
    EvaluationRunConfig,
    extract_evaluation_examples,
    run_evaluation,
)
from corpus_studio.exporters.cleaning import clean_rows
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
from corpus_studio.training.estimators import (
    build_training_token_budget,
    build_vram_estimate,
    parse_parameter_count,
    recommend_lora,
)
from corpus_studio.training.launch import (
    build_launch_plan,
    find_checkpoints,
    latest_checkpoint,
)
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


@app.command("benchmark")
def benchmark(
    input_path: Path,
    schema: str,
    models: list[str] = typer.Option(..., "--model", help="Model to benchmark (repeatable)."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    output_path: Optional[Path] = typer.Option(None, "--output-path", help="Write report JSON."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum examples per model."),
    score_threshold: float = typer.Option(70.0, "--score-threshold"),
    timeout_seconds: int = typer.Option(120, "--timeout-seconds"),
):
    """Benchmark one dataset across several models and compare/rank them."""

    validation_report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(validation_report)

    unique_models = list(dict.fromkeys(name.strip() for name in models if name.strip()))
    if not unique_models:
        typer.echo("Provide at least one --model.", err=True)
        raise typer.Exit(code=1)

    rows = list(read_jsonl(input_path))
    try:
        examples = extract_evaluation_examples(rows, schema)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    reports = []
    for model in unique_models:
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

        reports.append(
            run_evaluation(
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
        )

    benchmark_report = build_benchmark_report(input_path.stem, reports)
    payload = json.dumps(
        {
            "benchmark": benchmark_report.model_dump(),
            "model_reports": [report.model_dump() for report in reports],
        },
        indent=2,
    )
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

    provider_id = infer_provider_id(backend, base_url)
    route_id = model if provider_id == "openrouter" else None
    policy = resolve_policy(
        provider_id,
        model_id=model,
        route_id=route_id,
        overrides=load_overrides(input_path.parent),
    )

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
            policy=policy,
        )
    except ProviderPolicyError as exc:
        typer.echo(f"Provider policy blocked this action: {exc}", err=True)
        raise typer.Exit(code=2) from exc
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
    training_output_dir: str = typer.Option(
        "output",
        "--training-output-dir",
        help="Where the trainer writes checkpoints (relative paths resolve against the config's directory).",
    ),
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
            output_dir=training_output_dir,
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

    token_budget = build_training_token_budget(list(read_jsonl(input_path)), sequence_len)
    launch_plan = build_launch_plan(normalized_target, str(output_path))

    # Relative output dirs resolve against the config's directory (the launch CWD).
    resolved_output_dir = Path(training_output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = output_path.parent / resolved_output_dir

    vram_estimate = build_vram_estimate(
        base_model,
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
    )
    lora_recommendation = recommend_lora(
        parse_parameter_count(base_model), lora_r, lora_alpha
    )

    compatibility_warnings = training_compatibility_warnings(
        schema_id=schema,
        dataset_format=dataset_format or schema,
        target=normalized_target,
    )

    warnings = [
        "This command exports the config only; launch it with the emitted command or "
        "from the desktop Training tab.",
        "Review dataset rights, eval readiness, compute budget, and target tool docs before training.",
    ]
    if eval_dataset_path is None:
        warnings.append("No validation dataset path was provided; generate splits before training.")
    if token_budget.examples_over_sequence_len > 0:
        warnings.append(
            f"{token_budget.examples_over_sequence_len} example(s) exceed sequence_len="
            f"{sequence_len} (est. {token_budget.method}) and will be truncated."
        )
    if vram_estimate.parameter_count_billions is None:
        warnings.append(
            "No VRAM estimate: could not parse a parameter count from the base model name."
        )
    warnings.extend(lora_recommendation.warnings)
    warnings.extend(compatibility_warnings)

    typer.echo(
        json.dumps(
            {
                "target": normalized_target,
                "output_path": str(output_path),
                "training_launcher_implemented": True,
                "config": template.to_training_dict(),
                "config_text": config_text,
                "token_budget": token_budget.model_dump(),
                "launch": launch_plan.model_dump(),
                "training_output_dir": str(resolved_output_dir),
                "vram_estimate": vram_estimate.model_dump(),
                "lora_recommendation": lora_recommendation.model_dump(),
                "warnings": warnings,
                "compatibility_warnings": compatibility_warnings,
            },
            indent=2,
        )
    )


@app.command("training-run-list")
def training_run_list(
    project_dir: Path,
):
    """List durable training run records for a project (newest first).

    Reconciles any ``running`` record whose process is gone to ``interrupted``
    (best-effort pid liveness) and persists the change, so the headless view
    honors the same 'reconcile on load' invariant the desktop enforces.
    """

    from corpus_studio.training.run_registry import (
        list_run_records,
        pid_alive,
        reconcile_running_records,
        save_run_record,
    )

    records = list_run_records(project_dir)
    prior = {record.run_id: record.status for record in records}
    reconciled = reconcile_running_records(records, pid_alive, _utc_now_iso())
    for record in reconciled:
        if prior.get(record.run_id) != record.status:
            save_run_record(project_dir, record)

    typer.echo(json.dumps({"runs": [record.model_dump() for record in reconciled]}, indent=2))


@app.command("training-run-update")
def training_run_update(
    project_dir: Path,
    run_id: str = typer.Option(..., "--run-id", help="Run to update."),
    status: Optional[str] = typer.Option(None, "--status", help="New status."),
    exit_code: Optional[int] = typer.Option(None, "--exit-code"),
    after_eval_path: Optional[str] = typer.Option(None, "--after-eval-path"),
    after_eval_model: Optional[str] = typer.Option(None, "--after-eval-model"),
):
    """Headless status/eval-link update with light transition validation."""

    from corpus_studio.training.run_registry import (
        load_run_record,
        record_path,
        save_run_record,
        validate_transition,
    )

    path = record_path(project_dir, run_id)
    if not path.exists():
        typer.echo(f"No run record for '{run_id}'.", err=True)
        raise typer.Exit(code=1)

    record = load_run_record(path)
    updates: dict[str, object] = {"updated_at": _utc_now_iso()}
    if status is not None:
        try:
            validate_transition(record.status, status)
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        updates["status"] = status
    if exit_code is not None:
        updates["exit_code"] = exit_code
    if after_eval_path is not None:
        updates["after_eval_path"] = after_eval_path
    if after_eval_model is not None:
        updates["after_eval_model"] = after_eval_model

    saved = record.model_copy(update=updates)
    save_run_record(project_dir, saved)
    typer.echo(saved.model_dump_json(indent=2))


@app.command("artifact-register")
def artifact_register(
    project_dir: Path,
    run_id: str = typer.Option(..., "--run-id", help="Source training run."),
    path: str = typer.Option(..., "--path", help="Path to the adapter/checkpoint (referenced, never moved)."),
    kind: str = typer.Option("adapter", "--kind", help="User label (e.g. adapter, checkpoint)."),
    notes: str = typer.Option("", "--notes"),
):
    """Register (idempotently) a model artifact produced by a training run."""

    from corpus_studio.training.artifact_registry import register_artifact

    try:
        record = register_artifact(project_dir, run_id, path, kind=kind, notes=notes, now=_utc_now_iso())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(record.model_dump_json(indent=2))


@app.command("artifact-list")
def artifact_list(project_dir: Path):
    """List model artifacts (newest first) with computed path integrity."""

    from corpus_studio.training.artifact_registry import artifact_integrity, list_artifacts

    _ = artifact_integrity  # (kept for clarity; integrity is returned by list_artifacts)
    artifacts = [
        {**record.model_dump(), "integrity": integrity}
        for record, integrity in list_artifacts(project_dir)
    ]
    typer.echo(json.dumps({"artifacts": artifacts}, indent=2))


@app.command("artifact-update")
def artifact_update(
    project_dir: Path,
    artifact_id: str = typer.Option(..., "--artifact-id"),
    status: str = typer.Option(..., "--status", help="candidate | kept | rejected."),
):
    """Update an artifact's keep/reject status."""

    from corpus_studio.training.artifact_registry import update_artifact_status

    try:
        record = update_artifact_status(project_dir, artifact_id, status, now=_utc_now_iso())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(record.model_dump_json(indent=2))


@app.command("training-run-gate")
def training_run_gate_command(
    project_dir: Path,
    run_id: str = typer.Option(..., "--run-id", help="Run to regression-gate."),
):
    """Regression-gate a training run using its linked before/after eval reports."""

    from corpus_studio.evaluation.reports import EvaluationReport
    from corpus_studio.gates.runner import run_training_run_gate, save_gate_report
    from corpus_studio.training.run_registry import load_run_record, record_path

    path = record_path(project_dir, run_id)
    if not path.exists():
        typer.echo(f"No run record for '{run_id}'.", err=True)
        raise typer.Exit(code=1)

    record = load_run_record(path)

    def load_report(report_path: str) -> Optional[EvaluationReport]:
        try:
            return EvaluationReport.model_validate_json(
                Path(report_path).read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError, OSError):
            return None

    report = run_training_run_gate(record, load_report, generated_at=_utc_now_iso())
    save_gate_report(project_dir, report)
    typer.echo(report.model_dump_json(indent=2))


@app.command("training-checkpoints")
def training_checkpoints(
    output_dir: Path,
    target: str = typer.Option("axolotl_yaml", "--target", help="Training config target."),
    config_path: Optional[Path] = typer.Option(
        None, "--config-path", help="Rendered config to build a resume command for."
    ),
):
    """List training checkpoints in an output directory and build a resume command."""

    normalized_target = normalize_training_config_target(target)
    checkpoints = find_checkpoints(output_dir)
    latest = latest_checkpoint(output_dir)

    resume_command = None
    resume_argv: list[str] | None = None
    resume_supported = None
    if config_path is not None:
        plan = build_launch_plan(
            normalized_target,
            str(config_path),
            resume_checkpoint=str(output_dir / latest) if latest else None,
        )
        resume_command = plan.resume_command
        resume_argv = plan.resume_argv
        resume_supported = plan.resume_supported

    typer.echo(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoints": checkpoints,
                "latest_checkpoint": latest,
                "resume_command": resume_command,
                "resume_argv": resume_argv,
                "resume_supported": resume_supported,
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
def export(
    input_path: Path,
    output_path: Path,
    schema: str,
    dedupe: bool = typer.Option(
        False,
        "--dedupe",
        help="Drop exact and normalized-duplicate rows before export.",
    ),
    drop_low_information: bool = typer.Option(
        False,
        "--drop-low-information",
        help="Drop rows below the low-information token threshold.",
    ),
):
    """Validate and export a JSONL file, optionally cleaning it first."""
    report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(report)

    warnings: list[str] = []

    if dedupe or drop_low_information:
        rows = list(read_jsonl(input_path))
        kept, clean_result = clean_rows(
            rows,
            dedupe=dedupe,
            drop_low_information=drop_low_information,
        )
        write_jsonl(kept, output_path)

        manifest_path = output_path.with_name(output_path.name + ".cleaning_manifest.json")
        manifest_path.write_text(
            clean_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        payload = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "cleaned": True,
            "input_rows": clean_result.input_rows,
            "output_rows": clean_result.kept_rows,
            "removed_rows": clean_result.removed_rows,
            "removed_exact_duplicates": clean_result.removed_exact_duplicates,
            "removed_normalized_duplicates": clean_result.removed_normalized_duplicates,
            "removed_low_information": clean_result.removed_low_information,
            "manifest_path": str(manifest_path),
            "warnings": warnings,
        }
    else:
        # Verbatim copy, but still surface remaining duplicates so the quality
        # surfaces are not purely advisory when the deliverable is produced.
        quality = build_basic_quality_report(list(read_jsonl(input_path)))
        export_jsonl(input_path, output_path)
        if quality.duplicate_exact_count or quality.duplicate_normalized_count:
            warnings.append(
                f"Exported without cleaning: {quality.duplicate_exact_count} exact and "
                f"{quality.duplicate_normalized_count} normalized duplicate row(s) remain. "
                "Re-run with --dedupe to remove them."
            )
        payload = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "cleaned": False,
            "input_rows": quality.example_count,
            "output_rows": quality.example_count,
            "removed_rows": 0,
            "warnings": warnings,
        }

    typer.echo(json.dumps(payload, indent=2))


@app.command("arena-run")
def arena_run(
    input_path: Path,
    models: list[str] = typer.Option(..., "--model", help="Model to run (repeatable)."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Override provider base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key."),
    output_path: Optional[Path] = typer.Option(None, "--output-path", help="Write report JSON."),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", help="Save under arena_reports/."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Maximum prompts to run."),
    timeout_seconds: int = typer.Option(120, "--timeout-seconds"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Evaluator model that ranks responses."),
    judge_backend: str = typer.Option("ollama", "--judge-backend", help="Judge backend."),
    judge_base_url: Optional[str] = typer.Option(None, "--judge-base-url", help="Judge provider base URL."),
    judge_api_key: Optional[str] = typer.Option(None, "--judge-api-key", help="Judge API key."),
):
    """Run a prompt suite across several models and capture responses side by side.

    With ``--judge-model`` an evaluator model scores the responses and picks a
    winner. Judging is an evaluator activity, so evaluator-only providers
    (OpenAI/Anthropic) are permitted as the judge.
    """

    try:
        prompts = load_prompt_suite(input_path)
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if not prompts:
        typer.echo("No prompts found in the suite (need rows with a non-empty 'prompt').", err=True)
        raise typer.Exit(code=1)

    unique_models = list(dict.fromkeys(name.strip() for name in models if name.strip()))
    if not unique_models:
        typer.echo("Provide at least one --model.", err=True)
        raise typer.Exit(code=1)

    try:
        model_backends = [
            (
                model,
                _build_backend(
                    backend=backend,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                ),
            )
            for model in unique_models
        ]
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = run_arena(prompts, model_backends, limit=limit, generated_at=_utc_now_iso())

    if judge_model is not None:
        judge_provider = infer_provider_id(judge_backend, judge_base_url)
        judge_route = judge_model if judge_provider == "openrouter" else None
        judge_policy = resolve_policy(
            judge_provider,
            model_id=judge_model,
            route_id=judge_route,
            overrides=load_overrides(input_path.parent),
        )
        try:
            judge_client = _build_backend(
                backend=judge_backend,
                model=judge_model,
                base_url=judge_base_url,
                api_key=judge_api_key,
                timeout_seconds=timeout_seconds,
            )
            report = judge_arena(report, judge_client, judge_model, policy=judge_policy)
        except ProviderPolicyError as exc:
            typer.echo(f"Provider policy blocked judging: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

    payload = report.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if project_dir is not None:
        save_arena_report(project_dir, report, input_path.stem)

    typer.echo(payload)


@app.command("provider-policy")
def provider_policy(
    provider: Optional[str] = typer.Option(None, "--provider", help="Show one provider's effective policy."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id (for local approval scope)."),
    route: Optional[str] = typer.Option(None, "--route", help="OpenRouter route id."),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", help="Apply this project's overrides."),
):
    """Show effective provider role policies (with any project overrides applied)."""

    overrides = load_overrides(project_dir) if project_dir is not None else {}
    if provider is not None:
        route_id = route or (model if provider == "openrouter" else None)
        policy = resolve_policy(provider, model_id=model, route_id=route_id, overrides=overrides)
        typer.echo(policy.model_dump_json(indent=2))
        return

    policies = {
        provider_id: resolve_policy(provider_id, overrides=overrides).model_dump(mode="json")
        for provider_id in DEFAULT_PROVIDER_POLICIES
    }
    typer.echo(json.dumps({"providers": policies}, indent=2))


@app.command("provider-approve")
def provider_approve(
    provider: str = typer.Option(..., "--provider", help="Provider id (e.g. ollama, openrouter)."),
    project_dir: Path = typer.Option(..., "--project-dir", help="Where to write the approval override."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id to approve."),
    route: Optional[str] = typer.Option(None, "--route", help="OpenRouter route id to approve."),
    revoke: bool = typer.Option(False, "--revoke", help="Remove the approval instead."),
):
    """Approve (or revoke) trainable generation for a specific local model/route."""

    route_id = route or (model if provider == "openrouter" else None)

    if revoke:
        removed = revoke_generation(project_dir, provider, model_id=model, route_id=route_id)
        typer.echo(json.dumps({"revoked": removed, "provider": provider, "model": model, "route": route_id}))
        return

    # A provider that is evaluator-only by role cannot be approved for generation.
    resolved = resolve_policy(provider, model_id=model, route_id=route_id)
    generator_allowed = (
        ProviderRole.TRAINABLE_OUTPUT_GENERATOR in resolved.allowed_roles
        and ProviderRole.TRAINABLE_OUTPUT_GENERATOR not in resolved.blocked_roles
    )
    if not generator_allowed:
        typer.echo(
            f"{provider} is evaluator-only and cannot be approved for trainable generation.",
            err=True,
        )
        raise typer.Exit(code=2)

    key = approve_generation(project_dir, provider, model_id=model, route_id=route_id)
    effective = resolve_policy(
        provider, model_id=model, route_id=route_id, overrides=load_overrides(project_dir)
    )
    typer.echo(
        json.dumps(
            {
                "approved_key": key,
                "can_generate_trainable": effective.can_generate_trainable(),
                "requires_human_review": effective.requires_human_review,
            }
        )
    )


@app.command("gate-run")
def gate_run(
    input_path: Path,
    schema: str,
    scope: str = typer.Option("dataset", "--scope", help="dataset or export."),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", help="Write report under gate_reports/."),
):
    """Run gates over a dataset and emit a serializable pass/warn/block report."""

    normalized = scope.strip().lower()
    if normalized not in {"dataset", "export"}:
        typer.echo("Unsupported gate scope. Use dataset or export.", err=True)
        raise typer.Exit(code=1)

    try:
        rows = list(read_jsonl(input_path))
        generated_at = _utc_now_iso()
        if normalized == "dataset":
            report = run_dataset_gates(rows, schema, target=str(input_path), generated_at=generated_at)
        else:
            report = run_export_gates(rows, schema, target=str(input_path), generated_at=generated_at)
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if project_dir is not None:
        save_gate_report(project_dir, report)

    typer.echo(report.model_dump_json(indent=2))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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
