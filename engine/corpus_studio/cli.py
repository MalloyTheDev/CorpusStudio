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
from corpus_studio.evaluation.scorers import LlmJudgeScorer
from corpus_studio.exporters.cleaning import clean_rows
from corpus_studio.exporters.jsonl_exporter import export_jsonl, write_jsonl
from corpus_studio.exporters.preference_exporter import (
    analyze_preference_pairs,
    drop_degenerate_pairs,
    export_preference,
)
from corpus_studio.importers.hf_hub import (
    HfImportResult,
    fetch_rows,
    inspect_dataset,
    map_rows,
    suggest_mapping,
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
    try:
        rows = list(read_jsonl(path))
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    report = build_basic_quality_report(rows)
    typer.echo(report.model_dump_json(indent=2))


@app.command("dataset-debt")
def dataset_debt(
    path: Path,
    as_json: bool = typer.Option(False, "--json", help="Emit the DebtReport as JSON."),
):
    """Summarize a dataset's outstanding quality debt as a prioritized, graded ledger.

    Reuses the quality report (no new detection): it normalizes each signal by
    dataset size, ranks the debts, and grades the dataset so you know what to fix
    first. Secrets/PII are graded by presence, never by rate.
    """

    from corpus_studio.reporting.debt_report import build_debt_report, render_debt_report_markdown

    try:
        rows = list(read_jsonl(path))
    except (OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = build_debt_report(build_basic_quality_report(rows))
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(render_debt_report_markdown(report))


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


@app.command("hf-inspect")
def hf_inspect(dataset_id: str):
    """Inspect a public Hugging Face dataset: configs/splits, columns, and license.

    Read-only and public-only (no auth, no upload). Surfaces the license so you
    can decide whether the data may be used for training BEFORE importing.
    """
    try:
        inspection = inspect_dataset(dataset_id)
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not inspect '{dataset_id}': {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(inspection.model_dump_json(indent=2))


@app.command("hf-import")
def hf_import(
    dataset_id: str,
    out: Path = typer.Option(..., "--out", help="Staging JSONL path (NEVER examples.jsonl)."),
    schema: str = typer.Option(..., "--schema", help="Target built-in schema id."),
    config: str = typer.Option("default", "--config", help="Dataset config name."),
    split: str = typer.Option("train", "--split", help="Dataset split name."),
    limit: int = typer.Option(100, "--limit", help="Maximum rows to fetch."),
    map_: list[str] = typer.Option(
        [],
        "--map",
        help="Column mapping as schema_field=hf_column (repeatable); overrides auto-detect.",
    ),
):
    """Import rows from a public Hugging Face dataset into a STAGING JSONL file.

    Read-only and public-only: no auth, no upload. The engine never writes
    examples.jsonl — the staging file flows through the normal import-preview /
    quarantine path so the desktop stays the single writer. Imported data is NOT
    assumed to be training-licensed; the dataset license is reported.
    """
    # The engine must never write the dataset's single source of truth.
    if out.name == "examples.jsonl":
        typer.echo(
            "Refusing to write examples.jsonl: HF import writes a staging file that the "
            "desktop imports through preview/quarantine.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        dataset_schema = load_builtin_schema(schema)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if limit < 1:
        typer.echo("--limit must be at least 1.", err=True)
        raise typer.Exit(code=1)

    # Refuse gated datasets up front (slice 1 is public-only, no auth).
    try:
        inspection = inspect_dataset(dataset_id)
        if inspection.gated:
            typer.echo(
                f"'{dataset_id}' is gated (requires access approval); public import only.",
                err=True,
            )
            raise typer.Exit(code=2)
        page = fetch_rows(dataset_id, config, split, limit)
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not fetch '{dataset_id}': {exc}", err=True)
        raise typer.Exit(code=1) from exc

    mapping = suggest_mapping(page.columns, dataset_schema)
    for pair in map_:
        field_name, _, column = pair.partition("=")
        field_name, column = field_name.strip(), column.strip()
        if not field_name or not column:
            typer.echo(f"Invalid --map '{pair}'; expected schema_field=hf_column.", err=True)
            raise typer.Exit(code=1)
        mapping[field_name] = column

    staged = map_rows(page.rows, mapping)
    write_jsonl(staged, out)

    schema_field_names = [field.name for field in dataset_schema.fields]
    result = HfImportResult(
        dataset_id=dataset_id,
        config=config,
        split=split,
        schema_id=schema,
        fetched_rows=len(staged),
        mapping=mapping,
        unmapped_schema_fields=[name for name in schema_field_names if name not in mapping],
        unused_columns=[c for c in page.columns if c not in mapping.values()],
        license=inspection.license,
        license_note=inspection.license_note,
        out_path=str(out),
    )
    typer.echo(result.model_dump_json(indent=2))


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
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Evaluator model that scores each answer 0-100 (metric=llm_judge). "
        "Omit to use the offline keyword-overlap score.",
    ),
    judge_backend: str = typer.Option("ollama", "--judge-backend", help="Judge backend."),
    judge_base_url: Optional[str] = typer.Option(None, "--judge-base-url", help="Judge provider base URL."),
    judge_api_key: Optional[str] = typer.Option(None, "--judge-api-key", help="Judge API key."),
):
    """Run an Evaluation Lab pass against a local model backend.

    The automatic score is keyword-overlap recall (a lexical proxy, not a quality
    judgment) unless ``--judge-model`` selects an evaluator model to score 0-100 with a
    rationale. The judge provider must be evaluator-authorized by provider policy.
    """

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

    scorer = None
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
            # Constructing the scorer authorizes the judge for evaluation (fail fast).
            scorer = LlmJudgeScorer(judge_client, judge_model, policy=judge_policy)
        except ProviderPolicyError as exc:
            typer.echo(f"Provider policy blocked judging: {exc}", err=True)
            raise typer.Exit(code=2) from exc
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
        scorer=scorer,
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


def _load_artifact_context(project_dir: Path, artifact_id: str):
    """Load an artifact + its integrity + source run + an eval-report loader."""

    from corpus_studio.evaluation.reports import EvaluationReport
    from corpus_studio.training.artifact_registry import (
        artifact_integrity,
        artifact_path,
        load_artifact_record,
    )
    from corpus_studio.training.run_registry import load_run_record, record_path

    path = artifact_path(project_dir, artifact_id)
    if not path.exists():
        raise FileNotFoundError(f"No artifact '{artifact_id}'.")
    artifact = load_artifact_record(path)
    integrity = artifact_integrity(artifact)

    run = None
    run_path = record_path(project_dir, artifact.run_id)
    if run_path.exists():
        try:
            run = load_run_record(run_path)
        except Exception:  # noqa: BLE001
            run = None

    def load_report(report_path: str):
        try:
            return EvaluationReport.model_validate_json(
                Path(report_path).read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError, OSError):
            return None

    return artifact, integrity, run, load_report


@app.command("artifact-card")
def artifact_card(
    project_dir: Path,
    artifact_id: str = typer.Option(..., "--artifact-id"),
):
    """Render a weight card for an artifact (live projection; nothing stored)."""

    from corpus_studio.reporting.weight_card import (
        build_weight_card,
        render_weight_card_markdown,
    )

    try:
        artifact, integrity, run, load_report = _load_artifact_context(project_dir, artifact_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    before = load_report(run.before_eval_path) if run and run.before_eval_path else None
    after = load_report(run.after_eval_path) if run and run.after_eval_path else None
    card = build_weight_card(artifact, run, before, after, integrity)
    typer.echo(render_weight_card_markdown(card))


@app.command("artifact-gate")
def artifact_gate(
    project_dir: Path,
    artifact_id: str = typer.Option(..., "--artifact-id"),
):
    """Promote-gate an artifact (integrity + source-run regression) and save it."""

    from corpus_studio.gates.models import load_gate_thresholds
    from corpus_studio.gates.runner import run_artifact_gate, save_gate_report

    try:
        artifact, integrity, run, load_report = _load_artifact_context(project_dir, artifact_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = run_artifact_gate(
        artifact, integrity, run, load_report,
        thresholds=load_gate_thresholds(project_dir), generated_at=_utc_now_iso(),
    )
    save_gate_report(project_dir, report)
    typer.echo(report.model_dump_json(indent=2))


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

    from corpus_studio.gates.models import load_gate_thresholds

    report = run_training_run_gate(
        record, load_report, thresholds=load_gate_thresholds(project_dir), generated_at=_utc_now_iso()
    )
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


@app.command("gate-thresholds")
def gate_thresholds(project_dir: Path):
    """Show the effective gate thresholds for a project.

    Defaults merged with any project-local gate_thresholds.json. Edit that file
    (create it with these keys) to customize how gates block/warn.
    """

    from corpus_studio.gates.models import load_gate_thresholds

    typer.echo(load_gate_thresholds(project_dir).model_dump_json(indent=2))


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

    from corpus_studio.gates.models import gate_thresholds_path, load_gate_thresholds

    thresholds = load_gate_thresholds(project_dir) if project_dir is not None else None
    if project_dir is None and gate_thresholds_path(input_path.parent).exists():
        typer.echo(
            "Note: a gate_thresholds.json sits next to the input but was NOT applied. "
            "Pass --project-dir to use project thresholds.",
            err=True,
        )

    try:
        rows = list(read_jsonl(input_path))
        generated_at = _utc_now_iso()
        if normalized == "dataset":
            report = run_dataset_gates(
                rows, schema, thresholds=thresholds, target=str(input_path), generated_at=generated_at
            )
        else:
            report = run_export_gates(
                rows, schema, thresholds=thresholds, target=str(input_path), generated_at=generated_at
            )
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if project_dir is not None:
        save_gate_report(project_dir, report)

    typer.echo(report.model_dump_json(indent=2))


def _newest_dataset_gate_report(project_dir: Path) -> Optional[str]:
    """Absolute path to the newest dataset-scope gate report in the project, or None.

    Deterministic: only links a gate report already on disk; never runs a gate.
    """

    from corpus_studio.gates.runner import GATE_REPORTS_DIRNAME, load_gate_report

    directory = project_dir / GATE_REPORTS_DIRNAME
    if not directory.exists():
        return None
    best_path: Optional[Path] = None
    best_key: Optional[str] = None
    for path in directory.glob("dataset-*.json"):
        try:
            report = load_gate_report(path)
        except Exception:  # noqa: BLE001 - a corrupt report is not a link candidate.
            continue
        key = report.generated_at or ""
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    return str(best_path.resolve()) if best_path is not None else None


@app.command("dataset-version-create")
def dataset_version_create(
    project_dir: Path,
    label: str = typer.Option("", "--label", help="Human label for this version."),
    trigger: str = typer.Option(
        "manual",
        "--trigger",
        help="What produced this version: manual | manual_add | import_commit | pre_training.",
    ),
    link_run: list[str] = typer.Option(None, "--link-run", help="Source training run id (repeatable)."),
    link_artifact: list[str] = typer.Option(None, "--link-artifact", help="Model artifact id (repeatable)."),
    eval_report_path: Optional[str] = typer.Option(
        None, "--eval-report-path", help="Absolute path to a linked evaluation report."
    ),
    gate_report_path: Optional[str] = typer.Option(
        None, "--gate-report-path", help="Path to a linked dataset gate report (else newest is auto-linked)."
    ),
    stamp_run: Optional[str] = typer.Option(
        None, "--stamp-run", help="Also write source_snapshot_id=this version onto the given run."
    ),
    store_rows: bool = typer.Option(
        True,
        "--store-rows/--no-store-rows",
        help="Store row bodies in a content-addressed store so this version can be diffed (default: on).",
    ),
):
    """Capture a dataset version: fingerprint + row count of examples.jsonl with pinned lineage links.

    With row storage (the default) it also records each row in a content-addressed
    store plus an ordered manifest, so the version can later be diffed. Reads
    examples.jsonl and writes only under dataset_versions/; it never moves, copies,
    or deletes the dataset or any weight file.
    """

    from datetime import datetime, timezone

    from corpus_studio.versions.row_store import ROW_MANIFEST_ALGO
    from corpus_studio.versions.version_registry import (
        DatasetVersionRecord,
        capture_dataset,
        mint_version_id,
        save_row_manifest,
        save_version_record,
    )

    examples_path = project_dir / "examples.jsonl"
    capture = capture_dataset(examples_path, project_dir, store_rows=store_rows)
    rows_stored = capture.rows_stored
    if capture.content_fingerprint is None:
        typer.echo(
            "Note: examples.jsonl is missing or unreadable; recording a version without a fingerprint.",
            err=True,
        )
    elif store_rows and not rows_stored:
        # Readable dataset, but the row store could not be written: record a
        # fingerprint-only version rather than falsely claiming it is diffable.
        typer.echo(
            "Note: the row store could not be written; recording a fingerprint-only version (not diffable).",
            err=True,
        )
    elif rows_stored and capture.row_count > 0:
        typer.echo(
            f"Stored {capture.row_count} row(s) ({capture.new_rows_stored} new) to the row store.",
            err=True,
        )

    import secrets

    now_dt = datetime.now(timezone.utc)
    # A random token breaks ties: the wall clock can be too coarse to advance
    # between two in-process creates (esp. on Windows), and a pure-timestamp id
    # would collide and silently overwrite the earlier version's file.
    version_id = mint_version_id(
        now_dt.strftime("%Y%m%dT%H%M%S"), f"{now_dt.microsecond:06d}-{secrets.token_hex(3)}"
    )
    now_iso = now_dt.isoformat()

    record = DatasetVersionRecord(
        version_id=version_id,
        created_at=now_iso,
        updated_at=now_iso,
        label=label,
        trigger=trigger,
        row_count=capture.row_count,
        content_fingerprint=capture.content_fingerprint,
        source_run_ids=list(link_run or []),
        artifact_ids=list(link_artifact or []),
        eval_report_path=eval_report_path,
        gate_report_path=gate_report_path or _newest_dataset_gate_report(project_dir),
        rows_stored=rows_stored,
        stored_row_count=capture.row_count if rows_stored else 0,
        row_manifest_algo=ROW_MANIFEST_ALGO if rows_stored else None,
    )

    # Write the ordered manifest (references the store) before the record; the
    # record save below is the commit point.
    if rows_stored:
        save_row_manifest(project_dir, version_id, capture.row_ids)

    run_to_stamp = None
    if stamp_run is not None:
        from corpus_studio.training.run_registry import (
            load_run_record,
            record_path as run_record_path,
            save_run_record,
        )

        run_path = run_record_path(project_dir, stamp_run)
        if not run_path.exists():
            typer.echo(f"No training run '{stamp_run}' to stamp.", err=True)
            raise typer.Exit(code=1)
        run_to_stamp = load_run_record(run_path)
        if stamp_run not in record.source_run_ids:
            record.source_run_ids.append(stamp_run)

    # Commit the version FIRST, then write the run's back-link. If the version
    # save fails, no run is left pointing at a version that was never saved (a
    # version listing a run that lacks the back-link is tolerated; the reverse
    # corrupts lineage).
    save_version_record(project_dir, record)

    if run_to_stamp is not None:
        save_run_record(
            project_dir,
            run_to_stamp.model_copy(update={"source_snapshot_id": version_id, "updated_at": now_iso}),
        )

    typer.echo(record.model_dump_json(indent=2))


@app.command("dataset-version-list")
def dataset_version_list(project_dir: Path):
    """List dataset versions (newest first), each annotated with live integrity."""

    from corpus_studio.versions.version_registry import (
        compute_content_fingerprint,
        integrity_from_fingerprints,
        list_version_records,
    )

    records = list_version_records(project_dir)
    live_fingerprint = compute_content_fingerprint(project_dir / "examples.jsonl")
    versions = []
    for record in records:
        data = record.model_dump()
        data["current_integrity"] = integrity_from_fingerprints(
            record.content_fingerprint, live_fingerprint
        )
        versions.append(data)
    typer.echo(json.dumps({"versions": versions}, indent=2))


@app.command("dataset-version-show")
def dataset_version_show(
    project_dir: Path,
    version_id: str = typer.Option(..., "--version-id"),
    as_json: bool = typer.Option(False, "--json", help="Emit the resolved card as JSON instead of Markdown."),
):
    """Render a dataset version card (live projection; nothing stored)."""

    from corpus_studio.evaluation.reports import EvaluationReport
    from corpus_studio.gates.models import GateReport
    from corpus_studio.training.artifact_registry import (
        artifact_integrity,
        artifact_path,
        load_artifact_record,
    )
    from corpus_studio.training.run_registry import (
        load_run_record,
        record_path as run_record_path,
    )
    from corpus_studio.versions.version_card import (
        build_version_card,
        render_version_card_markdown,
    )
    from corpus_studio.versions.version_registry import (
        compute_content_fingerprint,
        load_version_record,
        record_path,
    )

    path = record_path(project_dir, version_id)
    if not path.exists():
        typer.echo(f"No dataset version '{version_id}'.", err=True)
        raise typer.Exit(code=1)
    record = load_version_record(path)

    live_fingerprint = compute_content_fingerprint(project_dir / "examples.jsonl")

    runs_by_id: dict[str, object] = {}
    for run_id in record.source_run_ids:
        run_path = run_record_path(project_dir, run_id)
        if run_path.exists():
            try:
                runs_by_id[run_id] = load_run_record(run_path)
            except Exception:  # noqa: BLE001 - a corrupt run record resolves as 'not found'.
                pass

    artifacts_by_id: dict[str, tuple[object, str]] = {}
    for artifact_id in record.artifact_ids:
        ap = artifact_path(project_dir, artifact_id)
        if ap.exists():
            try:
                artifact = load_artifact_record(ap)
                artifacts_by_id[artifact_id] = (artifact, artifact_integrity(artifact))
            except Exception:  # noqa: BLE001 - a corrupt artifact resolves as 'not found'.
                pass

    def load_eval(report_path: str):
        try:
            return EvaluationReport.model_validate_json(
                Path(report_path).read_text(encoding="utf-8")
            )
        # ValueError covers json.JSONDecodeError and a non-UTF-8 file
        # (UnicodeDecodeError); a corrupt linked report degrades to a card flag.
        except (ValidationError, ValueError, OSError):
            return None

    def load_gate(report_path: str):
        try:
            return GateReport.model_validate_json(Path(report_path).read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError):
            return None

    card = build_version_card(
        record,
        current_fingerprint=live_fingerprint,
        runs_by_id=runs_by_id,
        artifacts_by_id=artifacts_by_id,
        load_eval_report=load_eval,
        load_gate_report=load_gate,
    )

    if as_json:
        typer.echo(card.model_dump_json(indent=2))
    else:
        typer.echo(render_version_card_markdown(card))


@app.command("dataset-version-diff")
def dataset_version_diff(
    project_dir: Path,
    version_id: str = typer.Option(..., "--version-id", help="Base version."),
    other: str = typer.Option(..., "--other", help="Other version to compare against the base."),
    samples: int = typer.Option(5, "--samples", help="Sample added/removed rows to show."),
    as_json: bool = typer.Option(False, "--json", help="Emit the DatasetVersionDiff as JSON."),
):
    """Diff two dataset versions by their stored row manifests (read-only).

    Reports added/removed/common rows as multisets. Requires both versions to
    have been captured with row storage (``dataset-version-create --store-rows``,
    the default); a version without stored rows cannot be diffed. Never touches
    examples.jsonl.
    """

    from corpus_studio.versions.row_store import load_rows_by_id
    from corpus_studio.versions.version_diff import (
        diff_manifests,
        render_dataset_version_diff_markdown,
    )
    from corpus_studio.versions.version_registry import (
        load_row_manifest,
        load_version_record,
        record_path,
    )

    def manifest_or_exit(vid: str) -> list[str]:
        path = record_path(project_dir, vid)
        if not path.exists():
            typer.echo(f"No dataset version '{vid}'.", err=True)
            raise typer.Exit(code=1)
        try:
            record = load_version_record(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt record degrades cleanly.
            typer.echo(f"Version '{vid}' could not be read (corrupt record).", err=True)
            raise typer.Exit(code=1) from exc
        manifest = load_row_manifest(project_dir, vid)
        if not record.rows_stored or manifest is None:
            typer.echo(
                f"Version '{vid}' has no stored rows; recapture it with row storage "
                "(dataset-version-create --store-rows) to diff.",
                err=True,
            )
            raise typer.Exit(code=1)
        return manifest

    base_ids = manifest_or_exit(version_id)
    other_ids = manifest_or_exit(other)
    diff = diff_manifests(base_ids, other_ids, version_id, other)

    if as_json:
        typer.echo(diff.model_dump_json(indent=2))
        return

    limit = max(samples, 0)
    wanted = set(diff.added_row_ids[:limit]) | set(diff.removed_row_ids[:limit])
    rows = load_rows_by_id(project_dir, wanted) if wanted else {}
    sample_added = [rows[rid] for rid in diff.added_row_ids[:limit] if rid in rows]
    sample_removed = [rows[rid] for rid in diff.removed_row_ids[:limit] if rid in rows]
    typer.echo(render_dataset_version_diff_markdown(diff, sample_added, sample_removed))


@app.command("dataset-version-restore")
def dataset_version_restore(
    project_dir: Path,
    version_id: str = typer.Option(..., "--version-id", help="Version to reconstruct."),
    output: Path = typer.Option(
        ..., "--output", help="File to write the reconstructed rows to (never examples.jsonl)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite --output if it already exists."),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify the reconstruction against the recorded fingerprint (default: on).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the RestoreResult as JSON."),
):
    """Reconstruct a version's exact rows from the row store to --output.

    Rows are rebuilt in canonical form (keys normalized); by default the result is
    verified against the version's recorded fingerprint, proving it is semantically
    identical. The engine NEVER writes examples.jsonl — restore to another path and
    adopt it (in-place restore is a desktop operation). All-or-nothing: if any row
    is missing from the store, or verification fails, nothing is written.
    """

    from corpus_studio.versions.row_store import load_rows_by_id
    from corpus_studio.versions.version_registry import (
        load_row_manifest,
        load_version_record,
        record_path,
    )
    from corpus_studio.versions.version_restore import RestoreResult, reconstruct_and_verify

    path = record_path(project_dir, version_id)
    if not path.exists():
        typer.echo(f"No dataset version '{version_id}'.", err=True)
        raise typer.Exit(code=1)
    try:
        record = load_version_record(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt record degrades cleanly.
        typer.echo(f"Version '{version_id}' could not be read (corrupt record).", err=True)
        raise typer.Exit(code=1) from exc

    manifest = load_row_manifest(project_dir, version_id)
    if not record.rows_stored or manifest is None:
        typer.echo(
            f"Version '{version_id}' has no stored rows; recapture with row storage "
            "(dataset-version-create --store-rows) to restore.",
            err=True,
        )
        raise typer.Exit(code=1)

    # The engine never writes the dataset — refuse to target examples.jsonl.
    examples_path = project_dir / "examples.jsonl"
    try:
        targets_examples = output.resolve() == examples_path.resolve()
    except OSError:
        targets_examples = False
    if targets_examples:
        typer.echo(
            "Refusing to overwrite examples.jsonl; the engine never writes the dataset. "
            "Restore to another path and adopt it via the desktop.",
            err=True,
        )
        raise typer.Exit(code=1)

    if output.is_dir():
        typer.echo(f"--output '{output}' is a directory.", err=True)
        raise typer.Exit(code=1)
    if output.exists() and not force:
        typer.echo(f"'{output}' exists; pass --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    rows_by_id = load_rows_by_id(project_dir, set(manifest))
    lines, _computed, matches, missing_ids = reconstruct_and_verify(
        manifest, rows_by_id, record.content_fingerprint
    )

    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        typer.echo(
            f"{len(missing_ids)} row(s) missing from the store (e.g. {sample}); "
            "cannot faithfully restore.",
            err=True,
        )
        raise typer.Exit(code=1)

    verify_skipped = not verify
    if verify:
        if record.content_fingerprint is None:
            verify_skipped = True
            typer.echo(
                "Warning: cannot verify (no recorded fingerprint); writing a best-effort restore.",
                err=True,
            )
        elif not matches:
            typer.echo(
                "Reconstructed fingerprint does not match the recorded version; "
                "refusing to write a corrupted restore.",
                err=True,
            )
            raise typer.Exit(code=1)

    # Atomic write: a UNIQUE temp file beside --output (mkstemp is exclusive, so it
    # can't clobber a real sibling or race a concurrent restore), then os.replace.
    # Guarded so a write/replace failure (locked/read-only target, bad parent path)
    # degrades to a clean exit-1 and never leaves a dangling temp.
    import tempfile

    content = ("\n".join(lines) + "\n") if lines else ""
    tmp_path: Optional[Path] = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(output.parent), prefix=output.name + ".", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, output)
    except OSError as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        typer.echo(f"Could not write '{output}': {exc}", err=True)
        raise typer.Exit(code=1) from exc

    result = RestoreResult(
        version_id=version_id,
        rows_written=len(lines),
        verified=verify and not verify_skipped and matches,
        verify_skipped=verify_skipped,
        output_path=str(output),
    )
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        status = (
            "verified — fingerprint matches, semantically identical to the recorded version"
            if result.verified
            else ("unverified" if verify_skipped else "written")
        )
        typer.echo(
            f"Restored version {version_id}: {result.rows_written} row(s) -> {output} [{status}]. "
            "Rows are reconstructed in canonical form (keys normalized)."
        )


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
