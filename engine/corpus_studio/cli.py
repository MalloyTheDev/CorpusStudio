from pathlib import Path
import contextlib
import json
import os
import sqlite3
import sys
import tempfile
from typing import Any, Optional

import typer
from pydantic import ValidationError

from corpus_studio.ai_assist.assistant import run_ai_assist
from corpus_studio.arena.judge import judge_arena
from corpus_studio.arena.runner import load_prompt_suite, run_arena
from corpus_studio.arena.storage import save_arena_report
from corpus_studio.gates.runner import (
    run_chat_gates,
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
    should_report_progress,
)
from corpus_studio.evaluation.scorers import LlmJudgeScorer
from corpus_studio.exporters.cleaning import clean_rows
from corpus_studio.exporters.redaction import redact_rows
from corpus_studio.exporters.jsonl_exporter import export_jsonl, write_jsonl
from corpus_studio.exporters.tabular_exporter import (
    schema_is_csv_exportable,
    write_tabular,
)
from corpus_studio.exporters.preference_exporter import (
    analyze_preference_pairs,
    drop_degenerate_pairs,
    export_preference,
)
from corpus_studio.importers.hf_hub import (
    MAX_IMPORT_ROWS,
    HfImportResult,
    fetch_rows,
    inspect_dataset,
    map_rows,
    suggest_mapping,
)
from corpus_studio.importers.jsonl_importer import read_jsonl
from corpus_studio.importers.jsonl_preview import preview_jsonl_import
from corpus_studio.importers.tabular_importer import convert_tabular_to_jsonl
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
from corpus_studio.suites.models import SuiteCase
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


@app.command("platform-schemas")
def platform_schemas(
    out_dir: Optional[Path] = typer.Option(
        None, "--out", help="Write each contract's JSON Schema to this directory (+ index.json)."
    ),
):
    """Emit the language-neutral platform contract JSON Schemas (RunPlan, RunEvent, BackendManifest,
    EnvironmentProfile, FailureRecord, FitClassification, WorkerMessage, …) — the boundary the
    Rust core / Avalonia / Tauri clients consume. With --out, writes <Name>.schema.json files;
    otherwise prints the whole schema set to stdout."""
    from corpus_studio.platform import CONTRACT_VERSION, contract_schemas, export_json_schemas

    if out_dir is not None:
        written = export_json_schemas(out_dir)
        typer.echo(
            json.dumps(
                {"contract_version": CONTRACT_VERSION, "written": [str(p) for p in written]},
                indent=2,
            )
        )
        return
    typer.echo(
        json.dumps({"contract_version": CONTRACT_VERSION, "contracts": contract_schemas()}, indent=2)
    )


@app.command("platform-probe")
def platform_probe(
    json_out: bool = typer.Option(False, "--json", help="Emit the full EnvironmentProfile + CapabilityReport JSON."),
    out_dir: Optional[Path] = typer.Option(
        None, "--out", help="Write EnvironmentProfile.json + CapabilityReport.json to this directory."
    ),
    cache: bool = typer.Option(
        False, "--cache", help="Reuse a cached CapabilityReport when the host signature is unchanged (skips the probes)."
    ),
    store: Optional[Path] = typer.Option(
        None, "--store", help="Profile cache directory (default: ~/.corpus_studio/profiles)."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-run the probes and update the cache even on a signature hit."
    ),
):
    """Profile the current host and run the functional capability probes — 'readiness = a kernel
    actually ran', not 'the package imports'. Emits an EnvironmentProfile (OS/residency/GPUs/package
    locks + signature) and a CapabilityReport (per-probe PASS/KERNEL_STALL/… + effective
    capabilities + ready/cpu_toy_only/not_ready). With --cache, an unchanged host reuses the stored
    report instead of re-running the (torch-loading) probes."""
    from corpus_studio.platform.profiler import build_environment_profile
    from corpus_studio.platform.probes import run_capability_probes

    # Redirect probe-time stdout to stderr so a library banner printed at import (historically older
    # bitsandbytes) can't corrupt the JSON a client parses off stdout; the human/JSON output is below.
    with contextlib.redirect_stdout(sys.stderr):
        if cache:
            from corpus_studio.platform.profile_store import default_store_dir, resolve_capabilities

            resolved = resolve_capabilities(
                store or default_store_dir(),
                build_profile=build_environment_profile,
                run_probes=run_capability_probes,
                refresh=refresh,
            )
            profile, report = resolved.profile, resolved.report
            typer.echo(
                f"capabilities: {'cached' if resolved.cached else 'freshly probed'} "
                f"(signature {profile.environment_signature[:12]}…)",
                err=True,
            )
        else:
            profile = build_environment_profile()
            report = run_capability_probes(profile)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "EnvironmentProfile.json").write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
        (out_dir / "CapabilityReport.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )

    if json_out:
        typer.echo(
            json.dumps(
                {
                    "environment_profile": profile.model_dump(mode="json"),
                    "capability_report": report.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return

    gpu = profile.gpus[0].name if profile.gpus else "none detected"
    lines = [
        "Platform probe",
        f"  OS: {profile.host.os.value} ({profile.host.memory_residency_model.value})",
        f"  GPU: {gpu}",
        f"  env signature: {profile.environment_signature[:12]}…",
        f"  READINESS: {report.readiness}",
    ]
    for result in report.probe_results:
        lines.append(f"    {result.outcome.value:<12} {result.probe}"
                     + (f"  — {result.detail}" if result.detail else ""))
    if report.effective_capabilities is not None:
        eff = report.effective_capabilities
        proven = (
            [m.value for m in eff.precision_modes]
            + [m.value for m in eff.quantization_modes]
            + [m.value for m in eff.attention_impls]
        )
        lines.append(f"  proven on this host: {', '.join(proven) if proven else '(none)'}")
    typer.echo("\n".join(lines))


@app.command("platform-profiles")
def platform_profiles(
    store: Optional[Path] = typer.Option(
        None, "--store", help="Profile cache directory (default: ~/.corpus_studio/profiles)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the cached signatures as JSON."),
):
    """List the cached host profiles (from `platform-probe --cache`): one line per environment
    signature with its GPU + readiness verdict, so a re-probe is only needed when the host changed."""
    from corpus_studio.platform.profile_store import (
        default_store_dir,
        list_signatures,
        load_profile,
        load_report,
    )

    store_dir = store or default_store_dir()
    signatures = list_signatures(store_dir)
    if json_out:
        typer.echo(json.dumps({"store": str(store_dir), "signatures": signatures}, indent=2))
        return
    if not signatures:
        typer.echo(f"No cached profiles in {store_dir}.")
        return
    typer.echo(f"Cached profiles in {store_dir}:")
    for signature in signatures:
        profile = load_profile(signature, store_dir)
        report = load_report(signature, store_dir)
        gpu = profile.gpus[0].name if profile and profile.gpus else "no GPU"
        readiness = report.readiness if report else "?"
        typer.echo(f"  {signature[:12]}…  {gpu:<24}  {readiness}")


@app.command("platform-run")
def platform_run(
    plan_path: Optional[Path] = typer.Argument(
        None, help="Path to a RunPlan JSON. Omit and pass --demo to run the built-in echo plan."
    ),
    demo: bool = typer.Option(
        False, "--demo", help="Execute a built-in minimal plan (echo needs nothing; cpu_toy needs [train])."
    ),
    runner_name: str = typer.Option(
        "echo",
        "--runner",
        help="Runner: echo | cpu_toy | training. 'training' dispatches by the plan's backend "
        "(corpus_studio / unsloth — see 'platform-backends').",
    ),
    max_steps: Optional[int] = typer.Option(
        None, "--max-steps", help="Cap optimizer steps (cpu_toy / training runners)."
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out", help="Write the terminal RunManifest.json to this directory (atomic)."
    ),
    subprocess_mode: bool = typer.Option(
        False,
        "--subprocess",
        help="Run in a supervised CHILD process — the parent can KILL a hung run (a stall becomes a "
        "real KERNEL_STALL) and a backend crash is isolated from the core.",
    ),
    silence_timeout: float = typer.Option(
        600.0, "--timeout", help="[--subprocess] Kill the worker after this many seconds of silence."
    ),
):
    """Execute a RunPlan through the headless run supervisor: stream RunEvents to stderr and produce
    a RunManifest on stdout. 'echo' is a dependency-light no-op that proves the supervisor without a
    GPU or the [train] extra; 'cpu_toy' / 'training' run the real trainer (via the TrainingRunner,
    reading the plan's training_config_snapshot). 'training' dispatches to the framework the plan
    picked (backend_ref: corpus_studio → the first-party trainer, unsloth → the Unsloth backend).
    --subprocess runs it in a supervised child process the parent can time out + KILL (a hung run
    becomes a real KERNEL_STALL; a crash is isolated). The RunManifest classifies the terminal state
    (succeeded / failed / cancelled) with a FailureRecord taxonomy on abnormal termination."""
    from corpus_studio.platform.contracts import RunPlan
    from corpus_studio.platform.supervisor import EchoRunner, Runner, demo_run_plan, execute_run

    if runner_name not in ("echo", "cpu_toy", "training"):
        typer.echo(f"Unknown runner '{runner_name}' (echo | cpu_toy | training).", err=True)
        raise typer.Exit(2)

    if demo:
        if runner_name == "echo":
            plan = demo_run_plan()
        else:
            from corpus_studio.platform.runners import demo_training_plan

            plan = demo_training_plan(plan_id=f"demo-{runner_name}")
    elif plan_path is not None:
        plan = RunPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    else:
        typer.echo("Provide a RunPlan path argument, or pass --demo.", err=True)
        raise typer.Exit(2)

    if subprocess_mode:
        from corpus_studio.platform.subprocess_supervisor import execute_run_subprocess

        result = execute_run_subprocess(
            plan,
            runner_name=runner_name,
            max_steps=max_steps,
            silence_timeout_s=silence_timeout,
            out_dir=out_dir,
        )
    else:
        if runner_name == "echo":
            runner: Runner = EchoRunner()
        else:
            from corpus_studio.platform.runners import TrainingRunner

            runner = TrainingRunner(cpu_toy=(runner_name == "cpu_toy"), max_steps=max_steps)
        result = execute_run(plan, runner, out_dir=out_dir)
    for event in result.events:
        typer.echo(event.model_dump_json(), err=True)
    if out_dir is not None and result.artifacts:
        typer.echo(
            f"wrote {len(result.artifacts)} artifact manifest(s) to {out_dir}/artifacts/", err=True
        )
    typer.echo(result.manifest.model_dump_json(indent=2))
    if result.manifest.state != "succeeded":
        raise typer.Exit(1)


@app.command("platform-plan")
def platform_plan(
    base_model: str = typer.Option(..., "--base-model", help="The base model to fine-tune."),
    dataset_path: str = typer.Option(..., "--dataset", help="Path to the training JSONL."),
    dataset_ref: str = typer.Option("dataset", "--dataset-ref", help="Stable id for the dataset the plan references."),
    task_type: str = typer.Option("sft", "--task-type", help="Training task type (sft / preference / …)."),
    dataset_format: str = typer.Option("instruction", "--dataset-format", help="Row format: instruction (Alpaca) or chat (messages)."),
    output_dir: str = typer.Option("output", "--output-dir", help="Where the trainer saves the adapter (flows into the plan's training snapshot)."),
    sequence_len: int = typer.Option(4096, "--sequence-len", help="Max sequence length (flows into the plan verbatim)."),
    backend: str = typer.Option("corpus_studio", "--backend", help="Training framework to run on (see platform-backends)."),
    optim: Optional[str] = typer.Option(None, "--optim", help="Optimizer (e.g. adamw_torch | paged_adamw_8bit). Paged pages optimizer state to host RAM under pressure — a spike-safe safe-spill; validated against the backend."),
    use_liger: bool = typer.Option(False, "--use-liger", help="Fuse the cross-entropy loss (Liger) to drop the long-seq logits memory spike. Needs liger-kernel; Blackwell support unverified."),
    memory_efficient: bool = typer.Option(False, "--memory-efficient", help="Shortcut for a tight GPU: enable the memory-saving levers (paged optimizer + fused Liger loss). Explicit --optim / --use-liger override it."),
    allow_cpu_toy: bool = typer.Option(False, "--allow-cpu-toy", help="Permit a cpu-toy plan when the host is cpu-toy-only."),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Write the sealed RunPlan.json to this directory."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit {run_plan, fit_classification} as one JSON bundle to stdout."
    ),
):
    """Profile the host, prove its capabilities, and RESOLVE an immutable, hash-sealed RunPlan — the
    goal+data+hardware → runnable-plan step. Every ambiguous field (precision / quantization /
    attention / adapter) is resolved against what PROVED to work on THIS host: bf16 only when proven,
    nf4 only when bitsandbytes passed, and math attention on Blackwell (sm_120). An unready host is a
    clean PlannerError, never a silent downgrade."""
    from corpus_studio.platform.common import Ref
    from corpus_studio.platform.planner import PlannerConstraints, PlannerError, build_run_plan
    from corpus_studio.platform.probes import run_capability_probes
    from corpus_studio.platform.profiler import build_environment_profile

    # Harden the JSON contract: a probe that lazily imports a library which prints a banner to stdout
    # (historically older bitsandbytes) would prepend non-JSON bytes and break the client's parse.
    # Redirect probe-time stdout to stderr; the bundle is emitted to the real stdout below.
    with contextlib.redirect_stdout(sys.stderr):
        profile = build_environment_profile()
        report = run_capability_probes(profile)
    # --memory-efficient is a shortcut; explicit --optim / --use-liger win.
    resolved_optim = optim or ("paged_adamw_8bit" if memory_efficient else "adamw_torch")
    constraints = PlannerConstraints(
        base_model=base_model,
        dataset_path=dataset_path,
        task_type=task_type,
        dataset_format=dataset_format,
        output_dir=output_dir,
        sequence_len=sequence_len,
        backend=backend,
        optim=resolved_optim,
        use_liger=use_liger or memory_efficient,
        allow_cpu_toy=allow_cpu_toy,
    )
    try:
        plan = build_run_plan(
            profile=profile,
            capabilities=report,
            dataset_ref=Ref(id=dataset_ref),
            constraints=constraints,
            plan_id=f"plan-{profile.environment_signature[:8]}",
        )
    except PlannerError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc

    from corpus_studio.platform.calibrator import classify_fit

    fit = classify_fit(plan, profile)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "RunPlan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        (out_dir / "FitClassification.json").write_text(fit.model_dump_json(indent=2), encoding="utf-8")
    if json_out:
        # One bundle for a client (the Tauri shell / apps/web live flow): the sealed plan + the
        # predicted, not-measured fit. stdout stays pure JSON; the human line is stderr-only above.
        typer.echo(
            json.dumps(
                {
                    "run_plan": plan.model_dump(mode="json"),
                    "fit_classification": fit.model_dump(mode="json"),
                },
                indent=2,
            )
        )
        return
    typer.echo(f"predicted fit: {fit.classification.value} — {fit.rationale}", err=True)
    typer.echo(plan.model_dump_json(indent=2))


@app.command("platform-backends")
def platform_backends(
    json_out: bool = typer.Option(False, "--json", help="Emit the full BackendManifests as JSON."),
):
    """List the registered training backends — the frameworks you can pick to train on (corpus_studio,
    unsloth, …). Each declares what it can run; the planner resolves a plan against the chosen
    backend's declared support intersected with what actually proved to work on the host."""
    from corpus_studio.platform.backends import builtin_backends

    backends = builtin_backends()
    if json_out:
        typer.echo(json.dumps([b.model_dump(mode="json") for b in backends], indent=2))
        return
    for backend in backends:
        typer.echo(f"{backend.backend_id}  —  {backend.display_name}")
        typer.echo(
            f"    devices: {', '.join(d.value for d in backend.supported_devices)}"
            f"  |  precision: {', '.join(p.value for p in backend.precision_modes)}"
            f"  |  quant: {', '.join(q.value for q in backend.quantization_modes)}"
        )
        typer.echo(
            f"    adapters: {', '.join(a.value for a in backend.adapter_methods)}"
            f"  |  attention: {', '.join(a.value for a in backend.attention_impls)}"
        )


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


@app.command("run-provenance")
def run_provenance(project_dir: Path, config_path: Path):
    """Build a training run's reproducibility manifest: the dataset fingerprint +
    row count, the config SHA-256, and the engine version / platform. Best-effort —
    a missing dataset/config leaves that field null rather than failing."""
    from corpus_studio.training.provenance import build_run_provenance

    provenance = build_run_provenance(project_dir, config_path)
    typer.echo(provenance.model_dump_json(indent=2))


@app.command("import-convert")
def import_convert(path: Path, output_path: Path):
    """Convert a CSV/TSV/Parquet file to a JSONL staging file for import.

    Routed by extension: ``.parquet`` uses the typed columnar reader (values keep
    their type); ``.csv``/``.tsv``/other use the tabular reader (the header defines
    the keys; every cell is text). Either way this only reshapes the source ->
    JSONL — schema validation is the import-preview's job, so a value that violates
    the target schema quarantines the same as any JSONL/Hugging Face import.
    Parquet needs the optional ``[parquet]`` extra (a clear message if it's missing).
    """
    from corpus_studio.parquet_support import ParquetSupportError

    try:
        if path.suffix.lower() == ".parquet":
            from corpus_studio.importers.parquet_importer import convert_parquet_to_jsonl

            parquet_result = convert_parquet_to_jsonl(path, output_path)
            columns = parquet_result.columns
            rows_converted = parquet_result.rows_converted
            resolved_output = parquet_result.output_path
        else:
            result = convert_tabular_to_jsonl(path, output_path)
            columns = result.columns
            rows_converted = result.rows_converted
            resolved_output = result.output_path
    except ParquetSupportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, UnicodeError) as exc:
        typer.echo(f"Could not convert '{path}': {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "output_path": resolved_output,
                "rows_converted": rows_converted,
                "columns": columns,
            },
            indent=2,
        )
    )


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

    The staging file is UNTRUSTED until preview: rows are fetched from a third party
    and written without semantic validation, then vetted (schema/quality/PII) at the
    preview/quarantine step before any commit. --limit is capped (the page is buffered
    in memory) and nested column values are flattened to JSON strings so a staging row
    is always a flat, inspectable object.
    """
    # The engine must never write the dataset's single source of truth. Compare the basename
    # case-insensitively (casefold, NOT os.path.normcase — which only case-folds on Windows, so
    # it was a no-op on the Linux CI) so `--out Examples.jsonl` cannot slip past the guard on a
    # case-insensitive filesystem (Windows/macOS). Refusing the name on a case-sensitive OS too
    # is safe and conservative.
    if out.name.casefold() == "examples.jsonl":
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
    if limit > MAX_IMPORT_ROWS:
        typer.echo(
            f"--limit {limit} exceeds the import cap of {MAX_IMPORT_ROWS} rows; the staging file "
            "is buffered in memory. Import in smaller batches.",
            err=True,
        )
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
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        help="Trace-aware: strip the model's <think>…</think> and score only the ANSWER (so a "
        "reasoning model's thinking doesn't corrupt the score). Flags answers with no reasoning.",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Evaluator model that scores each answer 0-100 (metric=llm_judge). "
        "Omit to use the offline keyword-overlap score.",
    ),
    judge_backend: str = typer.Option("ollama", "--judge-backend", help="Judge backend."),
    judge_base_url: Optional[str] = typer.Option(None, "--judge-base-url", help="Judge provider base URL."),
    judge_api_key: Optional[str] = typer.Option(None, "--judge-api-key", help="Judge API key."),
    progress: bool = typer.Option(
        False,
        "--progress",
        help="Stream per-example progress ('[k/N] evaluated') to stderr during a long run. "
        "The report JSON still prints to stdout unchanged.",
    ),
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

    def _emit_progress(completed: int, total: int) -> None:
        # Progress goes to stderr so stdout stays the pure report JSON (mirrors suite-run's
        # "Running N case(s)" note). Throttled to ~100 updates so a large run doesn't flood
        # stderr with one line per example (a raising sink is swallowed by run_evaluation).
        if should_report_progress(completed, total):
            typer.echo(f"[{completed}/{total}] evaluated", err=True)

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
            reasoning=reasoning,
        ),
        examples,
        backend_client,
        limit=limit,
        scorer=scorer,
        progress_callback=_emit_progress if progress else None,
    )
    payload = report.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")

    typer.echo(payload)


def _evaluate_suite_case(case: "SuiteCase", project_dir: "Optional[Path]" = None) -> "EvaluationReport":
    """Run one suite case. A path case evaluates its dataset_path directly; a version-pinned
    case reconstructs + VERIFIES that dataset version to a temp file first (needs project_dir),
    then evaluates that. Raises on any failure; the suite runner isolates it into ERROR."""

    if case.version_id:
        if project_dir is None:
            raise ValueError(f"Case '{case.name}' pins a version but no --project-dir was given.")
        from corpus_studio.versions.version_restore import reconstruct_version_lines

        lines = reconstruct_version_lines(project_dir, case.version_id)  # raises on missing/verify-fail
        handle, tmp_name = tempfile.mkstemp(suffix=".jsonl", prefix=f"suite-{case.name}-")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write("\n".join(lines) + "\n")
            # Judge overrides come from the suite's PROJECT, not the temp reconstruction dir.
            return _evaluate_suite_dataset(case, tmp_path, project_dir)
        finally:
            tmp_path.unlink(missing_ok=True)

    dataset_path = Path(case.dataset_path or "")
    # Provider-policy overrides live in the suite's project; fall back to the dataset's dir
    # only when the suite was run by a bare file path with no --project-dir.
    overrides_dir = project_dir if project_dir is not None else dataset_path.parent
    return _evaluate_suite_dataset(case, dataset_path, overrides_dir)


def _evaluate_suite_dataset(case: "SuiteCase", dataset_path: Path, overrides_dir: Path) -> "EvaluationReport":
    """Evaluate a case against a concrete dataset file — the SAME policy-enforced eval path as
    ``eval-run`` (backend + optional evaluator-only judge scorer). ``overrides_dir`` is where
    project-local provider-policy overrides are read from (the suite's project)."""

    validation = validate_jsonl_file(dataset_path, case.schema_id)
    if not validation.valid:
        raise ValueError(f"Case '{case.name}': dataset failed {case.schema_id} validation.")

    rows = list(read_jsonl(dataset_path))
    examples = extract_evaluation_examples(rows, case.schema_id)
    backend_client = _build_backend(
        backend=case.backend,
        model=case.model,
        base_url=case.base_url,
        api_key=None,
        timeout_seconds=120,
    )

    scorer = None
    if case.metric == "llm_judge":
        judge_provider = infer_provider_id(case.judge_backend, case.judge_base_url)
        judge_route = case.judge_model if judge_provider == "openrouter" else None
        judge_policy = resolve_policy(
            judge_provider,
            model_id=case.judge_model,
            route_id=judge_route,
            overrides=load_overrides(overrides_dir),
        )
        judge_client = _build_backend(
            backend=case.judge_backend,
            model=case.judge_model or "",
            base_url=case.judge_base_url,
            api_key=None,
            timeout_seconds=120,
        )
        scorer = LlmJudgeScorer(judge_client, case.judge_model or "", policy=judge_policy)

    return run_evaluation(
        EvaluationRunConfig(
            dataset=dataset_path.stem,
            model=case.model,
            schema_id=case.schema_id,
            dataset_path=str(dataset_path),
            backend=case.backend,
            base_url=case.base_url,
            limit=case.limit,
            score_threshold=case.min_score if case.min_score is not None else 70.0,
            timeout_seconds=120,
        ),
        examples,
        backend_client,
        limit=case.limit,
        scorer=scorer,
    )


@app.command("suite-init")
def suite_init(
    name: str,
    project_dir: Path = typer.Option(Path("."), "--project-dir", help="Project directory (registry is evaluation_suites/)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing suite of this name."),
):
    """Scaffold an example evaluation suite at evaluation_suites/<name>.json for you to edit."""

    from corpus_studio.suites.registry import scaffold_suite

    try:
        path = scaffold_suite(project_dir, name, force=force)
    except (ValueError, FileExistsError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Created suite '{name}' at {path}. Edit its cases, then run: suite-run {name} --project-dir <dir>")


@app.command("suite-list")
def suite_list(
    project_dir: Path = typer.Option(Path("."), "--project-dir", help="Project directory (registry is evaluation_suites/)."),
    as_json: bool = typer.Option(False, "--json", help="Emit the suite list as JSON."),
):
    """List the evaluation suites registered under evaluation_suites/."""

    from corpus_studio.suites.registry import list_suite_definitions

    summaries = list_suite_definitions(project_dir)
    if as_json:
        typer.echo(json.dumps([summary.model_dump() for summary in summaries], indent=2))
        return
    if not summaries:
        typer.echo("No suites defined. Create one with: suite-init <name>")
        return
    for summary in summaries:
        if summary.valid:
            typer.echo(f"{summary.name} — {summary.case_count} case(s)")
        else:
            typer.echo(f"{summary.name} — invalid: {summary.error}")


@app.command("suite-run")
def suite_run(
    suite: str,
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", help="Registry + report dir (evaluation_suites/, suite_reports/)."),
    strict: bool = typer.Option(False, "--strict", help="Exit 2 when the suite verdict is block (CI/release gating)."),
):
    """Run an evaluation suite by file path OR by registered name. If SUITE is an existing file
    it is loaded as a path; otherwise it is a registered suite name resolved under
    evaluation_suites/ (requires --project-dir). Each case runs the existing eval + evaluation
    gate and the report rolls up PER METRIC. Advisory by default; --strict exits 2 on a block.
    Each case is a LIVE backend evaluation."""

    from corpus_studio.gates.models import GateStatus
    from corpus_studio.suites.registry import load_suite_by_name
    from corpus_studio.suites.runner import (
        append_suite_history,
        load_suite_definition,
        run_suite,
        save_suite_report,
    )

    suite_file = Path(suite)
    try:
        if suite_file.is_file():
            definition = load_suite_definition(suite_file)
        elif project_dir is None:
            typer.echo(
                f"'{suite}' is not a file. Pass --project-dir to run a registered suite by name, "
                "or give a suite file path.",
                err=True,
            )
            raise typer.Exit(code=1)
        else:
            definition = load_suite_by_name(project_dir, suite)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        typer.echo(f"Cannot load suite: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Running {len(definition.cases)} case(s) — each is a live backend evaluation.",
        err=True,
    )
    report = run_suite(
        definition,
        lambda case: _evaluate_suite_case(case, project_dir),
        generated_at=_utc_now_iso(),
    )

    if project_dir is not None:
        save_suite_report(project_dir, report)
        append_suite_history(project_dir, report)

    typer.echo(report.model_dump_json(indent=2))

    if strict and report.overall_status == GateStatus.BLOCK:
        raise typer.Exit(code=2)


@app.command("suite-history")
def suite_history(
    name: str,
    project_dir: Path = typer.Option(..., "--project-dir", help="Project holding suite_reports/history/."),
    as_json: bool = typer.Option(False, "--json", help="Emit the history as JSON."),
):
    """Show a registered suite's run history (oldest → newest) for trending pass/warn/block over time.

    Each point is the run time, the aggregate verdict, and per-status case counts — a count, never a
    folded quality score.
    """
    from corpus_studio.suites.runner import load_suite_history

    entries = load_suite_history(project_dir, name)
    if as_json:
        typer.echo(json.dumps([entry.model_dump() for entry in entries], indent=2))
        return

    if not entries:
        typer.echo(f"No run history for suite '{name}' yet.")
        return
    typer.echo(f"Suite '{name}' — {len(entries)} run(s):")
    for entry in entries:
        typer.echo(
            f"  {entry.generated_at or '?'}  {entry.overall_status.value.upper():5}  "
            f"{entry.passed}/{entry.total} passed"
            + (f", {entry.blocked} blocked" if entry.blocked else "")
            + (f", {entry.errored} errored" if entry.errored else "")
        )


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
    seed: int = typer.Option(
        42,
        "--seed",
        help="Training seed emitted into the config for reproducible weight init / shuffling. "
        "A fixed default keeps runs reproducible; the run's provenance manifest pins it via the config hash.",
    ),
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
            seed=seed,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    config_text = render_training_config(template)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config_text, encoding="utf-8")

    # Use the target model's own tokenizer for the budget when it's available (optional
    # `tokenizers` extra); otherwise this falls back to tiktoken / the heuristic, and the
    # budget's `method` field reports which ran so the count is never overstated as exact.
    token_budget = build_training_token_budget(
        list(read_jsonl(input_path)), sequence_len, model_id=base_model
    )
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
    # Higher math-attention-path estimate — used by the pre-flight OOM check on a Blackwell GPU, which
    # is forced onto the math attention path (the fused kernels deadlock there).
    vram_estimate_math = build_vram_estimate(
        base_model,
        lora_r=lora_r,
        sequence_len=sequence_len,
        micro_batch_size=micro_batch_size,
        math_attention=True,
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

    # Pre-flight: cheap fail-fast checks so a bad run is caught here, not hours in.
    from corpus_studio.training.preflight import run_training_preflight

    data_paths = [input_path]
    if eval_dataset_path is not None:
        data_paths.append(eval_dataset_path)
    preflight = run_training_preflight(
        config_path=output_path,
        launch_argv=launch_plan.argv,
        dependencies=launch_plan.dependencies,
        data_paths=data_paths,
        dataset_row_count=token_budget.example_count,
        examples_over_sequence_len=token_budget.examples_over_sequence_len,
        sequence_len=sequence_len,
        vram_min_gb=vram_estimate.total_gb_int4,
        vram_min_gb_math=vram_estimate_math.total_gb_int4,
    )

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
                "preflight": preflight.model_dump(),
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
    """Update an artifact's keep/reject status.

    A transition to ``kept`` is PROMOTE-GATED in the engine (integrity + source-run
    regression), so no caller — CLI, desktop, or script — can promote a modified/missing or
    regressed artifact by bypassing the UI. ``candidate``/``rejected`` are ungated.
    """

    from corpus_studio.training.artifact_registry import update_artifact_status

    if status == "kept":
        from corpus_studio.gates.models import GateStatus, load_gate_thresholds
        from corpus_studio.gates.runner import PromoteBlockedError, promote_artifact

        try:
            record, _report = promote_artifact(
                project_dir,
                artifact_id,
                now=_utc_now_iso(),
                thresholds=load_gate_thresholds(project_dir),
            )
        except PromoteBlockedError as exc:
            typer.echo("Promote gate blocked keeping this artifact:", err=True)
            for result in exc.report.results:
                if result.status == GateStatus.BLOCK:
                    typer.echo(f"  [{result.name}] {result.message}", err=True)
            raise typer.Exit(code=2) from exc
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        typer.echo(record.model_dump_json(indent=2))
        return

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
        artifact_content_integrity,
        artifact_path,
        load_artifact_record,
    )
    from corpus_studio.training.run_registry import load_run_record, record_path

    path = artifact_path(project_dir, artifact_id)
    if not path.exists():
        raise FileNotFoundError(f"No artifact '{artifact_id}'.")
    artifact = load_artifact_record(path)
    # Byte-exact integrity here: the weight card and the promote gate are the decision points,
    # so they read the weight bytes. The artifact LIST stays on the cheap size+mtime check.
    integrity = artifact_content_integrity(artifact)

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


@app.command("training-eval-plan")
def training_eval_plan(
    project_dir: Path,
    run_id: str = typer.Option(..., "--run-id", help="Finished run to evaluate."),
    eval_dataset: Optional[str] = typer.Option(
        None, "--eval-dataset", help="Held-out set to evaluate (defaults to the baseline's)."
    ),
    schema: Optional[str] = typer.Option(None, "--schema", help="Schema id (defaults to the baseline's)."),
    backend: str = typer.Option("ollama", "--backend", help="ollama or openai-compatible."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Serving endpoint (for openai-compatible)."),
    served_model: Optional[str] = typer.Option(
        None, "--served-model", help="Name the trained model is served under."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the plan as JSON."),
):
    """Close the train→eval loop: print the ordered steps to evaluate a finished run's model.

    The serve step is external (Ollama/vLLM/TGI); the eval/link/gate commands are exact. When
    ``--eval-dataset`` / ``--schema`` are omitted they are pre-filled from the run's baseline
    (before-eval) report so the after-eval compares like with like.
    """

    from corpus_studio.evaluation.reports import EvaluationReport
    from corpus_studio.training.eval_handoff import build_eval_handoff
    from corpus_studio.training.run_registry import load_run_record, record_path

    path = record_path(project_dir, run_id)
    if not path.exists():
        typer.echo(f"No run record for '{run_id}'.", err=True)
        raise typer.Exit(code=1)

    record = load_run_record(path)

    # Best-effort: pre-fill the held-out set + schema from the baseline eval so the
    # after-eval is comparable. A missing/unreadable baseline just leaves placeholders.
    dataset_default = eval_dataset
    schema_default = schema
    if (dataset_default is None or schema_default is None) and record.before_eval_path:
        try:
            before = EvaluationReport.model_validate_json(
                Path(record.before_eval_path).read_text(encoding="utf-8")
            )
        except (ValidationError, json.JSONDecodeError, OSError):
            before = None
        if before is not None and before.run_settings is not None:
            if dataset_default is None:
                dataset_default = before.run_settings.dataset_path or None
            if schema_default is None:
                schema_default = before.run_settings.schema_id or None

    plan = build_eval_handoff(
        record,
        project_dir=str(project_dir),
        eval_dataset_path=dataset_default or "",
        schema_id=schema_default or "",
        backend=backend,
        base_url=base_url,
        served_model=served_model or "",
    )

    if as_json:
        typer.echo(plan.model_dump_json(indent=2))
        return

    typer.echo(f"Evaluate the model from run {plan.run_id} (status: {plan.status})")
    if plan.note:
        typer.echo(plan.note)
    if not plan.ready:
        return
    for index, step in enumerate(plan.steps, start=1):
        typer.echo("")
        typer.echo(f"{index}. {step.title}")
        typer.echo(f"   {step.detail}")
        if step.command:
            typer.echo(f"   $ {step.command}")


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


@app.command("train-check")
def train_check(
    json_output: bool = typer.Option(False, "--json", help="Emit the machine-readable report instead of the table."),
):
    """Preflight the FIRST-PARTY training runtime (the opt-in [train] extra): which deps are present,
    whether a CUDA GPU is available, and whether a real 4-bit QLoRA run — or only the CPU toy path —
    is possible. Reads only the Python environment (no project needed); safe with none of the training
    deps installed. Exit stays 0 (the verdict is in the report)."""

    from corpus_studio.training.environment import (
        probe_training_runtime,
        render_training_runtime_text,
    )

    report = probe_training_runtime()
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(render_training_runtime_text(report))


@app.command("dataset-tokens")
def dataset_tokens(
    dataset_path: Path,
    base_model: str = typer.Option(..., "--base-model", help="Tokenizer to measure with (the model you'll train)."),
    dataset_format: str = typer.Option("chat", "--dataset-format", help="Row format: instruction | chat."),
    seq_len: int = typer.Option(4096, "--seq-len", help="The sequence_len to check truncation against."),
    sample: int = typer.Option(0, "--sample", help="Tokenize only the first N rows (0 = all)."),
    json_out: bool = typer.Option(False, "--json", help="Emit the TruncationReport as JSON."),
):
    """Measure a dataset's token-length distribution and how many examples a given sequence_len would
    TRUNCATE — the guardrail to run BEFORE training so you never silently train on cut-off outputs
    (the failure that taught the WBG model to emit incomplete JSON). Exit 3 when it truncates."""
    from corpus_studio.importers.jsonl_importer import read_jsonl
    from corpus_studio.training.trainer import (
        analyze_truncation,
        format_example_text,
        truncation_warning,
    )

    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 - the [train] extra (heavy).
    except ImportError:
        typer.echo("dataset-tokens needs the [train] extra (transformers) — install it first.", err=True)
        raise typer.Exit(code=2) from None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    rows = list(read_jsonl(dataset_path))
    if sample > 0:
        rows = rows[:sample]
    lengths: list[int] = []
    for row in rows:
        text = format_example_text(row, dataset_format, tokenizer)
        if text:
            lengths.append(len(tokenizer(text)["input_ids"]))
    report = analyze_truncation(lengths, seq_len)
    if json_out:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(
            f"{report.n_examples} examples | median {report.median_tokens} | max {report.max_tokens} tokens"
        )
        warning = truncation_warning(report)
        typer.echo(
            warning
            if warning
            else f"OK: sequence_len={seq_len} keeps every example whole (max {report.max_tokens} tokens)."
        )
    if report.truncates:
        raise typer.Exit(code=3)


@app.command("trace-validate")
def trace_validate(
    dataset_path: Path,
    json_out: bool = typer.Option(False, "--json", help="Emit the validation summary as JSON."),
    show: int = typer.Option(5, "--show", help="Show up to N invalid rows."),
):
    """Validate a reasoning-TRACE corpus: each row must parse to a Trace (prompt/context + thinking +
    answer), pass the STRUCTURAL checks (answer present, reasoning not a verbatim copy), AND the
    reasoning-QUALITY gate (no answer leaked into the prompt, no stray <think> tags, reasoning not
    token-thin). Reports how many rows carry a real thinking trace + the quality distribution. Exit 3
    if any row is structurally invalid OR a quality FAIL — the guardrail before training a reasoner."""
    from corpus_studio.importers.jsonl_importer import read_jsonl
    from corpus_studio.training.traces import trace_from_row, trace_quality, validate_trace

    rows = list(read_jsonl(dataset_path))
    total = len(rows)
    with_thinking = 0
    quality = {"pass": 0, "warn": 0, "fail": 0}
    invalid: list[tuple[int, list[str]]] = []
    for index, row in enumerate(rows):
        trace = trace_from_row(row)
        verdict = validate_trace(trace)
        gate = trace_quality(trace)
        with_thinking += int(verdict.has_thinking)
        quality[gate.status] += 1
        problems = list(verdict.errors) + ([f"quality:{i}" for i in gate.issues] if gate.status == "fail" else [])
        if not verdict.valid or gate.status == "fail":
            invalid.append((index, problems))
    pct_think = round(100 * with_thinking / total) if total else 0
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "total": total,
                    "with_thinking": with_thinking,
                    "quality": quality,
                    "blocked": len(invalid),
                    "issues": [{"row": i, "problems": e} for i, e in invalid[:50]],
                },
                indent=2,
            )
        )
    else:
        typer.echo(
            f"{total} traces | {with_thinking} with thinking ({pct_think}%) | "
            f"quality pass={quality['pass']} warn={quality['warn']} fail={quality['fail']} | "
            f"{len(invalid)} blocked"
        )
        for index, problems in invalid[:show]:
            typer.echo(f"  row {index}: {', '.join(problems)}")
    if invalid:
        raise typer.Exit(code=3)


@app.command("trace-generate")
def trace_generate(
    prompts_path: Path,
    out: Path = typer.Option(..., "--out", help="Write the accepted traces JSONL here (dataset_format=trace)."),
    backend: str = typer.Option("ollama", "--backend", help="Model backend: ollama | openai-compatible."),
    model: str = typer.Option(..., "--model", help="The model that GENERATES the reasoning — your own model (self-distillation) or a stronger teacher."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Backend base URL (default: the provider's local default)."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for an openai-compatible teacher."),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens per generation (reasoning + answer)."),
    temperature: float = typer.Option(0.7, "--temperature"),
    system: Optional[str] = typer.Option(None, "--system", help="Override the reasoning system prompt."),
    timeout_seconds: int = typer.Option(180, "--timeout-seconds"),
    limit: int = typer.Option(0, "--limit", help="Only the first N prompts (0 = all)."),
):
    """GENERATE reasoning traces from a prompt corpus via a model backend — self-distill (point --model
    at your own model) or a teacher (a bigger local model / an API model). Each output is
    prompt + <think>reasoning</think> + answer; the pipeline SELF-FILTERS — only traces with real
    reasoning that pass the quality gate are kept. Writes the accepted traces (train with
    --dataset-format trace). A per-prompt backend error rejects that prompt, never aborts the batch."""
    from corpus_studio.importers.jsonl_importer import read_jsonl
    from corpus_studio.training.trace_generation import (
        DEFAULT_REASONING_SYSTEM,
        backend_generate_fn,
        generate_traces,
        prompt_from_row,
    )

    try:
        client = _build_backend(backend, model, base_url, api_key, timeout_seconds)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    rows = list(read_jsonl(prompts_path))
    if limit > 0:
        rows = rows[:limit]
    prompts = [p for p in (prompt_from_row(row) for row in rows) if p]
    if not prompts:
        typer.echo("No usable prompts found (expected a prompt/instruction/question field or messages).", err=True)
        raise typer.Exit(code=2)

    generate_fn = backend_generate_fn(client, max_tokens=max_tokens, temperature=temperature)
    results = generate_traces(prompts, generate_fn, system=system or DEFAULT_REASONING_SYSTEM)
    accepted = [r for r in results if r.accepted and r.trace is not None]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for result in accepted:
            trace = result.trace
            assert trace is not None
            handle.write(
                json.dumps({"prompt": trace.prompt, "thinking": trace.thinking, "answer": trace.answer})
                + "\n"
            )
    typer.echo(
        f"prompts {len(prompts)} | accepted {len(accepted)} | rejected {len(results) - len(accepted)} "
        f"-> {out}",
        err=True,
    )
    typer.echo(str(out))


@app.command("train-run")
def train_run(
    config_path: Path,
    dataset_path: Optional[Path] = typer.Option(None, "--dataset-path", help="Override the config's dataset_path (e.g. the train split)."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Override where the adapter/checkpoints are written."),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Override the base model."),
    cpu_toy: bool = typer.Option(False, "--cpu-toy", help="Run the tiny CPU smoke path (a small model, a few steps, no GPU/bitsandbytes)."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Cap the number of training steps."),
    attn_implementation: Optional[str] = typer.Option(None, "--attn-implementation", help="Attention backend (eager | sdpa | flash_attention_2). Default auto: on Blackwell/sm_120 the fused flash/mem-efficient SDPA is disabled (it deadlocks the first backward there) and the math SDPA path is used."),
    optim: Optional[str] = typer.Option(None, "--optim", help="Optimizer (e.g. adamw_torch | paged_adamw_8bit). 'paged_adamw_8bit' pages optimizer state to host RAM under pressure — spike-safe on a tight GPU."),
    use_liger: bool = typer.Option(False, "--use-liger", help="Fuse the cross-entropy loss (Liger) to drop the full-vocab logits memory spike at long sequence_len. Needs the 'liger-kernel' package; Blackwell support unverified."),
    memory_efficient: bool = typer.Option(False, "--memory-efficient", help="Shortcut for a tight GPU: enable the memory-saving levers (paged optimizer + fused Liger loss). Explicit --optim / --use-liger override it."),
    save_steps: Optional[int] = typer.Option(None, "--save-steps", help="Checkpoint every N optimizer steps (default 50)."),
    save_total_limit: Optional[int] = typer.Option(None, "--save-total-limit", help="Keep only the N most recent checkpoints (default 3; pass 0 to keep ALL — old behavior)."),
):
    """RUN the training in-process (first-party trainer, opt-in [train] extra): read a CorpusStudio
    training config + dataset, build a TRL SFTTrainer with peft LoRA (4-bit QLoRA on GPU), train, and
    save the adapter + tokenizer + checkpoints. Preflighted by train-check — refuses with an install
    hint if the runtime is missing. Progress ('[step/total]') goes to stderr; the JSON result to stdout.

    A real GPU QLoRA cannot be run without a CUDA GPU + bitsandbytes; --cpu-toy proves the pipeline on
    CPU. Exit 2 when the runtime can't run the request."""

    from corpus_studio.training.trainer import (
        TrainerError,
        load_run_config_from_file,
        run_training,
    )

    # --memory-efficient is a shortcut; explicit --optim / --use-liger win. None = fall back to the
    # config's value (a bool flag can't distinguish "unset" from "false", so pass None when unset).
    resolved_optim = optim or ("paged_adamw_8bit" if memory_efficient else None)
    resolved_liger = True if (use_liger or memory_efficient) else None
    try:
        run_config = load_run_config_from_file(
            config_path,
            dataset_path=str(dataset_path) if dataset_path else None,
            output_dir=str(output_dir) if output_dir else None,
            base_model=base_model,
            cpu_toy=cpu_toy,
            max_steps=max_steps,
            attn_implementation=attn_implementation,
            optim=resolved_optim,
            use_liger=resolved_liger,
        )
    except (TrainerError, ValueError, json.JSONDecodeError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    # CLI overrides for checkpoint retention (None = leave the config/default). `--save-total-limit 0`
    # is the escape hatch for "keep ALL checkpoints" (maps to None, since the field requires >= 1).
    retention_updates: dict[str, object] = {}
    if save_steps is not None:
        retention_updates["save_steps"] = save_steps
    if save_total_limit is not None:
        retention_updates["save_total_limit"] = None if save_total_limit == 0 else save_total_limit
    if retention_updates:
        run_config = run_config.model_copy(update=retention_updates)

    def _progress(step: int, total: int, loss: Optional[float]) -> None:
        line = f"[{step}/{total}] step" + (f" loss={loss:.4f}" if loss is not None else "")
        typer.echo(line, err=True)

    try:
        result = run_training(run_config, progress_callback=_progress)
    except TrainerError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(result.model_dump_json(indent=2))


@app.command("train-merge")
def train_merge(
    adapter_path: Path,
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base to merge into (default: the adapter's recorded base)."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Where to write the merged model (default: <adapter>/../merged)."),
    strategy: str = typer.Option("auto", "--strategy", help="auto | gpu | cpu | adapter-only. auto = gpu→cpu→adapter-only."),
):
    """Merge a trained LoRA adapter into its base model, with a fallback for small-VRAM cards: a 7B
    fp16 merge (~14 GB) won't fit a 12 GB GPU, so `auto` tries GPU → CPU-offload → adapter-only (serve
    base+adapter unmerged). Progress → stderr; the JSON result → stdout. Exit 2 if every strategy fails."""

    from corpus_studio.training.merge import MergeError, merge_adapter

    def _progress(message: str) -> None:
        typer.echo(message, err=True)

    try:
        result = merge_adapter(
            adapter_path,
            base_model=base_model,
            output_dir=str(output_dir) if output_dir else None,
            strategy=strategy,
            progress=_progress,
        )
    except MergeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(result.model_dump_json(indent=2))


@app.command("model-card")
def model_card(
    adapter_path: Path,
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Override the base model recorded in the adapter."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="A CorpusStudio training config JSON to fold in (format / seq-len / lr / seed)."),
    output: Optional[Path] = typer.Option(None, "--output", help="Write the card here (default: print to stdout). train-run already writes <adapter>/MODEL_CARD.md."),
):
    """Render a Markdown model card for a trained LoRA adapter: the base model (+ the reminder that ITS
    license governs the result), the LoRA hyper-parameters (read from the adapter's adapter_config.json),
    the training settings, and honesty notes. train-run already writes MODEL_CARD.md next to the adapter;
    this regenerates it (e.g. with a training config or a base-model override). Reads only local files."""

    from corpus_studio.training.model_card import build_model_card

    training_config = None
    if config_path is not None:
        try:
            training_config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError) as exc:
            typer.echo(f"Could not read the training config {config_path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    card = build_model_card(
        adapter_path,
        base_model=base_model,
        training_config=training_config,
        generated_at=_utc_now_iso(),
    )

    if output is not None:
        try:
            Path(output).write_text(card, encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Could not write the model card to {output}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(str(output))
    else:
        typer.echo(card)


@app.command("model-fetch")
def model_fetch(
    repo_id: str,
    local_dir: Optional[Path] = typer.Option(None, "--local-dir", help="Download here (default: the HF cache, so `train-run --base-model <repo>` finds it offline)."),
    revision: Optional[str] = typer.Option(None, "--revision", help="Branch / tag / commit to fetch."),
    allow: Optional[list[str]] = typer.Option(None, "--allow", help="Restrict to these glob(s), e.g. --allow '*.safetensors' (the model card + *.json are always kept so it loads and its license is readable). Repeatable."),
):
    """Reliably download a base model from the Hugging Face Hub — RESUMABLE, so it survives dropped
    connections — and report its LICENSE. Prefer MIT/Apache/permissive base models: the base model's
    license governs what you can do with the trained result (data availability ≠ permission). JSON
    result → stdout, progress → stderr; exit 2 if the fetch fails or the training runtime is missing."""

    from corpus_studio.training.model_fetch import fetch_model

    def _progress(message: str) -> None:
        typer.echo(message, err=True)

    try:
        result = fetch_model(
            repo_id,
            local_dir=str(local_dir) if local_dir else None,
            revision=revision,
            allow_patterns=list(allow) if allow else None,
            progress=_progress,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if not result.license_permissive:
        typer.echo(
            f"⚠ License: {result.license or 'none declared'} — not clearly permissive; review before training.",
            err=True,
        )
    typer.echo(result.model_dump_json(indent=2))


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

    # Enforce the export gate BEFORE writing the deliverable, exactly as `export` does:
    # block on high-severity PII/secrets (and empty input / schema). A private key in a
    # preference dataset must not be exported. Quality issues only warn.
    from corpus_studio.gates.models import GateStatus

    export_gate = run_export_gates(rows, "preference")
    if export_gate.overall_status == GateStatus.BLOCK:
        typer.echo("Export blocked by the export gate:", err=True)
        for gate_result in export_gate.results:
            if gate_result.status == GateStatus.BLOCK:
                typer.echo(f"  [{gate_result.name}] {gate_result.message}", err=True)
        raise typer.Exit(code=2)

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
    redact_pii: bool = typer.Option(
        False,
        "--redact-pii",
        help="Mask detected PII/secrets (emails, SSNs, keys/tokens, cards) in the EXPORT with typed "
        "[REDACTED:kind] placeholders. Masks known high-precision patterns only — NOT a guarantee of "
        "de-identification. Writes a redaction manifest; never rewrites the input.",
    ),
    check_provenance: bool = typer.Option(
        False,
        "--check-provenance",
        help="Run the per-row provenance gate before writing: BLOCK the export (exit 2) if any row's "
        "declared meta.teacher is a provider you can't train on (e.g. Anthropic/OpenAI). Loads the "
        "project's provenance_allowlist.json (next to the input) and any --allow-teacher entries.",
    ),
    provenance_strict: bool = typer.Option(
        False,
        "--provenance-strict",
        help="With --check-provenance, also BLOCK on rows whose provenance is unknown (default: warn).",
    ),
    allow_teacher: Optional[list[str]] = typer.Option(
        None,
        "--allow-teacher",
        help="Declare a teacher/provider trainable-clean for --check-provenance (e.g. z-ai/glm-5.2). Repeatable.",
    ),
    export_format: str = typer.Option(
        "jsonl",
        "--format",
        help="Output format: jsonl (default, model-ready, all schemas), csv/tsv, or parquet. CSV/TSV is a "
        "flat-schema convenience — a schema with chat messages or nested objects is refused (use jsonl/parquet); "
        "scalar list fields (e.g. tags) are written as a '; '-joined cell. Parquet is columnar and supports "
        "every schema (chat/nested included) but needs the optional [parquet] extra.",
    ),
):
    """Validate and export a JSONL file, optionally cleaning it first."""
    export_format = export_format.strip().lower()
    if export_format not in ("jsonl", "csv", "tsv", "parquet"):
        typer.echo(
            f"Unknown --format '{export_format}'; expected jsonl, csv, tsv, or parquet.", err=True
        )
        raise typer.Exit(code=1)

    # Parquet needs the optional pyarrow extra — fail fast with the install hint
    # before any validation/cleaning work so no partial output is written.
    if export_format == "parquet":
        from corpus_studio.parquet_support import PARQUET_INSTALL_HINT, parquet_available

        if not parquet_available():
            typer.echo(PARQUET_INSTALL_HINT, err=True)
            raise typer.Exit(code=1)

    report = validate_jsonl_file(input_path, schema)
    _exit_if_invalid(report)

    # Fail fast: a nested schema can't become flat columns. Check before any work so
    # the message is immediate and no partial output is written.
    tabular_schema = None
    if export_format in ("csv", "tsv"):
        tabular_schema = load_builtin_schema(schema)
        exportable, blocking = schema_is_csv_exportable(tabular_schema)
        if not exportable:
            typer.echo(
                f"Cannot export schema '{schema}' to {export_format}: field(s) {blocking} are nested "
                "(chat messages / objects / lists of objects) and can't become flat columns. "
                "Export as JSONL instead.",
                err=True,
            )
            raise typer.Exit(code=2)

    # Enforce the export gate BEFORE writing the deliverable: block on PII/secrets (and
    # empty input / schema). Quality issues (duplicates / low-information) only warn — the
    # optional cleaning pass handles those. This is what makes "export blocks on PII" true.
    from corpus_studio.gates.models import GateStatus

    rows = list(read_jsonl(input_path))

    # Opt-in redaction runs BEFORE the export gate so masking the known PII/secret patterns is what
    # lets a blocked-on-PII dataset export (with the secrets masked) rather than being refused. It is a
    # known-pattern safety net, not de-identification — the manifest records what was masked.
    redaction_result = None
    if redact_pii:
        rows, redaction_result = redact_rows(rows)

    export_gate = run_export_gates(rows, schema)
    if export_gate.overall_status == GateStatus.BLOCK:
        typer.echo("Export blocked by the export gate:", err=True)
        for gate_result in export_gate.results:
            if gate_result.status == GateStatus.BLOCK:
                typer.echo(f"  [{gate_result.name}] {gate_result.message}", err=True)
        raise typer.Exit(code=2)

    warnings: list[str] = []

    # Opt-in per-row provenance enforcement: refuse to write a training deliverable that contains
    # rows generated by a provider you can't train on (the licensing counterpart to the PII gate).
    # Runs on the same rows the deliverable is built from; BLOCK exits before any write.
    provenance_result = None
    if check_provenance:
        from corpus_studio.gates.provenance_gate import (
            load_provenance_allowlist,
            render_provenance_gate_text,
            run_provenance_gate,
        )

        provenance_allowlist = load_provenance_allowlist(input_path.parent)
        for teacher_entry in allow_teacher or []:
            teacher_name = teacher_entry.strip()
            if teacher_name:
                provenance_allowlist.setdefault(teacher_name, "allow-listed via --allow-teacher")

        provenance_result = run_provenance_gate(
            rows,
            allowlist=provenance_allowlist,
            strict=provenance_strict,
            target=str(input_path),
        )
        if provenance_result.overall_status == GateStatus.BLOCK:
            typer.echo("Export blocked by the provenance gate:", err=True)
            typer.echo(render_provenance_gate_text(provenance_result), err=True)
            raise typer.Exit(code=2)
        if provenance_result.overall_status == GateStatus.WARN:
            warnings.append(
                f"Provenance: {provenance_result.unknown_rows} row(s) have unknown provenance "
                "(quarantine-until-verified). Re-run with --provenance-strict to block them."
            )

    # csv/tsv/parquet share the entire validate/redact/gate/clean pipeline above; only
    # the writer differs. tabular_schema is set (and proven flat) for csv/tsv; parquet
    # is columnar and needs no flat-schema gate (it represents nested types natively).
    tabular_columns: list[str] | None = None
    delimiter = "\t" if export_format == "tsv" else ","
    is_parquet = export_format == "parquet"
    if is_parquet:
        from corpus_studio.exporters.parquet_exporter import write_parquet

    if dedupe or drop_low_information:
        kept, clean_result = clean_rows(
            rows,
            dedupe=dedupe,
            drop_low_information=drop_low_information,
        )
        if is_parquet:
            _, tabular_columns = write_parquet(kept, output_path)
        elif tabular_schema is not None:
            _, tabular_columns = write_tabular(kept, output_path, tabular_schema, delimiter)
        else:
            write_jsonl(kept, output_path)

        manifest_path = output_path.with_name(output_path.name + ".cleaning_manifest.json")
        manifest_path.write_text(
            clean_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        payload = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "format": export_format,
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
        quality = build_basic_quality_report(rows)
        if is_parquet:
            _, tabular_columns = write_parquet(rows, output_path)
        elif tabular_schema is not None:
            _, tabular_columns = write_tabular(rows, output_path, tabular_schema, delimiter)
        elif redaction_result is not None:
            # Redaction changed the rows, so the deliverable is the redacted rows — not an input copy.
            write_jsonl(rows, output_path)
        else:
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
            "format": export_format,
            "cleaned": False,
            "input_rows": quality.example_count,
            "output_rows": quality.example_count,
            "removed_rows": 0,
            "warnings": warnings,
        }

    if tabular_columns is not None:
        payload["columns"] = tabular_columns

    if provenance_result is not None:
        # The export only reaches here when the verdict was not BLOCK (else it exited above),
        # so this records that the deliverable passed the provenance check (PASS or WARN).
        payload["provenance_checked"] = True
        payload["provenance_status"] = provenance_result.overall_status.value
        payload["quarantined_rows"] = provenance_result.quarantined_rows
        payload["unknown_provenance_rows"] = provenance_result.unknown_rows

    if redaction_result is not None:
        redaction_manifest_path = output_path.with_name(
            output_path.name + ".redaction_manifest.json"
        )
        redaction_manifest_path.write_text(
            redaction_result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        payload["redacted"] = True
        payload["redacted_spans"] = redaction_result.redacted_spans
        payload["redacted_rows"] = redaction_result.redacted_rows
        payload["redaction_manifest_path"] = str(redaction_manifest_path)
        if redaction_result.redacted_spans:
            warnings.append(
                f"Redacted {redaction_result.redacted_spans} PII/secret span(s) across "
                f"{redaction_result.redacted_rows} row(s) with [REDACTED:kind] placeholders. Known "
                "high-precision patterns only — NOT a guarantee of de-identification; review before publishing."
            )

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

    # Resolve a policy per generation model (arena responses are non-trainable, so this
    # authorizes each as an EVALUATION participant) and gate the run — a provider blocked from
    # the evaluator role cannot generate arena responses.
    gen_provider = infer_provider_id(backend, base_url)
    gen_overrides = load_overrides(input_path.parent)
    gen_policies: dict[str, Any] = {
        model: resolve_policy(
            gen_provider,
            model_id=model,
            route_id=(model if gen_provider == "openrouter" else None),
            overrides=gen_overrides,
        )
        for model in unique_models
    }
    try:
        report = run_arena(
            prompts, model_backends, limit=limit, generated_at=_utc_now_iso(), policies=gen_policies
        )
    except ProviderPolicyError as exc:
        typer.echo(f"Provider policy blocked arena generation: {exc}", err=True)
        raise typer.Exit(code=2) from exc

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


@app.command("gate-thresholds-set")
def gate_thresholds_set(
    project_dir: Path,
    values_json: str = typer.Option(
        ...,
        "--values-json",
        help="GateThresholds as a JSON object. Validated (ranges/finite) before writing; an invalid "
        "value is rejected rather than written.",
    ),
):
    """Validate and write a project's ``gate_thresholds.json`` from a JSON payload.

    The whole object is validated through the ``GateThresholds`` model, so an out-of-range or
    non-finite value is refused (exit 1) instead of producing a broken threshold file.
    """
    from pydantic import ValidationError

    from corpus_studio.gates.models import GateThresholds, save_gate_thresholds

    try:
        data = json.loads(values_json)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object of threshold values.")
        thresholds = GateThresholds(**data)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        typer.echo(f"Invalid gate thresholds: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    path = save_gate_thresholds(project_dir, thresholds)
    typer.echo(json.dumps({"path": str(path), "thresholds": thresholds.model_dump()}, indent=2))


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


@app.command("chat-gate")
def chat_gate(
    input_path: Path,
    schema: str = typer.Option("chat", "--schema", help="Schema id for per-message validation."),
    project_dir: Optional[Path] = typer.Option(None, "--project-dir", help="Write report under gate_reports/ and apply project thresholds."),
):
    """Gate a chat dataset's conversation structure (chat_suite scope). Advisory: prints a
    pass/warn/block report over input presence, per-message schema, and conversation-sequence
    structure. Verdicts structure, not semantic quality; exit code stays 0 (the verdict is in
    the report)."""

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
        report = run_chat_gates(
            rows, schema, thresholds=thresholds, target=str(input_path), generated_at=_utc_now_iso()
        )
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if project_dir is not None:
        save_gate_report(project_dir, report)

    typer.echo(report.model_dump_json(indent=2))


@app.command("provenance-gate")
def provenance_gate(
    input_path: Path,
    teacher_field: str = typer.Option(
        "meta.teacher",
        "--teacher-field",
        help="Dotted path to each row's generating-model tag (default meta.teacher).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Also BLOCK on rows with unknown provenance (default: WARN on unknown).",
    ),
    allow_teacher: Optional[list[str]] = typer.Option(
        None,
        "--allow-teacher",
        help="Declare a teacher (or provider) trainable-clean, e.g. --allow-teacher z-ai/glm-5.2. Repeatable.",
    ),
    project_dir: Optional[Path] = typer.Option(
        None,
        "--project-dir",
        help="Load the project's provenance_allowlist.json (an --allow-teacher entry adds to it).",
    ),
):
    """Gate a dataset's PER-ROW provenance: read each row's teacher (meta.teacher) and
    quarantine rows generated by a restricted provider (e.g. Anthropic/OpenAI) whose terms
    forbid training on their outputs. The licensing counterpart to provider-policy (which
    gates generation-time) and run-provenance (which fingerprints a run). The verdict
    (BLOCK/WARN/PASS) is the JSON report on stdout; a human table + verdict go to stderr.

    Exit stays 0 — the verdict IS the report. Honest scope: it trusts each row's DECLARED
    teacher (a mislabeled/omitted teacher is not caught by content), and an unknown teacher
    is quarantine-until-verified, never assumed safe."""

    from corpus_studio.gates.provenance_gate import (
        load_provenance_allowlist,
        render_provenance_gate_text,
        run_provenance_gate,
    )

    allowlist: dict[str, str] = load_provenance_allowlist(project_dir) if project_dir else {}
    for entry in allow_teacher or []:
        name = entry.strip()
        if name:
            allowlist.setdefault(name, "allow-listed via --allow-teacher")

    try:
        rows = list(read_jsonl(input_path))
    except (ValueError, json.JSONDecodeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    report = run_provenance_gate(
        rows,
        teacher_field=teacher_field,
        allowlist=allowlist,
        strict=strict,
        target=str(input_path),
        generated_at=_utc_now_iso(),
    )
    typer.echo(render_provenance_gate_text(report), err=True)
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


@app.command("dataset-version-gc")
def dataset_version_gc(
    project_dir: Path,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be pruned without rewriting the row store."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
):
    """Prune row-store rows that no dataset version references.

    Safe by construction: the rows to keep are the union of every version manifest, and a row that
    can't be positively identified as unreferenced is kept. If any manifest is unreadable, GC aborts
    rather than risk deleting referenced rows.
    """
    from corpus_studio.versions.gc import gc_row_store

    try:
        result = gc_row_store(project_dir, dry_run=dry_run)
    except OSError as exc:
        typer.echo(f"GC aborted (a version manifest could not be read): {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return

    action = "Would prune" if dry_run else "Pruned"
    typer.echo(
        f"Row store: {result.scanned_rows} row(s), {result.referenced_row_ids} referenced. "
        f"{action} {result.pruned_rows}, kept {result.kept_rows}."
    )


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

    # The engine never writes the dataset — refuse to target examples.jsonl. Compare
    # robustly: resolve() follows symlinks; os.path.samefile catches a case-insensitive
    # or hard-linked match when the target exists; a normcase compare covers a not-yet-
    # existing output on a case-insensitive filesystem (Windows/macOS).
    examples_path = project_dir / "examples.jsonl"
    try:
        resolved_output = output.resolve()
        resolved_examples = examples_path.resolve()
        if resolved_output.exists() and resolved_examples.exists():
            targets_examples = os.path.samefile(resolved_output, resolved_examples)
        else:
            targets_examples = (
                os.path.normcase(str(resolved_output))
                == os.path.normcase(str(resolved_examples))
            )
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


def main() -> None:
    """Console-script + ``-m`` entry point. Forces UTF-8 stdio BEFORE the CLI runs so
    non-ASCII output — including the ``→`` arrows in ``--help`` — never dies on a Windows
    cp1252 console (``UnicodeEncodeError``). The ``corpus-studio`` script points here so it
    gets the same UTF-8 handling as ``python -m corpus_studio.cli`` (previously only the
    ``__main__`` path reconfigured, so the bare ``corpus-studio --help`` crashed)."""
    _ensure_utf8_stdio()
    app()


if __name__ == "__main__":
    main()
