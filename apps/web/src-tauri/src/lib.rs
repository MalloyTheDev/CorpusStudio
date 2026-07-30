//! Corpus Studio Tauri 2 shell. The window is a thin client over the Python platform engine: each
//! command shells out to the `corpus-studio platform-*` CLI (stdout = a JSON contract, stderr =
//! telemetry) and returns the parsed contract to the React frontend. The engine remains the single
//! source of truth; the shell never contains platform logic (see docs/platform-architecture-epic).

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};

use serde_json::Value;

/// Run a `corpus-studio` subcommand and parse its JSON stdout into a contract value.
fn run_engine(args: &[&str]) -> Result<Value, String> {
    let output = Command::new("corpus-studio")
        .args(args)
        .output()
        .map_err(|e| format!("could not launch the engine: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("the engine returned non-JSON output: {e}"))
}

/// Profile the host + prove its capabilities (EnvironmentProfile + CapabilityReport).
#[tauri::command]
fn platform_probe() -> Result<Value, String> {
    run_engine(&["platform-probe", "--json", "--cache"])
}

/// The registered training backends and their declared capabilities (BackendManifest[] JSON) — the
/// "pick your framework" registry. The engine owns it; the shell only lists it.
#[tauri::command]
fn platform_backends() -> Result<Value, String> {
    run_engine(&["platform-backends", "--json"])
}

/// Resolve a hash-sealed RunPlan for a base model + dataset on the chosen backend, plus its predicted
/// fit — the `{run_plan, fit_classification}` bundle the live flow renders (`--json`).
#[tauri::command]
fn platform_plan(
    base_model: String,
    model_revision: Option<String>,
    dataset: String,
    sequence_len: u32,
    backend: Option<String>,
) -> Result<Value, String> {
    let seq = sequence_len.to_string();
    let mut args = vec![
        "platform-plan",
        "--base-model",
        &base_model,
        "--dataset",
        &dataset,
        "--sequence-len",
        &seq,
        "--json",
    ];
    let revision = model_revision.unwrap_or_default();
    if !revision.trim().is_empty() {
        args.push("--model-revision");
        args.push(&revision);
    }
    let backend_id = backend.unwrap_or_default();
    if !backend_id.is_empty() {
        args.push("--backend");
        args.push(&backend_id);
    }
    run_engine(&args)
}

/// Execute a hash-sealed RunPlan through the headless run supervisor and STREAM its RunEvents to the
/// frontend as they happen — the plan->run half of the one-training-authority flow. The engine emits
/// RunEvents on stderr (one per line) and the terminal RunManifest on stdout; the shell never owns run
/// behaviour, it only forwards the sealed plan to `platform-run --subprocess` and relays the engine's
/// own event stream + manifest. `on_event` is a Tauri channel the frontend subscribes to for live events.
#[tauri::command]
fn platform_run(
    plan: Value,
    out_dir: String,
    on_event: tauri::ipc::Channel<Value>,
) -> Result<Value, String> {
    // The frontend holds the sealed RunPlan object (from `platform_plan`); persist it to a temp file the
    // engine can read + hash-verify. platform-run re-canonicalizes for the hash, so exact bytes are fine.
    let plan_text =
        serde_json::to_string(&plan).map_err(|e| format!("the RunPlan is not serializable: {e}"))?;
    let plan_path = write_temp_jsonl(&plan_text)?;
    let plan_arg = plan_path.to_string_lossy().to_string();

    // The plan's sealed export.output_dir governs run artifacts; --out additionally writes the terminal
    // manifest + enables telemetry. Omit it when the caller passes no directory (the manifest still
    // arrives on stdout).
    let mut args: Vec<&str> = vec!["platform-run", &plan_arg, "--subprocess"];
    if !out_dir.trim().is_empty() {
        args.push("--out");
        args.push(&out_dir);
    }
    let child = Command::new("corpus-studio")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => {
            let _ = std::fs::remove_file(&plan_path);
            return Err(format!("could not launch the engine: {e}"));
        }
    };

    // Drain the terminal RunManifest (stdout) on a separate thread so a full stdout pipe can never
    // deadlock against the stderr event stream we relay below.
    let stdout = child.stdout.take();
    let stdout_reader = std::thread::spawn(move || {
        let mut buffer = String::new();
        if let Some(mut handle) = stdout {
            let _ = handle.read_to_string(&mut buffer);
        }
        buffer
    });

    // Relay each RunEvent (one JSON object per stderr line) to the frontend as it is emitted. A line
    // the engine did not format as JSON is forwarded as a `{ "log": ... }` note rather than dropped.
    if let Some(stderr) = child.stderr.take() {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let event = serde_json::from_str::<Value>(&line)
                .unwrap_or_else(|_| serde_json::json!({ "log": line }));
            let _ = on_event.send(event);
        }
    }

    let manifest_text = stdout_reader.join().unwrap_or_default();
    let _ = child.wait(); // reap the child; the RunManifest carries the authoritative success/failure state
    let _ = std::fs::remove_file(&plan_path);
    serde_json::from_str(&manifest_text).map_err(|e| {
        format!("the engine produced no parseable RunManifest ({e}); see the streamed events")
    })
}

// --- Data Studio ------------------------------------------------------------
// The engine is the single writer of examples.jsonl (via `examples-append`); the shell
// only authors rows into a temp file and dispatches the sanctioned CLI. It never mutates
// a dataset itself.

/// Run a `corpus-studio` subcommand and return its trimmed stdout as text (for commands that
/// print a value, e.g. `new-project` emits the created project directory).
fn run_engine_text(args: &[&str]) -> Result<String, String> {
    let output = Command::new("corpus-studio")
        .args(args)
        .output()
        .map_err(|e| format!("could not launch the engine: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Write authored JSONL to a unique temp file the engine can read; the caller removes it.
fn write_temp_jsonl(content: &str) -> Result<std::path::PathBuf, String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut path = std::env::temp_dir();
    path.push(format!("corpus-studio-rows-{}-{}.jsonl", std::process::id(), nanos));
    let mut file =
        std::fs::File::create(&path).map_err(|e| format!("could not write temp rows: {e}"))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("could not write temp rows: {e}"))?;
    Ok(path)
}

/// The built-in dataset schemas (id, name, fields) — the engine owns the registry.
#[tauri::command]
fn data_schemas() -> Result<Value, String> {
    run_engine(&["schemas"])
}

/// Local dataset projects: `{projects_root, count, projects:[{id, name, schema_id, ...}]}`.
#[tauri::command]
fn data_projects() -> Result<Value, String> {
    run_engine(&["project-list"])
}

/// Create a local dataset project; returns `{project_dir}` (the engine prints the path).
#[tauri::command]
fn data_new_project(project_id: String, name: String, schema: String) -> Result<Value, String> {
    let dir = run_engine_text(&["new-project", &project_id, &name, &schema])?;
    Ok(serde_json::json!({ "project_dir": dir }))
}

/// Validate/preview authored JSONL rows against a schema (accepted/rejected report).
#[tauri::command]
fn data_preview(schema: String, rows_jsonl: String) -> Result<Value, String> {
    let temp = write_temp_jsonl(&rows_jsonl)?;
    let result = run_engine(&["import-preview", &temp.to_string_lossy(), &schema]);
    let _ = std::fs::remove_file(&temp);
    result
}

/// Commit authored rows into the project's examples.jsonl via the sanctioned single writer
/// (`examples-append --skip-invalid`); invalid rows are reported, not silently dropped.
#[tauri::command]
fn data_append(project_dir: String, rows_jsonl: String) -> Result<Value, String> {
    let temp = write_temp_jsonl(&rows_jsonl)?;
    let result = run_engine(&[
        "examples-append",
        &project_dir,
        "--from",
        &temp.to_string_lossy(),
        "--skip-invalid",
        "--json",
    ]);
    let _ = std::fs::remove_file(&temp);
    result
}

/// The graded dataset-debt report for a project's examples.jsonl.
#[tauri::command]
fn data_debt(project_dir: String) -> Result<Value, String> {
    let mut examples = std::path::PathBuf::from(&project_dir);
    examples.push("examples.jsonl");
    run_engine(&["dataset-debt", &examples.to_string_lossy(), "--json"])
}

/// Stage a source file to JSONL the engine can preview/commit. A `.jsonl` source is used directly;
/// CSV/TSV/Parquet is converted via `import-convert` into a temp file (the bool marks it for cleanup).
/// The engine does all reading/writing/conversion; the shell only routes paths.
fn stage_source(source: &str) -> Result<(std::path::PathBuf, bool), String> {
    if source.to_lowercase().ends_with(".jsonl") {
        return Ok((std::path::PathBuf::from(source), false));
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut staging = std::env::temp_dir();
    staging.push(format!("corpus-studio-import-{}-{}.jsonl", std::process::id(), nanos));
    run_engine(&["import-convert", source, &staging.to_string_lossy()])?;
    Ok((staging, true))
}

/// Validate/preview a source file (JSONL/CSV/TSV/Parquet) against the project schema without
/// committing — the accepted/rejected report the Import panel renders.
#[tauri::command]
fn data_import_preview(schema: String, source_path: String) -> Result<Value, String> {
    let (staging, is_temp) = stage_source(&source_path)?;
    let result = run_engine(&["import-preview", &staging.to_string_lossy(), &schema]);
    if is_temp {
        let _ = std::fs::remove_file(&staging);
    }
    result
}

/// Commit a source file's schema-valid rows into examples.jsonl via `import-commit` (the sanctioned
/// single writer; invalid rows are reported, not dropped) and capture a version.
#[tauri::command]
fn data_import_commit(project_dir: String, source_path: String) -> Result<Value, String> {
    let (staging, is_temp) = stage_source(&source_path)?;
    let result = run_engine(&[
        "import-commit",
        &project_dir,
        "--from",
        &staging.to_string_lossy(),
        "--json",
    ]);
    if is_temp {
        let _ = std::fs::remove_file(&staging);
    }
    result
}

/// The full quality report for a project's examples.jsonl (duplicates, low-information, PII findings,
/// synthetic-pattern signals) — the detail behind the debt grade.
#[tauri::command]
fn data_quality(project_dir: String) -> Result<Value, String> {
    let mut examples = std::path::PathBuf::from(&project_dir);
    examples.push("examples.jsonl");
    run_engine(&["quality", &examples.to_string_lossy()])
}

/// Run the dataset gates (schema / quality / PII-secrets / leakage) and save a report under
/// gate_reports/. The verdict (pass/warn/block) is in the JSON; a block is not an error exit.
#[tauri::command]
fn data_gate_run(project_dir: String, schema: String) -> Result<Value, String> {
    let mut examples = std::path::PathBuf::from(&project_dir);
    examples.push("examples.jsonl");
    run_engine(&[
        "gate-run",
        &examples.to_string_lossy(),
        &schema,
        "--scope",
        "dataset",
        "--project-dir",
        &project_dir,
    ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            platform_probe,
            platform_plan,
            platform_backends,
            platform_run,
            data_schemas,
            data_projects,
            data_new_project,
            data_preview,
            data_append,
            data_debt,
            data_import_preview,
            data_import_commit,
            data_quality,
            data_gate_run
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Corpus Studio shell");
}
