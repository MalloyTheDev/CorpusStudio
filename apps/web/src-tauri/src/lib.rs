//! Corpus Studio Tauri 2 shell. The window is a thin client over the Python platform engine: each
//! command shells out to the `corpus-studio platform-*` CLI (stdout = a JSON contract, stderr =
//! telemetry) and returns the parsed contract to the React frontend. The engine remains the single
//! source of truth; the shell never contains platform logic (see docs/platform-architecture-epic).

use std::process::Command;

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

/// Resolve a hash-sealed RunPlan for a base model + dataset on the chosen backend (RunPlan JSON).
#[tauri::command]
fn platform_plan(
    base_model: String,
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
    ];
    let backend_id = backend.unwrap_or_default();
    if !backend_id.is_empty() {
        args.push("--backend");
        args.push(&backend_id);
    }
    run_engine(&args)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            platform_probe,
            platform_plan,
            platform_backends
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Corpus Studio shell");
}
