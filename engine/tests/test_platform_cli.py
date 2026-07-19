"""The platform CLI JSON contracts — the exact stdout the Tauri shell / apps/web live flow parses.
`platform-plan` runs the real profiler+probes, so on a torch-less CI host it would refuse (not_ready);
we inject a synthetic READY host to exercise the resolve→fit→bundle path, and prove the --backend
selection flows through the CLI (incl. the honest Unsloth-on-Blackwell refusal)."""

import json

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.common import PackageLock, Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    EnvHost,
    ExecutionCapabilityCombination,
    GpuDevice,
    ParameterAccountingReport,
    ParameterEvidenceGap,
    ParameterScope,
    ParameterWindow,
    ProbeResult,
)
from corpus_studio.platform.parameter_accounting import parameter_accounting_hash_for

runner = CliRunner()
_SIG = "b" * 64
_MODEL_REVISION = "1" * 40


def _platform_plan_args(tmp_path, *, base_model: str = "m") -> list[str]:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Say hello.", "output": "Hello."}) + "\n",
        encoding="utf-8",
    )
    return [
        "platform-plan",
        "--base-model",
        base_model,
        "--model-revision",
        _MODEL_REVISION,
        "--dataset",
        str(dataset),
    ]


def _ready_profile(cc_major: int = 8, os_name: str = "linux") -> EnvironmentProfile:
    return EnvironmentProfile(
        environment_signature=_SIG,
        host=EnvHost(os=os_name),
        gpus=[
            GpuDevice(
                index=0, kind="cuda", name="Synthetic", vram_total_bytes=12_000_000_000,
                compute_capability=f"{cc_major}.0", compute_capability_major=cc_major,
            )
        ],
    )


def _ready_report(attn: str = "sdpa", *, backend_id: str = "corpus_studio") -> CapabilityReport:
    from corpus_studio.platform.backends import get_backend

    first_party = get_backend("corpus_studio")
    assert first_party is not None
    kernel = {
        "eager": "eager",
        "math": "torch_sdpa_math",
        "sdpa": "torch_sdpa_flash",
    }[attn]
    selected_backend = get_backend(backend_id)
    combination = ExecutionCapabilityCombination.model_validate(
        {
            "runtime_mode": "training",
            "device": "cuda",
            "precision": "bf16",
            "quantization": "nf4",
            "adapter_method": "qlora",
            "attention_impl": attn,
            "attention_kernel": kernel,
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": "synthetic_execution",
        }
    )
    probe_results = [
        ProbeResult(
            probe="synthetic_axes",
            outcome="PASS",
            proves={
                "adapter": ["qlora"],
                "attention": [attn],
                "attention_kernel": [kernel],
                "checkpoint": ["adapter_only"],
                "loss": ["cross_entropy", "liger_fused_ce"],
                "optimizer": ["adamw_torch", "paged_adamw_8bit"],
                "precision": ["bf16"],
            },
        ),
        ProbeResult(
            probe="bnb_4bit_load",
            outcome="PASS",
            proves={"quantization": ["nf4"]},
        ),
        ProbeResult(
            probe="trainer_contract",
            outcome="PASS",
            proves={
                "trainer_field": first_party.trainer_fields,
                "trainer_init_field": first_party.trainer_init_fields,
            },
        ),
        ProbeResult(
            probe="synthetic_execution",
            outcome="PASS",
            execution_combinations=[combination],
        ),
    ]
    return CapabilityReport(
        backend_id=backend_id,
        backend_version=selected_backend.backend_version if selected_backend is not None else None,
        environment_ref=Ref(id=_SIG), readiness="ready",
        bitsandbytes_ok=True,
        probe_results=probe_results,
        installed_packages=[
            PackageLock(name=name, version="1.0")
            for name in [
                "accelerate",
                "bitsandbytes",
                "datasets",
                "liger-kernel",
                "peft",
                "torch",
                "transformers",
                "trl",
            ]
        ],
        effective_capabilities=EffectiveCapabilities(
            precision_modes=["bf16"], quantization_modes=["nf4"], attention_impls=[attn],
            attention_kernels=[kernel],
            adapter_methods=["qlora"],
            optimizers=["adamw_torch", "paged_adamw_8bit"],
            loss_impls=["cross_entropy", "liger_fused_ce"],
            checkpoint_impls=["adapter_only"],
            execution_contract_versions=["1.0.0"],
            execution_combinations=[combination],
            trainer_fields=first_party.trainer_fields,
            trainer_init_fields=first_party.trainer_init_fields,
        ),
    )


def _ready_host(
    monkeypatch,
    *,
    cc_major: int = 8,
    attn: str = "sdpa",
    os_name: str = "linux",
    backend_id: str = "corpus_studio",
) -> None:
    """Inject a synthetic ready host so the planner produces a plan without torch/a GPU present.
    ``os_name`` matters on Blackwell: only native Windows forces the math mandate (WSL/Linux keep sdpa)."""
    monkeypatch.setattr(
        "corpus_studio.platform.profiler.build_environment_profile",
        lambda: _ready_profile(cc_major, os_name),
    )
    monkeypatch.setattr(
        "corpus_studio.platform.probes.run_capability_probes",
        lambda _profile: _ready_report(attn, backend_id=backend_id),
    )


def _accounting_report():
    model_ref = Ref(id="model", hash={"value": "c" * 64})
    scope = ParameterScope(
        scope_id="model",
        kind="model",
        model_ref=model_ref,
        coordinate_universe_id="model-coordinates",
        coordinate_universe_sha256="c" * 64,
        definition="One model coordinate universe.",
    )
    draft = ParameterAccountingReport(
        report_id="parameter-report",
        report_hash="0" * 64,
        generated_at="2026-07-13T00:00:00Z",
        profile="model_static",
        status="incomplete",
        model_ref=model_ref,
        gaps=[
            ParameterEvidenceGap(
                gap_id="logical-gap",
                kind="logical",
                scope=scope,
                window=ParameterWindow(
                    window_id="static-model",
                    kind="static_snapshot",
                    definition="One static snapshot.",
                ),
                reason="missing_observation",
                explanation="Logical evidence is absent.",
                resolution="Supply measured evidence.",
            )
        ],
    )
    return draft.model_copy(update={"report_hash": parameter_accounting_hash_for(draft)})


# ---- platform-backends -------------------------------------------------------


def test_platform_backends_json_lists_the_registry():
    result = runner.invoke(app, ["platform-backends", "--json"])
    assert result.exit_code == 0
    ids = [b["backend_id"] for b in json.loads(result.stdout)]
    assert ids == ["corpus_studio", "unsloth"]


# ---- platform-probe (works torch-less: readiness present) --------------------


def test_platform_probe_json_bundles_profile_and_report():
    result = runner.invoke(app, ["platform-probe", "--json"])
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)
    assert "environment_profile" in bundle
    assert "capability_report" in bundle
    assert bundle["capability_report"]["readiness"] in {"ready", "cpu_toy_only", "not_ready"}


# ---- platform-plan --json (the live-flow bundle) -----------------------------


def test_platform_plan_json_bundles_plan_and_fit(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [*_platform_plan_args(tmp_path, base_model="Qwen/Qwen2.5-7B"), "--json"],
    )
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)
    assert len(bundle["run_plan"]["plan_hash"]) == 64
    assert bundle["run_plan"]["backend_ref"]["id"] == "corpus_studio"
    resolved = bundle["run_plan"]["resolved_execution"]
    assert len(resolved["configuration_hash"]) == 64
    assert resolved["inputs"]["model"]["resolved_revision"] == _MODEL_REVISION
    assert resolved["inputs"]["dataset"]["location"].endswith("dataset.jsonl")
    assert bundle["run_plan"]["training_config_snapshot"] == {}
    physical = bundle["run_plan"]["physical_execution"]
    assert physical["evidence_status"] == "planned_not_measured"
    assert physical["parallelism"]["world_size"] == 1
    assert physical["offload_rules"] == []
    # a predicted fit rides along — and it is never the measured-only NATIVE_SAFE
    assert bundle["fit_classification"]["classification"] != "NATIVE_SAFE"


def test_platform_plan_dataset_format_flows_through_the_cli(monkeypatch, tmp_path):
    # A chat dataset (messages) planned as chat: the resolved execution data policy is the exact
    # formatter contract consumed by the worker. The dataset MUST be structurally chat (the new
    # conformance preflight refuses a chat plan over an instruction-shaped dataset).
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--model-revision",
            _MODEL_REVISION,
            "--dataset",
            str(_chat_dataset(tmp_path)),
            "--dataset-format",
            "chat",
            "--chat-template-sha256",
            "c" * 64,
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)["run_plan"]["resolved_execution"]["data"]
    assert data["dataset_format"] == "chat"
    assert data["chat_template_sha256"] == "c" * 64
    assert "format" not in data


def test_platform_plan_allocator_policy_flows_through_the_cli(monkeypatch, tmp_path):
    # --allocator-policy + --max-split-size-mb seal the CUDA allocator config into the plan (so the
    # seq-4096 paged config no longer smuggles it via the dispatch env).
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [
            *_platform_plan_args(tmp_path),
            "--allocator-policy",
            "max_split_size",
            "--max-split-size-mb",
            "128",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    plan = json.loads(result.stdout)["run_plan"]
    assert plan["allocator_policy"] == "max_split_size"
    assert plan["allocator_max_split_size_mb"] == 128


def test_platform_plan_output_dir_flows_into_resolved_execution(monkeypatch, tmp_path):
    # --output-dir controls where the trainer saves the adapter (so a run can target a project dir,
    # not the CWD). It must reach the sealed worker configuration verbatim.
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [
            *_platform_plan_args(tmp_path),
            "--output-dir",
            "/some/project/adapters/wbg",
            "--json",
        ],
    )
    assert result.exit_code == 0
    resolved = json.loads(result.stdout)["run_plan"]["resolved_execution"]
    assert resolved["output_dir"] == "/some/project/adapters/wbg"


def test_platform_plan_memory_efficient_requires_an_exact_combination_probe(monkeypatch, tmp_path):
    # Independent paged-optimizer and Liger evidence cannot be combined into an executable claim.
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [*_platform_plan_args(tmp_path), "--memory-efficient", "--json"],
    )
    assert result.exit_code == 2
    assert "complete requested execution tuple" in result.output


def test_platform_plan_refuses_unsloth_without_execution_contract(monkeypatch, tmp_path):
    # Host capability evidence cannot upgrade a backend whose manifest does not implement the sealed
    # execution contract. Unsloth remains unavailable until its worker consumes contract 1.0.0.
    _ready_host(monkeypatch, cc_major=8, attn="sdpa", backend_id="unsloth")
    result = runner.invoke(
        app,
        [
            *_platform_plan_args(tmp_path),
            "--backend",
            "unsloth",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "resolved execution contract '1.0.0' not supported" in result.output


def test_platform_plan_rejects_unsloth_on_a_native_windows_blackwell_math_plan(
    monkeypatch, tmp_path
):
    # NATIVE WINDOWS + Blackwell forces math (WDDM flash deadlock); Unsloth declares no math, and it
    # also lacks the resolved execution contract, so the planner refuses through the CLI.
    _ready_host(
        monkeypatch,
        cc_major=12,
        attn="math",
        os_name="windows",
        backend_id="unsloth",
    )
    result = runner.invoke(
        app,
        [
                *_platform_plan_args(tmp_path),
                "--backend",
                "unsloth",
            ],
    )
    assert result.exit_code == 2
    assert "can't run this plan" in result.stderr or "can't run this plan" in result.output
    assert "attention 'math' not supported" in result.output
    assert "resolved execution contract '1.0.0' not supported" in result.output


def test_platform_plan_refuses_unsloth_on_wsl_without_execution_contract(monkeypatch, tmp_path):
    # The WDDM-only math mandate does not apply to a WSL profile, so attention resolves to proven SDPA.
    # The plan is still refused because Unsloth does not implement the resolved execution contract.
    _ready_host(
        monkeypatch, cc_major=12, attn="sdpa", os_name="wsl", backend_id="unsloth"
    )
    result = runner.invoke(
        app,
        [
                *_platform_plan_args(tmp_path),
                "--backend",
                "unsloth",
                "--json",
        ],
    )
    assert result.exit_code == 2
    assert "resolved execution contract '1.0.0' not supported" in result.output
    assert "attention 'sdpa' not supported" not in result.output


def test_platform_plan_json_survives_stdout_noise_from_a_probe(monkeypatch, tmp_path):
    # A probe/import that prints a banner to STDOUT (historically older bitsandbytes) must not corrupt
    # the JSON bundle the Tauri shell parses — the CLI redirects probe-time stdout to stderr.
    def _noisy_profile() -> EnvironmentProfile:
        print("=== Welcome to bitsandbytes! (BUG REPORT banner) ===")  # noqa: T201 - simulates the lib
        return _ready_profile()

    monkeypatch.setattr("corpus_studio.platform.profiler.build_environment_profile", _noisy_profile)
    monkeypatch.setattr(
        "corpus_studio.platform.probes.run_capability_probes", lambda _p: _ready_report()
    )
    result = runner.invoke(
        app, [*_platform_plan_args(tmp_path), "--json"]
    )
    assert result.exit_code == 0
    bundle = json.loads(result.stdout)  # pure JSON despite the banner
    assert len(bundle["run_plan"]["plan_hash"]) == 64
    assert "bitsandbytes" in result.stderr  # the banner was redirected off stdout, onto stderr


def test_platform_plan_loads_and_copies_a_hash_sealed_parameter_report(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    plan_args = _platform_plan_args(tmp_path)
    report = _accounting_report()
    report_path = tmp_path / "ParameterAccountingReport.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "plan"
    result = runner.invoke(
        app,
        [
            *plan_args,
            "--parameter-accounting-report",
            str(report_path),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.stdout)
    pinned = bundle["run_plan"]["parameter_accounting_ref"]
    assert pinned["id"] == report.report_id
    assert pinned["hash"]["value"] == report.report_hash
    assert (out / "ParameterAccountingReport.json").exists()


def test_platform_plan_refuses_a_tampered_parameter_report(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    plan_args = _platform_plan_args(tmp_path)
    report = _accounting_report().model_copy(update={"report_hash": "0" * 64})
    path = tmp_path / "tampered.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            *plan_args,
            "--parameter-accounting-report",
            str(path),
        ],
    )
    assert result.exit_code == 2
    assert "hash mismatch" in result.output


def test_platform_plan_refuses_unverified_nontrivial_physical_spec(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    plan_args = _platform_plan_args(tmp_path)
    spec = {
        "resources": [
            {
                "resource_id": "compute-0",
                "tier": "gpu",
                "device_kind": "cuda",
                "device_id": "cuda:0",
            },
            {
                "resource_id": "host-ram",
                "tier": "pageable_ram",
                "device_kind": "cpu",
                "device_id": "cpu:0",
            },
        ],
        "placements": [
            {
                "placement_id": "parameters-authoritative",
                "state": "parameters",
                "selector": {"whole_model": True},
                "resource_id": "compute-0",
                "role": "authoritative",
            }
        ],
        "offload_rules": [
            {
                "rule_id": "parameter-offload",
                "state": "parameters",
                "selector": {"whole_model": True},
                "source_resource_id": "compute-0",
                "target_resource_id": "host-ram",
                "mechanism": "cpu_copy",
                "trigger": "after_use",
            }
        ],
        "parallelism": {
            "world_size": 1,
            "ranks": [{"rank": 0, "resource_id": "compute-0"}],
        },
    }
    path = tmp_path / "physical.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            *plan_args,
            "--physical-spec",
            str(path),
        ],
    )
    assert result.exit_code == 2
    assert "can't run the physical plan" in result.output


def test_platform_run_refuses_a_tampered_plan_hash(tmp_path):
    from corpus_studio.platform.supervisor import demo_run_plan

    body = demo_run_plan().model_dump(mode="json")
    body["seed"] += 1
    path = tmp_path / "RunPlan.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    result = runner.invoke(app, ["platform-run", str(path)])
    assert result.exit_code == 2
    assert "plan_hash does not match" in result.output


def test_platform_run_help_exposes_the_bounded_preflight_deadline():
    from typer.main import get_command

    root = get_command(app)
    platform_run_command = root.commands["platform-run"]
    option = next(
        parameter
        for parameter in platform_run_command.params
        if "--preflight-timeout" in parameter.opts
    )
    assert option.default == 1800.0
    assert "Non-extendable deadline" in (option.help or "")


# ---- batching flags: --micro-batch-size / --gradient-accumulation-steps -------
# The planner defaults (micro_batch_size=1, gradient_accumulation_steps=8) are product-wide behavior;
# these opt-in flags let a caller seal an EXACT batching tuple (e.g. a preregistered research cell that
# mandates gradient accumulation 1) without changing what omitting the flags produces. The sealed value
# is consumed verbatim by the worker (train_config_from_resolved), so it must survive plan -> resolved
# execution -> trainer config with no post-seal override.


def _resolved_batching(run_plan: dict) -> tuple[int, int]:
    from corpus_studio.platform.contracts import ResolvedExecutionConfiguration
    from corpus_studio.training.trainer import train_config_from_resolved

    resolved = run_plan["resolved_execution"]
    execution = ResolvedExecutionConfiguration.model_validate(resolved)
    cfg = train_config_from_resolved(execution)
    return cfg.micro_batch_size, cfg.gradient_accumulation_steps


@pytest.mark.parametrize("flag", ["--micro-batch-size", "--gradient-accumulation-steps"])
def test_platform_plan_batching_flags_appear_in_help(flag):
    from typer.main import get_command

    root = get_command(app)
    platform_plan_command = root.commands["platform-plan"]
    option = next(
        parameter for parameter in platform_plan_command.params if flag in parameter.opts
    )
    # the option is exposed with help text and a positivity floor (click range min=1)
    assert (option.help or "").strip()
    assert getattr(option.type, "min", None) == 1


@pytest.mark.parametrize("flag", ["--micro-batch-size", "--gradient-accumulation-steps"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_platform_plan_rejects_nonpositive_batching(tmp_path, flag, value):
    # Range validation happens at the click layer, before host profiling, so no ready host is needed:
    # a zero or negative microstep count can never seal a plan.
    result = runner.invoke(app, [*_platform_plan_args(tmp_path), flag, value, "--json"])
    assert result.exit_code != 0


def test_platform_plan_sealed_batching_flows_to_plan_resolved_and_trainer(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    result = runner.invoke(
        app,
        [
            *_platform_plan_args(tmp_path),
            "--micro-batch-size",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0
    run_plan = json.loads(result.stdout)["run_plan"]
    # micro-batch=1 and grad-accum=1 reach the sealed RunPlan (top-level batching)
    assert run_plan["batching"]["micro_batch_size"] == 1
    assert run_plan["batching"]["fallback_grad_accumulation_steps"] == 1
    # both values reach the ResolvedExecutionConfiguration
    resolved = run_plan["resolved_execution"]["batching"]
    assert resolved["micro_batch_size"] == 1
    assert resolved["fallback_grad_accumulation_steps"] == 1
    # and the trainer receives the exact sealed values (no post-seal override)
    assert _resolved_batching(run_plan) == (1, 1)


def test_platform_plan_omitted_batching_flags_preserve_existing_defaults(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    result = runner.invoke(app, [*_platform_plan_args(tmp_path), "--json"])
    assert result.exit_code == 0
    run_plan = json.loads(result.stdout)["run_plan"]
    # omitting the flags preserves the product-wide defaults: micro_batch_size=1, grad-accum=8
    assert run_plan["batching"]["micro_batch_size"] == 1
    assert run_plan["batching"]["fallback_grad_accumulation_steps"] == 8
    resolved = run_plan["resolved_execution"]["batching"]
    assert resolved["micro_batch_size"] == 1
    assert resolved["fallback_grad_accumulation_steps"] == 8
    assert _resolved_batching(run_plan) == (1, 8)


# ---- dataset-format structural conformance (refuse before sealing a plan) -----
# Regression for the observed UNSUPPORTED_CONFIGURATION / "no usable training rows" failure: a chat
# dataset planned as instruction rendered zero rows and only failed AFTER GPU model allocation. The
# planner must now refuse such a plan on the CPU, before any plan id/hash is minted.


def _chat_dataset(tmp_path):
    # mirrors pipeline_smoke_fixture_v2.jsonl: a messages list of system/user/assistant turns
    ds = tmp_path / "chat.jsonl"
    ds.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Write the lowercase form of ALPHA."},
                    {"role": "assistant", "content": "alpha"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return ds


def test_platform_plan_refuses_chat_dataset_planned_as_instruction(monkeypatch, tmp_path):
    from corpus_studio.platform.execution_config import stable_file_sha256

    _ready_host(monkeypatch)
    ds = _chat_dataset(tmp_path)
    digest_before = stable_file_sha256(str(ds))
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--model-revision",
            _MODEL_REVISION,
            "--dataset",
            str(ds),
            "--dataset-format",
            "instruction",
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "structurally incompatible" in result.output
    # no plan id or plan hash is minted for a refused plan
    assert not out.exists() or not list(out.rglob("RunPlan.json"))
    # the dataset bytes are unchanged (read-only preflight, no auto-switch/repair)
    assert stable_file_sha256(str(ds)) == digest_before


def test_platform_plan_accepts_chat_dataset_planned_as_chat(monkeypatch, tmp_path):
    _ready_host(monkeypatch)
    ds = _chat_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model",
            "m",
            "--model-revision",
            _MODEL_REVISION,
            "--dataset",
            str(ds),
            "--dataset-format",
            "chat",
            "--chat-template-sha256",
            "c" * 64,
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)["run_plan"]["resolved_execution"]["data"]
    assert data["dataset_format"] == "chat"


def test_platform_plan_instruction_dataset_remains_compatible(monkeypatch, tmp_path):
    # existing instruction callers + the instruction default keep passing the new preflight
    _ready_host(monkeypatch)
    result = runner.invoke(app, [*_platform_plan_args(tmp_path), "--json"])
    assert result.exit_code == 0


def _partial_chat_dataset(tmp_path):
    # Two structurally-valid chat rows plus one that cannot render (no messages list). The set is
    # is_conformant (>=1 compatible) so the zero-row gate passes; the partial gate is what must fire.
    ds = tmp_path / "partial_chat.jsonl"
    good = {
        "messages": [
            {"role": "user", "content": "Write the lowercase form of ALPHA."},
            {"role": "assistant", "content": "alpha"},
        ]
    }
    ds.write_text(
        "\n".join(json.dumps(r) for r in (good, good, {"note": "not chat-shaped"})) + "\n",
        encoding="utf-8",
    )
    return ds


def test_platform_plan_refuses_partially_unrenderable_dataset(monkeypatch, tmp_path):
    # A plan must not silently seal claiming the whole dataset when only some rows render. Default is
    # fail closed: 2 of 3 chat rows render, 1 does not -> refuse with the exact counts, mint no plan.
    from corpus_studio.platform.execution_config import stable_file_sha256

    _ready_host(monkeypatch)
    ds = _partial_chat_dataset(tmp_path)
    digest_before = stable_file_sha256(str(ds))
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model", "m",
            "--model-revision", _MODEL_REVISION,
            "--dataset", str(ds),
            "--dataset-format", "chat",
            "--chat-template-sha256", "c" * 64,
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "1 of 3 row(s)" in result.output
    assert "over-claim the trained row count" in result.output
    assert "--allow-unrenderable-rows" in result.output
    # no plan id, plan hash, or sealed conformance is minted for a refused plan
    assert not out.exists() or not list(out.rglob("RunPlan.json"))
    assert not out.exists() or not list(out.rglob("DatasetConformance.json"))
    # read-only preflight: the dataset bytes are unchanged
    assert stable_file_sha256(str(ds)) == digest_before


def test_platform_plan_allow_unrenderable_rows_seals_and_records_conformance(monkeypatch, tmp_path):
    # With the explicit opt-in, the plan seals but is HONEST: it warns, and the sealed evidence +
    # --json bundle record exactly how many rows the dataset_format renders vs drops.
    _ready_host(monkeypatch)
    ds = _partial_chat_dataset(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "platform-plan",
            "--base-model", "m",
            "--model-revision", _MODEL_REVISION,
            "--dataset", str(ds),
            "--dataset-format", "chat",
            "--chat-template-sha256", "c" * 64,
            "--allow-unrenderable-rows",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0
    # the operator is warned on stderr (never silent) about the dropped rows
    assert "WARNING" in result.stderr and "1 of 3 row(s)" in result.stderr
    # the sealed conformance verdict is written next to the plan
    sealed = json.loads((out / "DatasetConformance.json").read_text(encoding="utf-8"))
    assert sealed == {
        "dataset_format": "chat",
        "total_rows": 3,
        "compatible_rows": 2,
        "rejected_rows": 1,
        "representative_rejections": [{"index": 2, "reason": "no non-empty 'messages' list"}],
    }
    # and the live-flow bundle carries the same verdict for the client
    bundle = json.loads(result.stdout)
    assert bundle["dataset_conformance"]["compatible_rows"] == 2
    assert bundle["dataset_conformance"]["rejected_rows"] == 1


def test_platform_plan_bundle_records_full_conformance_for_a_clean_dataset(monkeypatch, tmp_path):
    # A fully compatible dataset seals with rejected_rows == 0 recorded, so a downstream reader can
    # always trust the sealed row accounting rather than inferring "all rows" from silence.
    _ready_host(monkeypatch)
    result = runner.invoke(app, [*_platform_plan_args(tmp_path), "--json"])
    assert result.exit_code == 0
    conformance = json.loads(result.stdout)["dataset_conformance"]
    assert conformance["rejected_rows"] == 0
    assert conformance["compatible_rows"] == conformance["total_rows"] >= 1
