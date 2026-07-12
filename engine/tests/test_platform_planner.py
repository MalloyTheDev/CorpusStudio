"""Platform slice 6 — the run planner. Pure tests (no torch): synthetic EnvironmentProfile +
CapabilityReport drive every resolution path and the honesty invariants (Blackwell→math from
cc_major, proven-only precision/quant, sequence_len flows, cpu_toy never a silent downgrade, a real
sha256 plan-hash that excludes the volatile stamp). The rendered snapshot is round-tripped through
the actual TrainRunConfig the runner replays."""

import pytest

import corpus_studio.platform as P
from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    CapabilityReport,
    EffectiveCapabilities,
    EnvironmentProfile,
    EnvHost,
    GpuDevice,
)
from corpus_studio.platform.planner import (
    PlannerConstraints,
    PlannerError,
    build_run_plan,
    compute_plan_hash,
)
from corpus_studio.training.trainer import TrainRunConfig

_SIG = "a" * 64
_NOW = "2026-07-11T00:00:00+00:00"


def _profile(*, cc_major=None, os="linux"):
    gpus = []
    if cc_major is not None:
        gpus = [
            GpuDevice(
                index=0, kind="cuda", name="GPU", vram_total_bytes=12_000_000_000,
                compute_capability=f"{cc_major}.0", compute_capability_major=cc_major,
            )
        ]
    return EnvironmentProfile(environment_signature=_SIG, host=EnvHost(os=os), gpus=gpus)


def _report(*, readiness="ready", bnb=True, precisions=("bf16",), attn=("sdpa",), missing=()):
    eff = EffectiveCapabilities(
        precision_modes=list(precisions),
        quantization_modes=["nf4"] if bnb else [],
        attention_impls=list(attn),
        adapter_methods=["qlora"],
    )
    return CapabilityReport(
        backend_id="corpus_studio", environment_ref=Ref(id=_SIG), readiness=readiness,
        bitsandbytes_ok=bnb, effective_capabilities=eff, missing_packages=list(missing),
    )


def _plan(profile, report, *, now=_NOW, **kw):
    kw.setdefault("base_model", "Qwen/Qwen2.5-7B-Instruct")
    kw.setdefault("dataset_path", "data/examples.jsonl")
    constraints = PlannerConstraints(**kw)
    return build_run_plan(
        profile=profile, capabilities=report, dataset_ref=Ref(id="ds-1"),
        constraints=constraints, plan_id="p1", now=now,
    )


# ---- resolution paths -------------------------------------------------------


def test_native_windows_blackwell_host_forces_math_bf16_nf4_qlora():
    plan = _plan(_profile(cc_major=12, os="windows"), _report())
    assert plan.attention_backend.value == "math"  # native-Windows Blackwell (WDDM) mandate
    assert plan.precision.value == "bf16"
    assert plan.quantization.value == "nf4"
    assert plan.adapter.method.value == "qlora"
    # math is not a from_pretrained string → the snapshot leaves the trainer's own path in control.
    assert "attn_implementation" not in plan.training_config_snapshot


def test_wsl_blackwell_host_keeps_sdpa_not_math():
    # WSL is its own platform: the flash deadlock is Windows-WDDM-only, so a WSL Blackwell host does
    # NOT force math — it seals the proven sdpa (→ flash on Linux CUDA). The whole reason to run under
    # WSL (verified on a real 5070 under WSL2).
    plan = _plan(_profile(cc_major=12, os="wsl"), _report(attn=("sdpa",)))
    assert plan.attention_backend.value == "sdpa"


def test_non_blackwell_with_proven_sdpa_uses_sdpa():
    plan = _plan(_profile(cc_major=8), _report(attn=("sdpa",)))
    assert plan.attention_backend.value == "sdpa"


def test_no_proven_attention_falls_back_to_eager():
    plan = _plan(_profile(cc_major=8), _report(attn=()))
    assert plan.attention_backend.value == "eager"
    assert plan.training_config_snapshot["attn_implementation"] == "eager"


def test_bf16_not_proven_falls_back_to_fp32():
    plan = _plan(_profile(cc_major=8), _report(precisions=("fp16",)))
    assert plan.precision.value == "fp32"


def test_no_bitsandbytes_gives_no_quant_and_lora():
    plan = _plan(_profile(cc_major=8), _report(bnb=False))
    assert plan.quantization.value == "none"
    assert plan.adapter.method.value == "lora"


def test_explicit_attention_override_wins():
    plan = _plan(_profile(cc_major=8), _report(), attention_backend="flash_attention_2")
    assert plan.attention_backend.value == "flash_attention_2"
    assert plan.training_config_snapshot["attn_implementation"] == "flash_attention_2"


def test_native_windows_blackwell_rejects_an_explicit_unsafe_attention_override():
    # The native-Windows Blackwell (WDDM) math mandate outranks the request. The fused/flash family
    # deadlocks outright there, and plain sdpa can DISPATCH to the flash kernel — only math/eager are
    # guaranteed safe under WDDM+sm_120.
    for unsafe in ("flash_attention_2", "mem_efficient", "sdpa"):
        with pytest.raises(PlannerError, match="deadlock"):
            _plan(_profile(cc_major=12, os="windows"), _report(), attention_backend=unsafe)


def test_wsl_blackwell_allows_an_explicit_sdpa_override():
    # On WSL the deadlock does not apply, so an explicit sdpa override is honored (not refused).
    plan = _plan(_profile(cc_major=12, os="wsl"), _report(), attention_backend="sdpa")
    assert plan.attention_backend.value == "sdpa"


def test_native_windows_blackwell_allows_only_math_and_eager_explicit_attention():
    for safe in ("eager", "math"):
        plan = _plan(_profile(cc_major=12, os="windows"), _report(), attention_backend=safe)
        assert plan.attention_backend.value == safe


def test_unsloth_refused_on_native_windows_blackwell_even_with_an_explicit_sdpa_override():
    # The "Unsloth refused on native-Windows sm_120" invariant must NOT be bypassable: an explicit sdpa
    # (which Unsloth declares) is itself refused there, so Unsloth can't be sealed by any attention path.
    with pytest.raises(PlannerError, match="deadlock"):
        _plan(_profile(cc_major=12, os="windows"), _report(), backend="unsloth", attention_backend="sdpa")


def test_unsupported_adapter_method_is_rejected():
    # dora / ia3 / full_finetune are in the enum but the corpus_studio backend declares only lora/qlora,
    # so the planner refuses rather than emit a plan that would be silently trained as plain LoRA.
    for method in ("dora", "ia3", "full_finetune"):
        with pytest.raises(PlannerError, match="adapter"):
            _plan(_profile(cc_major=12), _report(), adapter_method=method)


# ---- multi-backend selection ------------------------------------------------


def test_backend_ref_reflects_the_chosen_backend():
    # Unsloth on a non-Blackwell host with a proven sdpa plan.
    plan = _plan(_profile(cc_major=8), _report(attn=("sdpa",)), backend="unsloth")
    assert plan.backend_ref.id == "unsloth"


def test_unknown_backend_is_rejected():
    with pytest.raises(PlannerError, match="unknown backend"):
        _plan(_profile(cc_major=8), _report(), backend="megatron")


def test_backend_that_cannot_run_the_resolved_plan_is_rejected_with_alternatives():
    # Unsloth can't do the math attention a NATIVE-WINDOWS Blackwell plan requires → refused,
    # corpus_studio named. (On WSL the plan seals sdpa, which Unsloth CAN run — see the CLI test.)
    with pytest.raises(PlannerError, match="can't run this plan"):
        _plan(_profile(cc_major=12, os="windows"), _report(), backend="unsloth")


def test_default_backend_is_corpus_studio():
    plan = _plan(_profile(cc_major=12), _report())
    assert plan.backend_ref.id == "corpus_studio"


def test_lora_and_qlora_adapters_are_allowed():
    assert _plan(_profile(cc_major=8), _report(bnb=False), adapter_method="lora").adapter.method.value == "lora"
    assert _plan(_profile(cc_major=12), _report(), adapter_method="qlora").adapter.method.value == "qlora"


# ---- cpu-toy + readiness (honesty) ------------------------------------------


def test_cpu_toy_only_with_optin_yields_a_cpu_toy_plan():
    plan = _plan(_profile(), _report(readiness="cpu_toy_only", bnb=False), allow_cpu_toy=True)
    assert plan.precision.value == "fp32"
    assert plan.quantization.value == "none"
    assert plan.attention_backend.value == "eager"
    assert plan.training_config_snapshot["cpu_toy"] is True


def test_cpu_toy_only_without_optin_raises():
    with pytest.raises(PlannerError, match="cpu"):
        _plan(_profile(), _report(readiness="cpu_toy_only"))


def test_not_ready_raises_with_missing_packages():
    with pytest.raises(PlannerError, match="not ready"):
        _plan(_profile(), _report(readiness="not_ready", missing=["torch", "bitsandbytes"]))


def test_unsupported_task_type_raises():
    with pytest.raises(PlannerError, match="task_type"):
        _plan(_profile(cc_major=12), _report(), task_type="telepathy")


def test_unsupported_attention_override_raises():
    with pytest.raises(PlannerError, match="attention_backend"):
        _plan(_profile(cc_major=8), _report(), attention_backend="quantum")


# ---- sequence_len flows (no hardcoded calibration value) --------------------


def test_sequence_len_flows_verbatim():
    plan = _plan(_profile(cc_major=12), _report(), sequence_len=1792)
    assert plan.sequence.max_sequence_len == 1792
    assert plan.training_config_snapshot["sequence_len"] == 1792


# ---- the snapshot round-trips through the real trainer config ---------------


def test_snapshot_validates_as_a_trainrunconfig():
    plan = _plan(_profile(cc_major=12), _report())
    cfg = TrainRunConfig.model_validate(plan.training_config_snapshot)
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_format == "instruction"  # NOT silently defaulted from a wrong "format" key


def test_snapshot_uses_dataset_format_key_not_format():
    plan = _plan(_profile(cc_major=12), _report(), dataset_format="chat")
    assert plan.training_config_snapshot["dataset_format"] == "chat"
    assert "format" not in plan.training_config_snapshot


# ---- plan_hash (the immutability seal) --------------------------------------


def test_plan_hash_is_a_real_lowercase_sha256():
    plan = _plan(_profile(cc_major=12), _report())
    assert P.RunPlan.model_validate_json(plan.model_dump_json()) == plan
    assert len(plan.plan_hash) == 64
    assert plan.plan_hash == plan.plan_hash.lower()
    assert plan.plan_hash != "0" * 64


def test_plan_hash_excludes_the_volatile_created_at():
    a = _plan(_profile(cc_major=12), _report(), now="2026-01-01T00:00:00+00:00")
    b = _plan(_profile(cc_major=12), _report(), now="2027-12-31T23:59:59+00:00")
    assert a.created_at != b.created_at
    assert a.plan_hash == b.plan_hash  # identical plan body → identical seal


def test_plan_hash_changes_when_a_planned_field_changes():
    base = _plan(_profile(cc_major=12), _report())
    other = _plan(_profile(cc_major=12), _report(), learning_rate=1e-5)
    assert base.plan_hash != other.plan_hash


def test_compute_plan_hash_is_order_independent():
    assert compute_plan_hash({"a": 1, "b": 2}) == compute_plan_hash({"b": 2, "a": 1})


# ---- linkage ----------------------------------------------------------------


def test_environment_ref_links_the_profile_signature():
    plan = _plan(_profile(cc_major=12), _report())
    assert plan.environment_ref.id == _SIG
    assert plan.dataset_ref.id == "ds-1"
    assert plan.backend_ref.id == "corpus_studio"


def test_default_clock_stamps_created_at():
    plan = _plan(_profile(cc_major=12), _report(), now=None)
    assert plan.created_at is not None
    assert plan.created_at.endswith("+00:00")


def test_an_invalid_resolved_field_becomes_planner_error():
    # sequence_len=0 fails SequenceSpec.max_sequence_len (ge=1) → a clean PlannerError, not a raw
    # pydantic ValidationError leaking out.
    with pytest.raises(PlannerError, match="invalid"):
        _plan(_profile(cc_major=12), _report(), sequence_len=0)
