"""Platform slice 8 — the fit calibrator. Pure tests (torch-free): synthetic RunPlan + profile drive
every fit band. The bands are asserted relative to the ACTUAL estimator output (computed in-test) so
they can't drift, and the honesty invariant — an estimate NEVER earns NATIVE_SAFE — is enforced."""

import corpus_studio.platform as P
from corpus_studio.platform.calibrator import classify_fit
from corpus_studio.platform.contracts import EnvHost, EnvironmentProfile, GpuDevice
from corpus_studio.platform.enums import FitClass
from corpus_studio.training.estimators import build_vram_estimate

_GB = 1_000_000_000


def _plan(*, base_model="Qwen/Qwen2.5-7B-Instruct", quantization="nf4", attention="math",
          sequence_len=4096, micro_batch=1, lora_r=16, cpu_toy=False, precision="bf16"):
    from corpus_studio.platform.supervisor import demo_run_plan

    body = demo_run_plan().model_dump(mode="json")
    body["base_model"] = base_model
    body["quantization"] = quantization
    body["precision"] = precision
    body["attention_backend"] = attention
    body["sequence"]["max_sequence_len"] = sequence_len
    body["batching"]["micro_batch_size"] = micro_batch
    body["adapter"] = {
        "method": "qlora" if quantization in ("nf4", "int4", "fp4") else "lora",
        "lora_r": lora_r,
        "lora_alpha": lora_r * 2,
    }
    if cpu_toy:
        body["training_config_snapshot"] = {"cpu_toy": True}
    return P.RunPlan.model_validate(body)


def _profile(*, capacity_gb=12.0, residency="wddm", has_gpu=True):
    gpus = []
    if has_gpu:
        gpus = [GpuDevice(index=0, kind="cuda", name="GPU", vram_total_bytes=int(capacity_gb * _GB),
                          compute_capability_major=12)]
    return EnvironmentProfile(
        environment_signature="a" * 64,
        host=EnvHost(os="windows", memory_residency_model=residency),
        gpus=gpus,
    )


def _peak_gb(quantization="nf4", math_attention=True):
    """The estimator's own peak for the default 7B plan — so tests track the calibrated arithmetic."""
    est = build_vram_estimate("Qwen/Qwen2.5-7B-Instruct", sequence_len=4096, adapter="qlora",
                              math_attention=math_attention)
    return {"nf4": est.total_gb_int4, "int8": est.total_gb_int8}.get(quantization, est.total_gb_fp16)


# ---- bands ------------------------------------------------------------------


def test_comfortable_fit_is_native_unproven_not_safe():
    peak = _peak_gb()
    fit = classify_fit(_plan(), _profile(capacity_gb=peak + 4.0))
    assert fit.classification == FitClass.NATIVE_UNPROVEN
    assert fit.classification != FitClass.NATIVE_SAFE  # an estimate is never a proven fit
    assert fit.estimated_peak_bytes and fit.estimated_peak_bytes > 0
    assert fit.attention_path is not None and fit.attention_path.value == "math"
    assert "not measured" in fit.rationale.lower()


def test_within_safety_margin_is_marginal():
    peak = _peak_gb()
    fit = classify_fit(_plan(), _profile(capacity_gb=peak + 0.5))  # < 1.5 GB headroom
    assert fit.classification == FitClass.MARGINAL


def test_over_capacity_on_wddm_is_a_silent_spill():
    peak = _peak_gb()
    fit = classify_fit(_plan(), _profile(capacity_gb=peak - 2.0, residency="wddm"))
    assert fit.classification == FitClass.ACCIDENTAL_WDDM_SPILL
    assert fit.headroom_bytes is not None and fit.headroom_bytes < 0
    assert "spill" in fit.rationale.lower()


def test_over_capacity_on_linux_is_hard_oom():
    peak = _peak_gb()
    fit = classify_fit(_plan(), _profile(capacity_gb=peak - 2.0, residency="linux_dedicated"))
    assert fit.classification == FitClass.FAIL
    assert "oom" in fit.rationale.lower()


def test_over_capacity_on_unified_memory_pages():
    peak = _peak_gb()
    fit = classify_fit(_plan(), _profile(capacity_gb=peak - 2.0, residency="unified_memory"))
    assert fit.classification == FitClass.ACCIDENTAL_UNIFIED_MEMORY_PAGING


# ---- honesty guards ---------------------------------------------------------


def test_never_returns_native_safe():
    for cap in (4.0, 8.0, 16.0, 48.0, 200.0):
        fit = classify_fit(_plan(), _profile(capacity_gb=cap))
        assert fit.classification != FitClass.NATIVE_SAFE


def test_no_gpu_is_native_unproven():
    fit = classify_fit(_plan(), _profile(has_gpu=False))
    assert fit.classification == FitClass.NATIVE_UNPROVEN
    assert "no gpu" in fit.rationale.lower()


def test_cpu_toy_plan_is_not_a_vram_concern():
    fit = classify_fit(_plan(cpu_toy=True), _profile(capacity_gb=8.0))
    assert fit.classification == FitClass.NATIVE_UNPROVEN
    assert "cpu" in fit.rationale.lower()


def test_unparseable_model_name_is_native_unproven():
    fit = classify_fit(_plan(base_model="mystery-model"), _profile(capacity_gb=80.0))
    assert fit.classification == FitClass.NATIVE_UNPROVEN
    assert fit.estimated_peak_bytes is None  # nothing to estimate → no fabricated number


# ---- arithmetic + quantization selection ------------------------------------


def test_headroom_is_capacity_minus_estimated_peak():
    fit = classify_fit(_plan(), _profile(capacity_gb=80.0))
    assert fit.headroom_bytes == fit.device_capacity_bytes - fit.estimated_peak_bytes


def test_unquantized_estimate_is_heavier_than_nf4():
    cap = _profile(capacity_gb=80.0)
    nf4 = classify_fit(_plan(quantization="nf4"), cap)
    fp16 = classify_fit(_plan(quantization="none"), cap)
    assert fp16.estimated_peak_bytes > nf4.estimated_peak_bytes


def test_int8_quant_uses_the_int8_estimate():
    fit = classify_fit(_plan(quantization="int8"), _profile(capacity_gb=80.0))
    expected = int(_peak_gb("int8") * _GB)
    assert fit.estimated_peak_bytes == expected


def test_fp32_precision_is_costed_heavier_than_bf16():
    cap = _profile(capacity_gb=200.0)
    bf16 = classify_fit(_plan(quantization="none", precision="bf16"), cap)
    fp32 = classify_fit(_plan(quantization="none", precision="fp32"), cap)
    # fp32 weights are 2x fp16 → the un-quantized fp32 plan must estimate strictly heavier, so it
    # isn't silently under-costed and wrongly predicted to fit.
    assert fp32.estimated_peak_bytes > bf16.estimated_peak_bytes


def test_gptq_quant_uses_the_int4_tier_not_fp16():
    cap = _profile(capacity_gb=200.0)
    gptq = classify_fit(_plan(quantization="gptq"), cap)
    fp16 = classify_fit(_plan(quantization="none", precision="bf16"), cap)
    # A sub-16-bit scheme must not be sized at the fp16 (2 bytes/param) weight tier.
    assert gptq.estimated_peak_bytes < fp16.estimated_peak_bytes


def test_fit_classification_roundtrips():
    fit = classify_fit(_plan(), _profile(capacity_gb=80.0))
    assert P.FitClassification.model_validate_json(fit.model_dump_json()) == fit
