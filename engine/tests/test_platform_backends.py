"""Multi-backend registry — the 'pick your framework' substrate. Pure tests (no torch): every backend
declares its surface honestly, and selection resolves a plan's requirements against it. The load-
bearing honesty case: Unsloth (flash/sdpa only) is filtered OUT of a Blackwell math plan."""

import corpus_studio.platform as P
from corpus_studio.platform.backends import (
    builtin_backends,
    compatible_backends,
    get_backend,
    unmet_requirements,
)

_BLACKWELL = dict(
    os="linux", device="cuda", task_type="sft", precision="bf16", quantization="nf4",
    adapter_method="qlora", attention="math",
)
_AMPERE = {**_BLACKWELL, "attention": "sdpa"}


def test_builtin_backends_roundtrip():
    ids = [b.backend_id for b in builtin_backends()]
    assert ids == ["corpus_studio", "unsloth"]
    for backend in builtin_backends():
        assert P.BackendManifest.model_validate_json(backend.model_dump_json()) == backend


def test_get_backend():
    assert get_backend("corpus_studio").backend_id == "corpus_studio"
    assert get_backend("unsloth").display_name.startswith("Unsloth")
    assert get_backend("megatron") is None


def test_corpus_studio_runs_a_blackwell_math_plan():
    assert unmet_requirements(get_backend("corpus_studio"), **_BLACKWELL) == []


def test_unsloth_refuses_the_math_attention_a_blackwell_plan_needs():
    reasons = unmet_requirements(get_backend("unsloth"), **_BLACKWELL)
    assert any("attention 'math'" in r for r in reasons)


def test_unsloth_has_no_cpu_path():
    reasons = unmet_requirements(get_backend("unsloth"), **{**_AMPERE, "device": "cpu"})
    assert any("device 'cpu'" in r for r in reasons)


def test_unmet_reports_every_unsupported_field():
    reasons = unmet_requirements(
        get_backend("unsloth"),
        os="macos", device="cpu", task_type="reward", precision="fp8",
        quantization="gptq", adapter_method="dora", attention="xformers",
    )
    # OS, device, task, precision, quant, adapter, attention — all seven are unsupported.
    assert len(reasons) == 7


def test_compatible_backends_for_a_math_plan_is_corpus_only():
    ids = [b.backend_id for b in compatible_backends(**_BLACKWELL)]
    assert ids == ["corpus_studio"]  # Unsloth can't do math → routed away from Blackwell


def test_compatible_backends_for_an_sdpa_plan_includes_unsloth():
    ids = [b.backend_id for b in compatible_backends(**_AMPERE)]
    assert set(ids) == {"corpus_studio", "unsloth"}


def test_no_backend_fits_an_impossible_configuration():
    impossible = {**_AMPERE, "adapter_method": "full_finetune"}
    assert compatible_backends(**impossible) == []
