"""Tests for the environment profiler + functional capability probes (corpus_studio.platform).

Pure — these run in CI without torch. They lock the "readiness = a kernel actually ran, not the
package imports" contract: the profiler builds a deterministic EnvironmentProfile signature and maps
the existing GPU probes into the contract; the probe framework runs (injectable) probes, never lets
one crash the runner, derives effective-capabilities only from PASSes, and degrades to a clean
`not_ready` CapabilityReport when torch is absent (importing nothing heavy).
"""

from __future__ import annotations

import platform as _pf
import sys

import pytest

import corpus_studio.platform as P
from corpus_studio.platform import profiler
from corpus_studio.platform.common import PackageLock
from corpus_studio.platform.contracts import (
    EnvHost,
    EnvironmentProfile,
    ExecutionCapabilityCombination,
    GpuDevice,
)
from corpus_studio.platform.enums import (
    DeviceKind,
    FailureTaxonomy,
    MemoryResidencyModel,
    OperatingSystem,
    PrecisionMode,
)
from corpus_studio.platform.probes import ProbeOutcome, run_capability_probes
from corpus_studio.training.environment import GpuInfo, TrainingRuntimeReport
from corpus_studio.training.gpu_probe import GpuMemory


# ---- import boundary ---------------------------------------------------------


def test_profiler_and_probes_are_torch_free():
    # importing the environment manager must not pull the heavy stack
    assert "torch" not in sys.modules
    assert "bitsandbytes" not in sys.modules


# ---- profiler ----------------------------------------------------------------


def test_build_environment_profile_shape():
    prof = P.build_environment_profile()
    assert isinstance(prof, EnvironmentProfile)
    assert len(prof.environment_signature) == 64  # sha256 hex
    assert prof.host.os in set(OperatingSystem)
    assert len(prof.packages) == len(profiler.PROFILE_PACKAGES)


def test_environment_signature_is_deterministic():
    # volatile fields (free memory, timestamps) are excluded → stable across runs
    assert (
        P.build_environment_profile().environment_signature
        == P.build_environment_profile().environment_signature
    )


def test_os_residency_mapping(monkeypatch):
    for name, os_enum, residency in [
        ("Windows", OperatingSystem.windows, MemoryResidencyModel.wddm),
        ("Linux", OperatingSystem.linux, MemoryResidencyModel.linux_dedicated),
        ("Darwin", OperatingSystem.macos, MemoryResidencyModel.unified_memory),
        ("Plan9", OperatingSystem.unknown, MemoryResidencyModel.unknown),
    ]:
        monkeypatch.setattr(_pf, "system", lambda name=name: name)
        assert profiler._operating_system() == (os_enum, residency)


def test_supported_dtypes_by_capability():
    assert profiler._supported_dtypes(None) == []
    assert PrecisionMode.bf16 not in profiler._supported_dtypes(7)  # Turing: no bf16
    assert PrecisionMode.bf16 in profiler._supported_dtypes(8)  # Ampere+
    assert PrecisionMode.fp8 in profiler._supported_dtypes(9)  # Hopper+
    assert PrecisionMode.fp8 not in profiler._supported_dtypes(8)


def test_gpu_mapping_from_existing_probes(monkeypatch):
    monkeypatch.setattr(
        profiler,
        "probe_training_runtime",
        lambda: TrainingRuntimeReport(
            gpu=GpuInfo(
                available=True,
                device_count=1,
                name="NVIDIA GeForce RTX 5070",
                total_memory_gb=12.0,
                compute_capability="12.0",
            )
        ),
    )
    monkeypatch.setattr(
        profiler,
        "probe_gpu_memory",
        lambda: GpuMemory(total_gb=12.0, free_gb=11.6, compute_capability="12.0"),
    )
    gpus = profiler._gpus()
    assert len(gpus) == 1
    g = gpus[0]
    assert g.kind == DeviceKind.cuda
    assert g.name == "NVIDIA GeForce RTX 5070"
    assert g.compute_capability_major == 12
    assert g.vram_total_bytes == 12_000_000_000
    assert g.vram_free_bytes == 11_600_000_000
    assert PrecisionMode.bf16 in g.supported_dtypes


def test_gpu_mapping_empty_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(profiler, "probe_training_runtime", lambda: TrainingRuntimeReport())
    monkeypatch.setattr(profiler, "probe_gpu_memory", lambda: None)
    assert profiler._gpus() == []


# ---- probe framework (injected fakes) ----------------------------------------


def _profile(*, gpus=None, packages=None, os_enum=OperatingSystem.linux) -> EnvironmentProfile:
    return EnvironmentProfile(
        environment_signature="a" * 64,
        host=EnvHost(os=os_enum),
        gpus=gpus or [],
        packages=packages or [],
    )


def _blackwell_gpu() -> GpuDevice:
    return GpuDevice(
        index=0, kind=DeviceKind.cuda, name="RTX 5070", compute_capability="12.0",
        compute_capability_major=12,
    )


def _execution_combo(*, training: bool) -> ExecutionCapabilityCombination:
    return ExecutionCapabilityCombination.model_validate(
        {
            "runtime_mode": "training" if training else "cpu_toy",
            "device": "cuda" if training else "cpu",
            "precision": "bf16" if training else "fp32",
            "quantization": "nf4" if training else "none",
            "adapter_method": "qlora" if training else "lora",
            "attention_impl": "math" if training else "eager",
            "attention_kernel": "torch_sdpa_math" if training else "eager",
            "optimizer": "adamw_torch",
            "loss_impl": "cross_entropy",
            "checkpoint_impl": "adapter_only",
            "export_format": "adapter_peft",
            "execution_contract_version": "1.0.0",
            "probe": "exact_execution",
        }
    )


def test_run_probes_with_injected_fakes():
    def pass_bf16(_p):
        return ProbeOutcome(FailureTaxonomy.PASS, "ok", proves={"precision": ["bf16"]})

    def stall(_p):
        return ProbeOutcome(FailureTaxonomy.KERNEL_STALL, "sm_120 deadlock")

    def boom(_p):
        raise RuntimeError("kaboom")

    registry = {"bf16": pass_bf16, "flash": stall, "explode": boom}
    report = run_capability_probes(_profile(), registry=registry)

    outcomes = {r.probe: r.outcome for r in report.probe_results}
    assert outcomes["bf16"] == FailureTaxonomy.PASS
    assert outcomes["flash"] == FailureTaxonomy.KERNEL_STALL
    # a probe that raises must NOT crash the runner → ENVIRONMENT_FAILURE
    assert outcomes["explode"] == FailureTaxonomy.ENVIRONMENT_FAILURE
    # effective capabilities come only from the PASS
    assert report.effective_capabilities is not None
    assert report.effective_capabilities.precision_modes == [PrecisionMode.bf16]


def test_effective_capabilities_ignore_non_pass_proves():
    def fail_but_claims(_p):
        return ProbeOutcome(FailureTaxonomy.FAIL, "no", proves={"precision": ["bf16"]})

    report = run_capability_probes(_profile(), registry={"x": fail_but_claims})
    assert report.effective_capabilities is not None
    assert report.effective_capabilities.precision_modes == []


def test_capability_report_rejects_effective_claims_not_in_passing_results():
    from pydantic import ValidationError

    from corpus_studio.platform.contracts import CapabilityReport, EffectiveCapabilities, ProbeResult
    from corpus_studio.platform.common import Ref

    with pytest.raises(ValidationError, match="precision_modes"):
        CapabilityReport(
            backend_id="corpus_studio",
            environment_ref=Ref(id="environment"),
            readiness="not_ready",
            probe_results=[
                ProbeResult(
                    probe="precision",
                    outcome=FailureTaxonomy.PASS,
                    proves={"precision": ["fp32"]},
                )
            ],
            effective_capabilities=EffectiveCapabilities(precision_modes=["bf16"]),
        )


def test_execution_contract_evidence_requires_trainer_surface_and_exact_tuple():
    def passing(_p):
        return ProbeOutcome(FailureTaxonomy.PASS)
    combo = _execution_combo(training=False)
    registry = {
        "trainer_contract": passing,
        "exact_execution": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS,
            execution_combinations=[combo],
        ),
    }
    report = run_capability_probes(_profile(), registry=registry)
    assert report.effective_capabilities is not None
    assert report.effective_capabilities.execution_contract_versions == ["1.0.0"]

    registry["exact_execution"] = lambda _p: ProbeOutcome(FailureTaxonomy.NUMERICAL_FAILURE)
    failed = run_capability_probes(_profile(), registry=registry)
    assert failed.effective_capabilities is not None
    assert failed.effective_capabilities.execution_contract_versions == []


def test_effective_capabilities_carry_only_probed_physical_tokens():
    def physical_probe(_p):
        return ProbeOutcome(
            FailureTaxonomy.PASS,
            "physical path exercised",
            proves={
                "offload": ["controlled_parameter_offload"],
                "placement_tier": ["gpu", "nvme"],
                "placement_mode": ["tiered"],
                "parallelism": ["expert"],
                "communication_backend": ["nccl"],
            },
        )

    report = run_capability_probes(_profile(), registry={"physical": physical_probe})
    effective = report.effective_capabilities
    assert effective is not None
    assert [item.value for item in effective.offload_strategies] == [
        "controlled_parameter_offload"
    ]
    assert [item.value for item in effective.placement_tiers] == ["gpu", "nvme"]
    assert [item.value for item in effective.placement_modes] == ["tiered"]
    assert [item.value for item in effective.parallelism_kinds] == ["expert"]
    assert [item.value for item in effective.communication_backends] == ["nccl"]


def test_unknown_probe_name_is_unsupported():
    report = run_capability_probes(_profile(), probes=["does_not_exist"], registry={})
    assert report.probe_results[0].outcome == FailureTaxonomy.UNSUPPORTED_CONFIGURATION


def test_readiness_ready_only_when_a_complete_cuda_tuple_passes():
    combo = _execution_combo(training=True)
    reg = {
        "cuda_available": lambda _p: ProbeOutcome(FailureTaxonomy.PASS),
        "bnb_4bit_load": lambda _p: ProbeOutcome(FailureTaxonomy.PASS),
        "exact_execution": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS,
            execution_combinations=[combo],
        ),
    }
    report = run_capability_probes(
        _profile(packages=[PackageLock(name="torch", version="2.7.0")]), registry=reg
    )
    assert report.readiness == "ready"
    assert report.bitsandbytes_ok is True


def test_independent_axis_probes_cannot_be_unioned_into_complete_support():
    registry = {
        "precision": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS, proves={"precision": ["bf16"]}
        ),
        "quantization": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS, proves={"quantization": ["nf4"]}
        ),
        "adapter": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS, proves={"adapter": ["qlora"]}
        ),
        "optimizer": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS, proves={"optimizer": ["adamw_torch"]}
        ),
    }
    report = run_capability_probes(_profile(), registry=registry)
    assert report.readiness == "not_ready"
    assert report.effective_capabilities is not None
    assert report.effective_capabilities.execution_combinations == []


def test_readiness_cpu_toy_only_when_a_complete_cpu_tuple_passes():
    combo = _execution_combo(training=False)
    reg = {
        "cuda_available": lambda _p: ProbeOutcome(FailureTaxonomy.FAIL, "cpu build"),
        "exact_execution": lambda _p: ProbeOutcome(
            FailureTaxonomy.PASS,
            execution_combinations=[combo],
        ),
    }
    report = run_capability_probes(
        _profile(packages=[PackageLock(name="torch", version="2.7.0+cpu")]), registry=reg
    )
    assert report.readiness == "cpu_toy_only"
    assert "torch" not in report.missing_packages  # installed, just no GPU


def test_readiness_not_ready_when_torch_absent():
    reg = {"cuda_available": lambda _p: ProbeOutcome(FailureTaxonomy.ENVIRONMENT_FAILURE, "no torch")}
    report = run_capability_probes(_profile(packages=[PackageLock(name="torch", version=None)]),
                                   registry=reg)
    assert report.readiness == "not_ready"
    assert "torch" in report.missing_packages


# ---- built-in probes (behavior provable without torch) -----------------------


def test_flash_probe_short_circuits_kernel_stall_on_native_windows_blackwell():
    from corpus_studio.platform.probes import _probe_flash_attn_backward

    # NATIVE WINDOWS + Blackwell (WDDM): the flash backward would deadlock, so the probe reports the
    # hazard WITHOUT executing/importing torch.
    out = _probe_flash_attn_backward(_profile(gpus=[_blackwell_gpu()], os_enum=OperatingSystem.windows))
    assert out.taxonomy == FailureTaxonomy.KERNEL_STALL
    assert "sm_120" in (out.detail or "")
    assert "torch" not in sys.modules  # the hazard is reported without executing/importing torch


def test_flash_probe_executes_on_wsl_blackwell_not_short_circuited():
    from corpus_studio.platform.probes import _probe_flash_attn_backward

    # WSL Blackwell: the deadlock does NOT apply, so the probe must NOT short-circuit — it actually
    # attempts the flash backward (on a real WSL GPU it PASSes and proves flash/sdpa; here in the
    # torch-free gate it degrades to ENVIRONMENT_FAILURE). The key invariant: it is NOT a KERNEL_STALL,
    # so a WSL host can prove flash and the planner can seal sdpa.
    out = _probe_flash_attn_backward(_profile(gpus=[_blackwell_gpu()], os_enum=OperatingSystem.wsl))
    assert out.taxonomy != FailureTaxonomy.KERNEL_STALL


def test_gpu_responsive_probe_flags_a_wedged_gpu(monkeypatch):
    from corpus_studio.platform import probes

    # A wedged GPU (the WSL2 GPU-PV state a crashed run leaves) → ENVIRONMENT_FAILURE with the
    # OS-specific reset instruction, not a generic failure the operator can't act on.
    monkeypatch.setattr(probes, "probe_gpu_responsive", lambda: "CUDA error: device not ready")
    out = probes._probe_gpu_responsive(_profile(os_enum=OperatingSystem.wsl))
    assert out.taxonomy == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert "wsl --terminate" in (out.detail or "")


def test_gpu_responsive_probe_passes_on_a_live_gpu(monkeypatch):
    from corpus_studio.platform import probes

    monkeypatch.setattr(probes, "probe_gpu_responsive", lambda: None)
    assert probes._probe_gpu_responsive(_profile()).taxonomy == FailureTaxonomy.PASS


def test_gpu_responsive_probe_absent_is_not_a_wedge(monkeypatch):
    from corpus_studio.platform import probes

    # No GPU is UNSUPPORTED (nothing to reset), never a false wedge that tells CI to reset a GPU.
    monkeypatch.setattr(probes, "probe_gpu_responsive", lambda: "no CUDA GPU available")
    assert probes._probe_gpu_responsive(_profile()).taxonomy == FailureTaxonomy.UNSUPPORTED_CONFIGURATION


def test_builtin_probes_degrade_cleanly_without_torch():
    # the real registry against a real profile in a torch-absent venv → not_ready, no crash
    profile = P.build_environment_profile()
    report = run_capability_probes(profile)
    assert report.readiness == "not_ready"
    torch_probes = {
        "cuda_available",
        "bf16_matmul",
        "bnb_4bit_load",
        "checkpoint_reload",
        "dense_optimizer_step",
    }
    outcomes = {r.probe: r.outcome for r in report.probe_results}
    for name in torch_probes:
        assert outcomes[name] == FailureTaxonomy.ENVIRONMENT_FAILURE
    assert "torch" not in sys.modules
