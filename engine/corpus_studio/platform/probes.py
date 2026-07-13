"""Functional capability probes — readiness means a kernel actually RAN, not "the package imports".

Each probe executes a tiny real operation (a bf16 matmul, a 4-bit load, a flash-attention backward, a
checkpoint round-trip) and returns a :class:`~corpus_studio.platform.enums.FailureTaxonomy` outcome. A
probe that PASSES contributes to the ``effective_capabilities`` (what actually works on THIS host),
which is what the planner should resolve a RunPlan against — not a backend's static claims.

Dependency-light: this module imports NO torch at load time. Every torch/bitsandbytes import is lazy,
inside a probe body, so a core-only install still runs the framework and reports each hardware probe as
``ENVIRONMENT_FAILURE`` (→ ``readiness = not_ready``) instead of crashing.

The one probe that must not actually execute on **native-Windows** Blackwell sm_120 is
``flash_attn_backward``: the fused flash SDPA backward deadlocks on the first backward under the
Windows WDDM driver (documented in ``training/environment.py``), so there it short-circuits to
``KERNEL_STALL`` rather than hanging the probe process. Outside native Windows, the known WDDM refusal
does not apply, so the probe executes and must itself PASS before it proves flash/sdpa. WSL has
separately labeled passing evidence; bare-Linux RTX 5070 behavior remains unverified.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .common import PackageLock, Ref
from .contracts import CapabilityReport, EffectiveCapabilities, EnvironmentProfile, ProbeResult
from .enums import (
    AdapterMethod,
    AttentionImpl,
    CommunicationBackend,
    FailureTaxonomy,
    MemoryTier,
    OffloadStrategy,
    ParallelismKind,
    PlacementMode,
    PrecisionMode,
    QuantizationMode,
)
from .gpu_health import classify_gpu_health, probe_gpu_responsive, wedged_gpu_remediation
from .host_platform import flash_sdpa_deadlocks

_TX = FailureTaxonomy

# training-stack distributions that gate readiness (a subset of the profile's package list).
_TRAIN_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")


@dataclass
class ProbeOutcome:
    """A probe's result. ``proves`` maps a capability axis (precision/quantization/attention/adapter)
    to the concrete tokens this probe demonstrated when it PASSED — the input to
    ``effective_capabilities``."""

    taxonomy: FailureTaxonomy
    detail: str | None = None
    measured: dict = field(default_factory=dict)
    proves: dict[str, list[str]] = field(default_factory=dict)


ProbeFn = Callable[[EnvironmentProfile], ProbeOutcome]


def _max_cc_major(profile: EnvironmentProfile) -> int:
    return max((g.compute_capability_major or 0 for g in profile.gpus), default=0)


# --- built-in probes ------------------------------------------------------------------------------


def _probe_cuda_available(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    try:
        if torch.cuda.is_available():
            return ProbeOutcome(_TX.PASS, f"{torch.cuda.device_count()} CUDA device(s)")
        return ProbeOutcome(_TX.FAIL, "torch present but no CUDA device (CPU build or no GPU)")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_bf16_matmul(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
        b = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
        finite = bool(torch.isfinite(a @ b).all().item())
        if finite:
            return ProbeOutcome(
                _TX.PASS, f"bf16 matmul on {device}", proves={"precision": ["bf16"]}
            )
        return ProbeOutcome(_TX.NUMERICAL_FAILURE, "bf16 matmul produced non-finite values")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_bnb_4bit_load(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import bitsandbytes  # noqa: F401,PLC0415
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"bitsandbytes/torch not importable: {exc}")
    try:
        if not torch.cuda.is_available():
            return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "bitsandbytes 4-bit requires CUDA")
        from bitsandbytes.nn import Linear4bit  # noqa: PLC0415

        layer = Linear4bit(16, 16, bias=False).cuda()
        out = layer(torch.randn(2, 16, device="cuda", dtype=torch.float16))
        finite = bool(torch.isfinite(out).all().item())
        if finite:
            return ProbeOutcome(
                _TX.PASS,
                "Linear4bit forward ok",
                proves={"quantization": ["nf4", "int4"], "adapter": ["qlora"]},
            )
        return ProbeOutcome(_TX.NUMERICAL_FAILURE, "Linear4bit produced non-finite values")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_flash_attn_backward(profile: EnvironmentProfile) -> ProbeOutcome:
    # Known-hazard short-circuit: the fused flash SDPA backward deadlocks on Blackwell sm_120 ONLY
    # under the native-Windows WDDM driver. Report it WITHOUT executing THERE so the probe never hangs.
    # Outside native Windows the known WDDM refusal does not apply, so the probe executes. Only its
    # PASS result proves flash/sdpa on that exact host. WSL evidence does not prove bare-Linux behavior.
    if flash_sdpa_deadlocks(profile.host.os, _max_cc_major(profile)):
        return ProbeOutcome(
            _TX.KERNEL_STALL,
            "native Windows + sm_120 (Blackwell): the fused flash SDPA backward deadlocks under the "
            "Windows WDDM driver — not executed to avoid hanging the probe; use math/eager SDPA, or "
            "use a non-WDDM host only after its flash capability probe passes.",
        )
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as functional  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    if not torch.cuda.is_available():
        return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, "no CUDA GPU for a flash-attention probe")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = functional.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        return ProbeOutcome(
            _TX.PASS,
            "flash SDPA forward+backward ok",
            proves={"attention": ["flash_attention_2", "sdpa"]},
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.FAIL, str(exc))


def _probe_checkpoint_reload(profile: EnvironmentProfile) -> ProbeOutcome:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"torch not importable: {exc}")
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    try:
        tensor = torch.randn(4, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            torch.save({"w": tensor}, path)
            # weights_only=True: defensive default even though this file is the probe's own tensor.
            loaded = torch.load(path, map_location="cpu", weights_only=True)["w"]
        if bool(torch.equal(tensor, loaded)):
            return ProbeOutcome(_TX.PASS, "checkpoint save/reload round-trip ok")
        return ProbeOutcome(_TX.CHECKPOINT_FAILURE, "reloaded tensor differs from saved")
    except Exception as exc:  # noqa: BLE001
        return ProbeOutcome(_TX.CHECKPOINT_FAILURE, str(exc))


def _probe_gpu_responsive(profile: EnvironmentProfile) -> ProbeOutcome:
    # Detect a WEDGED GPU up front — the WSL2 GPU-PV state a crashed run leaves behind, where every
    # subsequent process fails with 'device not ready' regardless of config. Diagnosing it here (with
    # the OS-specific reset instruction) turns a cascade of cryptic failures into one clear "reset your
    # GPU" message. Runs first so a wedge is caught before the heavier probes hit the same wall.
    error = probe_gpu_responsive()
    health = classify_gpu_health(error)
    if health == "healthy":
        return ProbeOutcome(_TX.PASS, "GPU responds to a tiny CUDA op (not wedged)")
    if health == "absent":
        return ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, error or "no CUDA GPU for a health probe")
    if health == "wedged":
        return ProbeOutcome(_TX.ENVIRONMENT_FAILURE, wedged_gpu_remediation(profile.host.os))
    return ProbeOutcome(_TX.FAIL, f"GPU health probe returned an unclassified error: {error}")


BUILTIN_PROBES: dict[str, ProbeFn] = {
    "gpu_responsive": _probe_gpu_responsive,
    "cuda_available": _probe_cuda_available,
    "bf16_matmul": _probe_bf16_matmul,
    "bnb_4bit_load": _probe_bnb_4bit_load,
    "flash_attn_backward": _probe_flash_attn_backward,
    "checkpoint_reload": _probe_checkpoint_reload,
}


# --- runner ---------------------------------------------------------------------------------------


def _resolve_readiness(
    by_probe: dict[str, FailureTaxonomy], profile: EnvironmentProfile
) -> Literal["ready", "cpu_toy_only", "not_ready"]:
    """ready = a CUDA + 4-bit path proved out; cpu_toy_only = torch is installed but no GPU/4-bit
    path proved; not_ready = torch absent / nothing usable."""
    torch_installed = any(
        p.name == "torch" and p.version is not None for p in profile.packages
    )
    cuda_ok = by_probe.get("cuda_available") == FailureTaxonomy.PASS
    bnb_ok = by_probe.get("bnb_4bit_load") == FailureTaxonomy.PASS
    if cuda_ok and bnb_ok:
        return "ready"
    if torch_installed or cuda_ok:
        return "cpu_toy_only"
    return "not_ready"


def _effective(proven: dict[str, set[str]]) -> EffectiveCapabilities:
    return EffectiveCapabilities(
        precision_modes=[PrecisionMode(v) for v in sorted(proven.get("precision", set()))],
        quantization_modes=[
            QuantizationMode(v) for v in sorted(proven.get("quantization", set()))
        ],
        attention_impls=[AttentionImpl(v) for v in sorted(proven.get("attention", set()))],
        adapter_methods=[AdapterMethod(v) for v in sorted(proven.get("adapter", set()))],
        offload_strategies=[
            OffloadStrategy(v) for v in sorted(proven.get("offload", set()))
        ],
        placement_tiers=[
            MemoryTier(v) for v in sorted(proven.get("placement_tier", set()))
        ],
        placement_modes=[
            PlacementMode(v) for v in sorted(proven.get("placement_mode", set()))
        ],
        parallelism_kinds=[
            ParallelismKind(v) for v in sorted(proven.get("parallelism", set()))
        ],
        communication_backends=[
            CommunicationBackend(v)
            for v in sorted(proven.get("communication_backend", set()))
        ],
    )


def run_capability_probes(
    profile: EnvironmentProfile,
    *,
    backend_id: str = "corpus_studio",
    backend_version: str | None = None,
    probes: Sequence[str] | None = None,
    registry: dict[str, ProbeFn] | None = None,
) -> CapabilityReport:
    """Run the requested probes against ``profile`` and build a :class:`CapabilityReport`.

    A probe is never allowed to crash the runner — any exception becomes an ``ENVIRONMENT_FAILURE``
    result. ``registry`` (defaulting to :data:`BUILTIN_PROBES`) is injectable so the framework can be
    unit-tested with fakes. ``effective_capabilities`` is the union of what the PASSED probes proved on
    this host — the intersection with a backend's declared surface belongs to the planner.
    """
    reg = registry if registry is not None else BUILTIN_PROBES
    names: Iterable[str] = probes if probes is not None else list(reg)

    results: list[ProbeResult] = []
    by_probe: dict[str, FailureTaxonomy] = {}
    proven: dict[str, set[str]] = {
        "precision": set(),
        "quantization": set(),
        "attention": set(),
        "adapter": set(),
        "offload": set(),
        "placement_tier": set(),
        "placement_mode": set(),
        "parallelism": set(),
        "communication_backend": set(),
    }
    for name in names:
        fn = reg.get(name)
        if fn is None:
            outcome = ProbeOutcome(_TX.UNSUPPORTED_CONFIGURATION, f"unknown probe '{name}'")
        else:
            try:
                outcome = fn(profile)
            except Exception as exc:  # noqa: BLE001 - a probe must never crash the runner.
                outcome = ProbeOutcome(_TX.ENVIRONMENT_FAILURE, f"probe raised: {exc}")
        results.append(
            ProbeResult(
                probe=name, outcome=outcome.taxonomy, detail=outcome.detail, measured=outcome.measured
            )
        )
        by_probe[name] = outcome.taxonomy
        if outcome.taxonomy == _TX.PASS:
            for axis, tokens in outcome.proves.items():
                proven.setdefault(axis, set()).update(tokens)

    installed = [p for p in profile.packages if p.version is not None]
    missing = [p.name for p in profile.packages if p.version is None and p.name in _TRAIN_PACKAGES]
    return CapabilityReport(
        backend_id=backend_id,
        backend_version=backend_version,
        environment_ref=Ref(id=profile.environment_signature),
        generated_at=datetime.now(timezone.utc).isoformat(),
        readiness=_resolve_readiness(by_probe, profile),
        bitsandbytes_ok=by_probe.get("bnb_4bit_load") == _TX.PASS,
        installed_packages=[PackageLock(name=p.name, version=p.version) for p in installed],
        missing_packages=missing,
        probe_results=results,
        effective_capabilities=_effective(proven),
    )
