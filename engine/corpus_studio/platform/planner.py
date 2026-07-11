"""The run planner — platform slice 6.

The missing verb in the goal + data + hardware → runnable-plan loop: ``profiler`` describes the host
(:class:`EnvironmentProfile`), ``probes`` proves what actually works on it (:class:`CapabilityReport`),
and the ``supervisor`` / ``TrainingRunner`` execute a plan — but a :class:`RunPlan` had to be
hand-authored until now. :func:`build_run_plan` composes those inputs plus a small
:class:`PlannerConstraints` (the user intent the host can't decide) into ONE valid, immutable,
``plan_hash``-sealed RunPlan, resolving every ambiguous field AHEAD OF TIME against what PROVED to
work on this host — the runtime decisions ``training.trainer.resolve_run_plan`` /
``resolve_attention_implementation`` make late, moved forward.

Honesty non-negotiables baked in:
* Blackwell (GPU ``compute_capability_major >= 12``) forces ``attention_backend = math`` — asserted
  from the profile, independent of probe output (the flash probe short-circuits to KERNEL_STALL on
  sm_120 without executing, so flash is correctly absent from the proven set).
* Nothing is claimed that wasn't PROVEN: ``bf16`` only when it's in the effective precision modes;
  ``nf4`` only when bitsandbytes passed; resolve against ``effective_capabilities``, never the raw
  profile or a backend's declared surface.
* ``sequence_len`` flows from the constraints — never a hardcoded calibration value.
* ``cpu_toy`` is never a silent downgrade of a real-training intent: an unready host raises
  :class:`PlannerError` unless cpu-toy was explicitly requested.
* A *planned* fit is NOT a proven fit — this slice picks a VALID plan; whether it FITS the VRAM is a
  separate calibrator/measured-run concern and is deliberately not asserted here.

Dependency-light: stdlib + platform contracts only, no torch at module load.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ValidationError

from corpus_studio.platform.contracts import (
    CapabilityReport,
    EnvironmentProfile,
    RunPlan,
)
from corpus_studio.platform.common import Ref
from corpus_studio.platform.enums import (
    AdapterMethod,
    AttentionImpl,
    ExportFormat,
    TaskType,
)

# attn_implementation strings the trainer passes to from_pretrained. math / sdpa / mem_efficient /
# xformers are NOT from_pretrained values (they are SDPA backends toggled inside the trainer), so we
# leave attn_implementation unset for those and let the trainer's own proven Blackwell path fire.
_EXPLICIT_ATTN = frozenset({"eager", "flash_attention_2", "flash_attention_3"})
_LORA_FAMILY = frozenset({"lora", "qlora", "dora"})
_BLACKWELL_MAJOR = 12
_DEFAULT_BACKEND = "corpus_studio"


class PlannerError(Exception):
    """A request the host cannot honor — the ahead-of-time twin of ``trainer.TrainerError`` (not
    ready, cpu-toy-only without opt-in, or an unsupported constraint)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PlannerConstraints:
    """The user intent the host can't decide — everything else the planner resolves from the
    environment + proven capabilities. ``sequence_len`` etc. flow into the plan verbatim (never a
    hardcoded calibration value)."""

    base_model: str
    dataset_path: str
    task_type: str = "sft"
    dataset_format: str = "instruction"
    adapter_method: str | None = None  # None → auto: qlora when quantized, else lora
    lora_r: int = 16
    lora_alpha: int = 32
    sequence_len: int = 4096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    seed: int = 42
    output_dir: str = "output"
    supervised_token_accumulation_target: int | None = None
    attention_backend: str | None = None  # explicit override; else resolved from the host
    export_format: str = "adapter_peft"
    allow_cpu_toy: bool = False


def _require_enum(value: str, enum_cls: type[Enum], label: str) -> None:
    valid = {member.value for member in enum_cls}
    if value not in valid:
        raise PlannerError(
            f"unsupported {label} '{value}'; expected one of: {', '.join(sorted(valid))}"
        )


def _max_cc_major(profile: EnvironmentProfile) -> int | None:
    majors = [g.compute_capability_major for g in profile.gpus if g.compute_capability_major is not None]
    return max(majors) if majors else None


def _resolve_attention(explicit: str | None, cc_major: int | None, proven_attn: set[str]) -> str:
    if explicit is not None:
        _require_enum(explicit, AttentionImpl, "attention_backend")
        return explicit
    if cc_major is not None and cc_major >= _BLACKWELL_MAJOR:
        return AttentionImpl.math.value  # Blackwell mandate — asserted, not probe-derived
    if AttentionImpl.sdpa.value in proven_attn:
        return AttentionImpl.sdpa.value
    return AttentionImpl.eager.value  # universal safe fallback


def compute_plan_hash(plan_body: Mapping[str, Any]) -> str:
    """The immutability seal: sha256 over the canonicalized plan body. Mirrors
    ``profiler._environment_signature`` (compact, key-sorted JSON) — the engine-wide content-identity
    convention. The caller MUST exclude ``plan_hash`` (a hash can't include itself) and ``created_at``
    (volatile) so two byte-identical plans minted at different instants seal to the same hash."""
    canonical = json.dumps(plan_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_run_plan(
    *,
    profile: EnvironmentProfile,
    capabilities: CapabilityReport,
    dataset_ref: Ref,
    constraints: PlannerConstraints,
    plan_id: str,
    now: str | None = None,
) -> RunPlan:
    """Resolve one immutable, hash-sealed :class:`RunPlan` from the host profile + proven
    capabilities + dataset + user constraints. Raises :class:`PlannerError` when the host can't honor
    the request (not ready; cpu-toy-only without ``allow_cpu_toy``; an unsupported constraint)."""
    _require_enum(constraints.task_type, TaskType, "task_type")
    _require_enum(constraints.export_format, ExportFormat, "export_format")

    effective = capabilities.effective_capabilities
    proven_precisions = {p.value for p in effective.precision_modes} if effective else set()
    proven_attn = {a.value for a in effective.attention_impls} if effective else set()
    cc_major = _max_cc_major(profile)

    # --- run mode (honest, readiness-driven) ---
    if capabilities.readiness == "ready":
        cpu_toy = False
    elif capabilities.readiness == "cpu_toy_only":
        if not constraints.allow_cpu_toy:
            raise PlannerError(
                "only the CPU-toy smoke path is available on this host; pass allow_cpu_toy to plan "
                "it, or provision a GPU + [train] runtime for a real run."
            )
        cpu_toy = True
    else:  # not_ready
        missing = ", ".join(capabilities.missing_packages) or "the training runtime"
        raise PlannerError(
            f"the environment is not ready for training (missing: {missing}); "
            "run 'corpus-studio train-check' to see what's needed."
        )

    # --- resolve the ambiguous fields against PROVEN capabilities ---
    if cpu_toy:
        precision = "fp32"
        quantization = "none"
        attention_backend = AttentionImpl.eager.value
    else:
        precision = "bf16" if "bf16" in proven_precisions else "fp32"
        quantization = "nf4" if capabilities.bitsandbytes_ok else "none"
        attention_backend = _resolve_attention(constraints.attention_backend, cc_major, proven_attn)

    adapter_method = constraints.adapter_method or ("qlora" if quantization == "nf4" else "lora")
    _require_enum(adapter_method, AdapterMethod, "adapter_method")

    token_target = constraints.supervised_token_accumulation_target or max(
        1, constraints.sequence_len * constraints.micro_batch_size * constraints.gradient_accumulation_steps
    )

    # attn_implementation string only for real from_pretrained backends; math/sdpa → unset so the
    # trainer's own Blackwell-safe path stays in control.
    snapshot: dict[str, Any] = {
        "base_model": constraints.base_model,
        "dataset_path": constraints.dataset_path,
        "output_dir": constraints.output_dir,
        "dataset_format": constraints.dataset_format,  # trainer field name (NOT "format")
        "sequence_len": constraints.sequence_len,
        "lora_r": constraints.lora_r,
        "lora_alpha": constraints.lora_alpha,
        "micro_batch_size": constraints.micro_batch_size,
        "gradient_accumulation_steps": constraints.gradient_accumulation_steps,
        "learning_rate": constraints.learning_rate,
        "seed": constraints.seed,
    }
    if attention_backend in _EXPLICIT_ATTN:
        snapshot["attn_implementation"] = attention_backend
    if cpu_toy:
        snapshot["cpu_toy"] = True

    adapter: dict[str, Any] = {"method": adapter_method}
    if adapter_method in _LORA_FAMILY:
        adapter["lora_r"] = constraints.lora_r
        adapter["lora_alpha"] = constraints.lora_alpha

    body: dict[str, Any] = {
        "plan_id": plan_id,
        "plan_hash": "0" * 64,  # placeholder — replaced by the real seal below
        "backend_ref": {"id": _DEFAULT_BACKEND},
        "environment_ref": {"id": profile.environment_signature},
        "dataset_ref": dataset_ref.model_dump(mode="json"),
        "task_type": constraints.task_type,
        "base_model": constraints.base_model,
        "precision": precision,
        "quantization": quantization,
        "adapter": adapter,
        "optimizer": {"impl": "adamw_torch", "learning_rate": constraints.learning_rate},
        "loss_impl": "cross_entropy",
        "attention_backend": attention_backend,
        "sequence": {"max_sequence_len": constraints.sequence_len},
        "batching": {
            "micro_batch_size": constraints.micro_batch_size,
            "supervised_token_accumulation_target": token_target,
            "fallback_grad_accumulation_steps": constraints.gradient_accumulation_steps,
        },
        "checkpoint_policy": {"impl": "adapter_only"},
        "gradient_checkpointing": True,
        "export": {"format": constraints.export_format, "output_dir": constraints.output_dir},
        "seed": constraints.seed,
        "training_config_snapshot": snapshot,
    }

    try:
        draft = RunPlan.model_validate({**body, "created_at": None})
    except ValidationError as exc:
        raise PlannerError(f"the resolved plan is invalid: {exc}") from exc

    # Seal over the FULLY-DEFAULTED canonical plan, excluding the seal itself + the volatile stamp.
    plan_hash = compute_plan_hash(draft.model_dump(mode="json", exclude={"plan_hash", "created_at"}))
    return draft.model_copy(update={"plan_hash": plan_hash, "created_at": now or _now_iso()})
