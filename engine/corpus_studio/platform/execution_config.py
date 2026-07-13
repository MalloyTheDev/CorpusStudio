"""Hashing and immutable-input checks for resolved execution configurations.

Pure control-plane code: no torch, Transformers, or network access. The same stable-read functions
are used at planning and immediately before worker execution so a mutable path cannot retain the
identity of different bytes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from .common import HashRef, Ref
from .contracts import (
    CapabilityReport,
    ExecutionInputBinding,
    ResolvedExecutionConfiguration,
    RunPlan,
)

_IGNORED_DIRECTORIES = {".git", "__pycache__"}
_FORMATTER_IDENTITIES = {
    "instruction": "corpus-studio:instruction-alpaca-v1",
    "chat": "corpus-studio:tokenizer-chat-template-v1",
    "trace": "corpus-studio:structured-trace-renderer-v1",
}
_RUNTIME_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class ExecutionConfigurationError(ValueError):
    """A seal or immutable-input check failed before model loading."""


def run_scoped_training_output(
    config: ResolvedExecutionConfiguration,
    run_id: str,
) -> Path:
    """Resolve the final trainer directory from the sealed root/layout and fresh run identity."""

    if config.output_layout != "run_scoped_v1":  # pragma: no cover - literal contract defense
        raise ExecutionConfigurationError(
            f"unsupported resolved output layout {config.output_layout!r}"
        )
    if not _RUNTIME_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ExecutionConfigurationError("run_id is unsafe for run-scoped output resolution")
    return Path(config.output_dir) / "runs" / run_id / "artifacts" / "adapter"


def required_runner_lane(plan: RunPlan) -> str:
    """Return the only runner lane allowed to consume ``plan``."""

    execution = plan.resolved_execution
    if execution is not None:
        if plan.backend_ref.id != "corpus_studio":
            raise ExecutionConfigurationError(
                "resolved training plans require the first-party corpus_studio worker"
            )
        return "cpu_toy" if execution.runtime_mode == "cpu_toy" else "training"
    if plan.backend_ref.id == "echo":
        if plan.task_type.value != "evaluation":
            raise ExecutionConfigurationError(
                "the echo backend is restricted to explicitly non-training evaluation plans"
            )
        return "echo"
    raise ExecutionConfigurationError(
        "a non-echo plan without ResolvedExecutionConfiguration has no executable runner lane"
    )


def verify_runner_lane(plan: RunPlan, runner_name: str) -> None:
    expected = required_runner_lane(plan)
    if runner_name != expected:
        raise ExecutionConfigurationError(
            f"runner lane {runner_name!r} cannot execute this plan; sealed lane is {expected!r}"
        )


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execution_configuration_hash_for(config: ResolvedExecutionConfiguration) -> str:
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_execution_configuration_hash(config: ResolvedExecutionConfiguration) -> bool:
    return config.configuration_hash == execution_configuration_hash_for(config)


def capability_report_hash_for(report: CapabilityReport) -> str:
    return canonical_sha256(report.model_dump(mode="json"))


def capability_report_ref_for(report: CapabilityReport) -> Ref:
    digest = capability_report_hash_for(report)
    return Ref(
        id=f"capability-{report.backend_id}-{digest[:12]}",
        hash=HashRef(value=digest),
    )


def formatter_identity(dataset_format: str) -> tuple[str, str]:
    try:
        formatter_id = _FORMATTER_IDENTITIES[dataset_format]
    except KeyError as exc:
        raise ExecutionConfigurationError(
            f"no sealed formatter exists for dataset format {dataset_format!r}"
        ) from exc
    try:
        from corpus_studio.training.trainer import format_example_text  # noqa: PLC0415

        sources = [inspect.getsource(format_example_text)]
        if dataset_format == "trace":
            from corpus_studio.training.traces import (  # noqa: PLC0415
                format_trace,
                trace_from_row,
            )

            sources.extend((inspect.getsource(trace_from_row), inspect.getsource(format_trace)))
    except (ImportError, OSError, TypeError) as exc:
        raise ExecutionConfigurationError(
            f"cannot inspect the sealed formatter implementation for {dataset_format!r}: {exc}"
        ) from exc
    return formatter_id, canonical_sha256({"formatter_id": formatter_id, "sources": sources})


def huggingface_input_ref(kind: str, repository: str, revision: str) -> Ref:
    digest = hashlib.sha256(
        f"huggingface:{kind}:{repository}@{revision}".encode("utf-8")
    ).hexdigest()
    return Ref(id=f"{kind}-{digest[:12]}", hash=HashRef(value=digest))


def _is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _stable_file_read(path: str | Path, *, capture: bool) -> tuple[bytes | None, str]:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        raise ExecutionConfigurationError(f"execution input file does not exist: {candidate}")
    if _is_link_like(candidate):
        raise ExecutionConfigurationError(f"execution input cannot be a link: {candidate}")
    digest = hashlib.sha256()
    captured = bytearray() if capture else None
    try:
        before = candidate.stat()
        with candidate.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened_before.st_dev, opened_before.st_ino):
                raise ExecutionConfigurationError(
                    f"execution input was replaced while opening: {candidate}"
                )
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
            opened_after = os.fstat(handle.fileno())
        after = candidate.stat()
    except OSError as exc:
        raise ExecutionConfigurationError(f"cannot hash execution input {candidate}: {exc}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened_before = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
    )
    identity_opened_after = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
    )
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if not (
        identity_before
        == identity_opened_before
        == identity_opened_after
        == identity_after
    ):
        raise ExecutionConfigurationError(f"execution input changed while hashing: {candidate}")
    return bytes(captured) if captured is not None else None, digest.hexdigest()


def stable_file_bytes(path: str | Path) -> tuple[bytes, str]:
    """Read one immutable input once and return the exact bytes plus their digest."""

    content, digest = _stable_file_read(path, capture=True)
    assert content is not None
    return content, digest


def stable_file_sha256(path: str | Path) -> str:
    _, digest = _stable_file_read(path, capture=False)
    return digest


def stable_directory_sha256(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_dir():
        raise ExecutionConfigurationError(f"execution input directory does not exist: {candidate}")
    if _is_link_like(candidate):
        raise ExecutionConfigurationError(f"execution input root cannot be a link: {candidate}")
    root = candidate.resolve(strict=True)
    records: list[dict[str, object]] = []
    for current_raw, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        kept: list[str] = []
        for name in sorted(dirnames):
            child = current / name
            if name in _IGNORED_DIRECTORIES:
                continue
            if _is_link_like(child):
                raise ExecutionConfigurationError(f"execution input contains a linked directory: {child}")
            resolved = child.resolve(strict=True)
            if not _within(resolved, root):
                raise ExecutionConfigurationError(f"execution input escapes its root: {child}")
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            child = current / name
            if _is_link_like(child):
                raise ExecutionConfigurationError(f"execution input contains a linked file: {child}")
            resolved = child.resolve(strict=True)
            if not _within(resolved, root):
                raise ExecutionConfigurationError(f"execution input escapes its root: {child}")
            stat = child.stat()
            records.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "sha256": stable_file_sha256(child),
                }
            )
    if not records:
        raise ExecutionConfigurationError(f"execution input directory is empty: {candidate}")
    return canonical_sha256(records)


def local_input_binding(
    *,
    kind: str,
    location: str,
    ref_id: str,
    directory: bool,
) -> ExecutionInputBinding:
    digest = stable_directory_sha256(location) if directory else stable_file_sha256(location)
    return ExecutionInputBinding.model_validate(
        {
            "kind": kind,
            "ref": {"id": ref_id, "hash": {"algo": "sha256", "value": digest}},
            "source": "local_directory" if directory else "local_file",
            "location": location,
            "content_sha256": digest,
        }
    )


def verify_execution_inputs(config: ResolvedExecutionConfiguration) -> None:
    for binding in (config.inputs.dataset, config.inputs.model, config.inputs.tokenizer):
        if binding.source == "huggingface":
            if binding.resolved_revision is None:  # contract validation should already prevent this.
                raise ExecutionConfigurationError(
                    f"{binding.kind} repository is not pinned to an immutable revision"
                )
            continue
        observed = (
            stable_file_sha256(binding.location)
            if binding.source == "local_file"
            else stable_directory_sha256(binding.location)
        )
        if observed != binding.content_sha256:
            raise ExecutionConfigurationError(
                f"{binding.kind} input bytes changed after planning: {binding.location}"
            )


def verify_execution_objective(
    config: ResolvedExecutionConfiguration,
    *,
    task_type: str,
) -> None:
    """Bind the sealed objective definition to the semantics this dense worker implements."""

    from corpus_studio.platform.objectives import get_objective  # noqa: PLC0415

    objective = get_objective(config.objective_ref.id)
    if objective is None:
        raise ExecutionConfigurationError(
            f"sealed training objective {config.objective_ref.id!r} is not in the current registry"
        )
    observed_hash = (
        config.objective_ref.hash.value
        if config.objective_ref.hash is not None
        else None
    )
    if observed_hash != objective.objective_hash:
        raise ExecutionConfigurationError("sealed training objective hash is stale or mismatched")
    if objective.coarse_task_type is None or objective.coarse_task_type.value != task_type:
        raise ExecutionConfigurationError("sealed training objective does not match the RunPlan task")
    adapter = config.adapter.method.value
    if adapter not in {item.value for item in objective.adaptation_methods}:
        raise ExecutionConfigurationError("sealed adapter method does not match the objective")
    requirement = objective.backend_requirement
    if requirement.task_type is None or requirement.task_type.value != task_type:
        raise ExecutionConfigurationError("objective backend task requirement does not match the plan")
    if config.loss_impl not in requirement.loss_impls:
        raise ExecutionConfigurationError("sealed loss implementation is outside the objective")
    quantization = config.precision.quantized_storage_format
    allowed_quantization = set(requirement.quantization_modes)
    if allowed_quantization:
        if quantization not in allowed_quantization:
            raise ExecutionConfigurationError("sealed quantization does not match the objective")
    elif quantization.value != "none":
        raise ExecutionConfigurationError("this objective requires an unquantized base model")
    formats = {
        variant.dataset_format
        for input_spec in objective.dataset_inputs
        for variant in input_spec.variants
    }
    if config.data.dataset_format not in formats:
        raise ExecutionConfigurationError("sealed dataset format is outside the objective")
    if "adapter" not in {item.kind.value for item in objective.expected_artifacts if item.required}:
        raise ExecutionConfigurationError("sealed objective does not require the adapter artifact emitted")
