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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import HashRef, Ref
from .contracts import (
    CapabilityReport,
    ExecutionInputBinding,
    ResolvedExecutionConfiguration,
    ResolvedFullFinetuneExecutionConfiguration,
    ResolvedPreferenceExecutionConfiguration,
    ResolvedRewardExecutionConfiguration,
    ResolvedRolloutExecutionConfiguration,
    ResolvedPretrainingExecutionConfiguration,
    RunPlan,
)

_IGNORED_DIRECTORIES = {".git", "__pycache__"}
_FORMATTER_IDENTITIES = {
    "instruction": "corpus-studio:instruction-alpaca-v1",
    "chat": "corpus-studio:tokenizer-chat-template-v1",
    "trace": "corpus-studio:structured-trace-renderer-v1",
    "preference": "corpus-studio:preference-pair-v1",
}
_RUNTIME_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class ExecutionConfigurationError(ValueError):
    """A seal or immutable-input check failed before model loading."""


def run_scoped_training_output(
    config: (
        ResolvedExecutionConfiguration
        | ResolvedPretrainingExecutionConfiguration
        | ResolvedPreferenceExecutionConfiguration
        | ResolvedFullFinetuneExecutionConfiguration
        | ResolvedRewardExecutionConfiguration
        | ResolvedRolloutExecutionConfiguration
    ),
    run_id: str,
    *,
    leaf: str = "adapter",
) -> Path:
    """Resolve the final trainer directory from the sealed root/layout and fresh run identity. ``leaf``
    is the artifact kind under ``artifacts/``: "adapter" for the SFT/DPO PEFT export, "model" for a
    full-parameter pretraining export."""

    if config.output_layout != "run_scoped_v1":  # pragma: no cover - literal contract defense
        raise ExecutionConfigurationError(
            f"unsupported resolved output layout {config.output_layout!r}"
        )
    if not _RUNTIME_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ExecutionConfigurationError("run_id is unsafe for run-scoped output resolution")
    return Path(config.output_dir) / "runs" / run_id / "artifacts" / leaf


PREFERENCE_NOT_EXECUTABLE_REASON = (
    "a sealed preference (DPO) plan runs on the first-party PreferenceRunner lane, not the SFT/pretraining "
    "runner: dispatch it through platform-run so required_runner_lane selects the 'preference' lane. (The "
    "same refusal fires if the 'preference_dpo' execution variant is not workload_verified for this "
    "backend.)"
)

PRETRAINING_NOT_EXECUTABLE_REASON = (
    "a from-scratch / continued pretraining plan runs on the first-party PretrainingRunner lane, not the "
    "SFT/DPO runner: dispatch it through platform-run so required_runner_lane selects the 'pretraining' "
    "lane. (The same refusal fires if the 'pretraining' execution variant is not workload_verified for "
    "this backend.)"
)

FULL_FINETUNE_NOT_EXECUTABLE_REASON = (
    "a full-parameter fine-tune runs on the first-party full-parameter lane, not the SFT/DPO adapter "
    "runner: dispatch it through platform-run so required_runner_lane selects the 'full_finetune' lane. "
    "(The same refusal fires if the 'dense_full_finetune' execution variant is not workload_verified for "
    "this backend.)"
)

REWARD_NOT_EXECUTABLE_REASON = (
    "a sealed reward-model plan runs on the first-party reward lane, not the SFT/DPO adapter runner: "
    "dispatch it through platform-run so required_runner_lane selects the 'reward' lane. (The same refusal "
    "fires - as it does now - while the 'reward_model' execution variant is not workload_verified for this "
    "backend: the reward-head trainer branch + a measured run + the promoting wheel are the gated "
    "milestone.)"
)

ROLLOUT_NOT_EXECUTABLE_REASON = (
    "a sealed on-policy RL (rollout) plan runs on the first-party rollout lane, not the SFT/DPO adapter "
    "runner: dispatch it through platform-run so required_runner_lane selects the 'rollout' lane. (The same "
    "refusal fires - as it does now - while the 'on_policy_rl' execution variant is not workload_verified "
    "for this backend: the rollout+reward+GRPO worker + a measured run + the promoting wheel are the gated "
    "milestone.)"
)


def required_runner_lane(plan: RunPlan) -> str:
    """Return the only runner lane allowed to consume ``plan``."""

    execution = plan.resolved_execution
    if execution is not None:
        if plan.backend_ref.id != "corpus_studio":
            raise ExecutionConfigurationError(
                "resolved training plans require the first-party corpus_studio worker"
            )
        return "cpu_toy" if execution.runtime_mode == "cpu_toy" else "training"
    if plan.resolved_preference_execution is not None:
        # A sealed preference (DPO) plan is admitted at planning; at EXECUTION it is admitted only once the
        # preference_dpo variant reaches workload_verified (a measured GPU run through the first-party
        # PreferenceRunner + the supervisor adapter reload-verify). Gate on the ladder - keyed by the
        # SPECIFIC objective (only dpo_qlora has a built shape) - then route to the PreferenceRunner lane;
        # the SFT/pretraining lanes never run the DPO reference / log-prob path.
        from corpus_studio.platform.enums import TaskType  # noqa: PLC0415
        from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
            ExecutionVariantRefused,
            admit_task_execution_variant,
            reference_execution_variants,
        )

        try:
            admit_task_execution_variant(
                TaskType.preference,
                declared_variants=reference_execution_variants(),
                objective_id=plan.resolved_preference_execution.objective_ref.id,
            )
        except ExecutionVariantRefused as exc:
            raise ExecutionConfigurationError(PREFERENCE_NOT_EXECUTABLE_REASON) from exc
        return "preference"
    if plan.resolved_pretraining_execution is not None:
        # Pretraining is admitted at planning; at EXECUTION it is admitted only once the pretraining
        # variant reaches workload_verified (a measured GPU run through the first-party PretrainingRunner
        # + the supervisor reload-verify). Gate on the ladder, then route to the PretrainingRunner lane -
        # the SFT/DPO lane never runs a full-parameter model.
        from corpus_studio.platform.enums import TaskType  # noqa: PLC0415
        from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
            ExecutionVariantRefused,
            admit_task_execution_variant,
            reference_execution_variants,
        )

        try:
            admit_task_execution_variant(
                TaskType.pretraining, declared_variants=reference_execution_variants()
            )
        except ExecutionVariantRefused as exc:
            raise ExecutionConfigurationError(PRETRAINING_NOT_EXECUTABLE_REASON) from exc
        return (
            "pretraining_cpu_toy"
            if plan.resolved_pretraining_execution.runtime_mode == "cpu_toy"
            else "pretraining"
        )
    if plan.resolved_full_finetune_execution is not None:
        # Full-parameter SFT is admitted at planning; at EXECUTION it is admitted only once the
        # dense_full_finetune variant reaches workload_verified (the full-parameter worker + full-model
        # reload-verify + a measured run). Gate on the ladder, keyed by the full-parameter SFT shape, then
        # route to the full-finetune lane - the adapter SFT lane never trains a full model.
        from corpus_studio.platform.enums import TaskType  # noqa: PLC0415
        from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
            ExecutionVariantRefused,
            admit_task_execution_variant,
            reference_execution_variants,
        )

        try:
            admit_task_execution_variant(
                TaskType.sft,
                is_full_parameter=True,
                declared_variants=reference_execution_variants(),
            )
        except ExecutionVariantRefused as exc:
            raise ExecutionConfigurationError(FULL_FINETUNE_NOT_EXECUTABLE_REASON) from exc
        return "full_finetune"
    if plan.resolved_reward_execution is not None:
        # A sealed reward-model plan is admitted at planning; at EXECUTION it is admitted only once the
        # reward_model variant reaches workload_verified (the reward-head worker + reload-verify + a
        # measured run). Gate on the ladder, keyed by the specific reward objective (only the pairwise
        # reward_model has a built shape), then route to the reward lane - the adapter/DPO lanes never
        # train a scalar score head.
        from corpus_studio.platform.enums import TaskType  # noqa: PLC0415
        from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
            ExecutionVariantRefused,
            admit_task_execution_variant,
            reference_execution_variants,
        )

        try:
            admit_task_execution_variant(
                TaskType.reward,
                declared_variants=reference_execution_variants(),
                objective_id=plan.resolved_reward_execution.objective_ref.id,
            )
        except ExecutionVariantRefused as exc:
            raise ExecutionConfigurationError(REWARD_NOT_EXECUTABLE_REASON) from exc
        return "reward"
    if plan.resolved_rollout_execution is not None:
        # A sealed on-policy RL plan is admitted at planning; at EXECUTION it is admitted only once the
        # on_policy_rl variant reaches workload_verified (the rollout+reward+GRPO worker + a measured run).
        # Gate on the ladder, keyed by the specific on-policy objective (only grpo has a built shape), then
        # route to the rollout lane - the adapter/DPO/reward lanes never sample + optimize on-policy rollouts.
        from corpus_studio.platform.enums import TaskType  # noqa: PLC0415
        from corpus_studio.platform.execution_variants import (  # noqa: PLC0415
            ExecutionVariantRefused,
            admit_task_execution_variant,
            reference_execution_variants,
        )

        try:
            admit_task_execution_variant(
                TaskType.grpo,
                declared_variants=reference_execution_variants(),
                objective_id=plan.resolved_rollout_execution.objective_ref.id,
            )
        except ExecutionVariantRefused as exc:
            raise ExecutionConfigurationError(ROLLOUT_NOT_EXECUTABLE_REASON) from exc
        return "rollout"
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


def preference_execution_configuration_hash_for(
    config: ResolvedPreferenceExecutionConfiguration,
) -> str:
    """Seal a DPO execution configuration exactly as the SFT sibling is sealed (canonical JSON over
    every field but ``configuration_hash``). A separate function keeps the two seals decoupled: the
    byte-locked SFT seal can never be perturbed by a DPO change."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_preference_execution_configuration_hash(
    config: ResolvedPreferenceExecutionConfiguration,
) -> bool:
    return config.configuration_hash == preference_execution_configuration_hash_for(config)


def reward_execution_configuration_hash_for(
    config: ResolvedRewardExecutionConfiguration,
) -> str:
    """Seal a reward-model execution configuration exactly as the SFT/DPO siblings are sealed (canonical
    JSON over every field but ``configuration_hash``). A separate function keeps the seals decoupled: the
    byte-locked SFT seal can never be perturbed by a reward change."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_reward_execution_configuration_hash(
    config: ResolvedRewardExecutionConfiguration,
) -> bool:
    return config.configuration_hash == reward_execution_configuration_hash_for(config)


def rollout_execution_configuration_hash_for(
    config: ResolvedRolloutExecutionConfiguration,
) -> str:
    """Seal an on-policy RL execution configuration exactly as the SFT/DPO/reward siblings are sealed
    (canonical JSON over every field but ``configuration_hash``). A separate function keeps the seals
    decoupled: the byte-locked SFT seal can never be perturbed by a rollout change."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_rollout_execution_configuration_hash(
    config: ResolvedRolloutExecutionConfiguration,
) -> bool:
    return config.configuration_hash == rollout_execution_configuration_hash_for(config)


def pretraining_execution_configuration_hash_for(
    config: ResolvedPretrainingExecutionConfiguration,
) -> str:
    """Seal a pretraining execution configuration exactly as the SFT/DPO siblings are sealed (canonical
    JSON over every field but ``configuration_hash``). A separate function keeps the seals decoupled: the
    byte-locked SFT seal can never be perturbed by a pretraining change."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_pretraining_execution_configuration_hash(
    config: ResolvedPretrainingExecutionConfiguration,
) -> bool:
    return config.configuration_hash == pretraining_execution_configuration_hash_for(config)


def full_finetune_execution_configuration_hash_for(
    config: ResolvedFullFinetuneExecutionConfiguration,
) -> str:
    """Seal a full-parameter fine-tune execution configuration exactly as the SFT/DPO/pretraining siblings
    are sealed (canonical JSON over every field but ``configuration_hash``). A separate function keeps the
    seals decoupled: the byte-locked SFT seal can never be perturbed by a full-finetune change."""
    return canonical_sha256(config.model_dump(mode="json", exclude={"configuration_hash"}))


def verify_full_finetune_execution_configuration_hash(
    config: ResolvedFullFinetuneExecutionConfiguration,
) -> bool:
    return config.configuration_hash == full_finetune_execution_configuration_hash_for(config)


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


def preference_formatter_identity() -> tuple[str, str]:
    """The sealed identity of the preference-pair formatter, DISTINCT from :func:`formatter_identity`'s
    SFT ``format_example_text`` (which reads instruction/messages/trace fields, not a preference pair's
    ``prompt``/``chosen``/``rejected``). Returns the id + a content digest of ``format_preference_pair``'s
    source, so a DPO run formats every pair identically and a formatter change fails closed."""
    formatter_id = _FORMATTER_IDENTITIES["preference"]
    try:
        from corpus_studio.training.trainer import format_preference_pair  # noqa: PLC0415

        sources = [inspect.getsource(format_preference_pair)]
    except (ImportError, OSError, TypeError) as exc:
        raise ExecutionConfigurationError(
            f"cannot inspect the sealed preference formatter implementation: {exc}"
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


def verify_run_scoped_output_path(
    config: (
        ResolvedExecutionConfiguration
        | ResolvedPretrainingExecutionConfiguration
        | ResolvedPreferenceExecutionConfiguration
        | ResolvedFullFinetuneExecutionConfiguration
        | ResolvedRewardExecutionConfiguration
        | ResolvedRolloutExecutionConfiguration
    ),
    run_id: str,
    *,
    observed_path: str | Path | None = None,
    require_exists: bool = False,
    leaf: str = "adapter",
) -> Path:
    """Require the exact lexical run output and reject link-like descendants before/after training."""

    sealed_root = Path(config.output_dir).absolute()
    expected = run_scoped_training_output(config, run_id, leaf=leaf).absolute()
    candidate = Path(observed_path).absolute() if observed_path is not None else expected
    if candidate != expected:
        raise ExecutionConfigurationError(
            "trainer output differs from the exact sealed run-scoped output path"
        )
    try:
        expected.relative_to(sealed_root)
    except ValueError as exc:  # pragma: no cover - construction above is defensive-by-shape.
        raise ExecutionConfigurationError("run-scoped output escapes its sealed root") from exc

    current = sealed_root
    relative_parts = expected.relative_to(sealed_root).parts
    for part in (None, *relative_parts):
        if part is not None:
            current = current / part
        try:
            exists = current.exists() or current.is_symlink()
        except OSError as exc:
            raise ExecutionConfigurationError(
                f"cannot inspect run-scoped output component: {current}"
            ) from exc
        if exists and _is_link_like(current):
            raise ExecutionConfigurationError(
                f"run-scoped output contains a link-like component: {current}"
            )
        if exists:
            try:
                resolved = current.resolve(strict=True)
                root_resolved = sealed_root.resolve(strict=sealed_root.exists())
            except (OSError, RuntimeError) as exc:
                raise ExecutionConfigurationError(
                    f"cannot resolve run-scoped output component: {current}"
                ) from exc
            if not _within(resolved, root_resolved):
                raise ExecutionConfigurationError(
                    f"run-scoped output component escapes the sealed root: {current}"
                )
    if require_exists and (not expected.is_dir() or _is_link_like(expected)):
        raise ExecutionConfigurationError(
            "trainer did not produce a regular run-scoped output directory"
        )
    return expected


def run_scoped_pretraining_output(
    config: ResolvedPretrainingExecutionConfiguration,
    run_id: str,
) -> Path:
    """The full-parameter pretraining sibling of :func:`run_scoped_training_output` - the run-scoped
    ``model`` export directory (never the SFT ``adapter`` leaf)."""

    return run_scoped_training_output(config, run_id, leaf="model")


def verify_run_scoped_pretraining_output_path(
    config: ResolvedPretrainingExecutionConfiguration,
    run_id: str,
    *,
    observed_path: str | Path | None = None,
    require_exists: bool = False,
) -> Path:
    """Require the exact lexical run-scoped ``model`` output for a pretraining run and reject link-like
    descendants (the pretraining analog of :func:`verify_run_scoped_output_path`)."""

    return verify_run_scoped_output_path(
        config, run_id, observed_path=observed_path, require_exists=require_exists, leaf="model"
    )


def _stable_file_read(
    path: str | Path,
    *,
    capture: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[bytes | None, str]:
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
            bytes_read = 0
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                if captured is not None:
                    captured.extend(chunk)
                bytes_read += len(chunk)
                if progress_callback is not None:
                    progress_callback(bytes_read, opened_before.st_size)
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


def stable_file_bytes(
    path: str | Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[bytes, str]:
    """Read one immutable input once and return the exact bytes plus their digest."""

    content, digest = _stable_file_read(
        path,
        capture=True,
        progress_callback=progress_callback,
    )
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


def _verify_execution_input_bindings(bindings: tuple[ExecutionInputBinding, ...]) -> None:
    for binding in bindings:
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


def verify_execution_inputs(config: ResolvedExecutionConfiguration) -> None:
    """Revalidate every sealed input against current local bytes."""

    _verify_execution_input_bindings(
        (config.inputs.dataset, config.inputs.model, config.inputs.tokenizer)
    )


def verify_execution_non_dataset_inputs(config: ResolvedExecutionConfiguration) -> None:
    """Revalidate model/tokenizer inputs when the consumer owns the stable dataset read.

    The training worker uses :func:`stable_file_bytes` to hash and capture the dataset exactly once,
    then parses those captured bytes. Rehashing that binding here would add a redundant full pass.
    """

    _verify_execution_input_bindings((config.inputs.model, config.inputs.tokenizer))


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
