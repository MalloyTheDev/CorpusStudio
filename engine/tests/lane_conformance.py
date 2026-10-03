"""The lane conformance matrix: every sealed execution variant against every execution guarantee.

Four separate P1 defects were one mistake repeated: a new training lane was added by copying an
older one, and the copy silently dropped a guarantee the original enforced. Dataset bytes were not
re-verified at consumption (#862), the pinned model/tokenizer identity and loader policy were not
lowered (#863), the sealed no-truncation policy was ignored (#861), and the subprocess parent
admitted a success it had not re-derived (#860). Each was found only after the lane shipped.

This module makes the lane contract explicit instead of implicit. :data:`MATRIX` declares, for every
variant in ``execution_config.RESOLVED_EXECUTION_FIELDS`` and every guarantee in :data:`GUARANTEES`,
one :class:`Cell`:

* ``ENFORCED``  - the lane applies the guarantee today, and ``proof`` names the test that shows it.
* ``GAP``       - the lane can execute without it; ``issue`` tracks the work. The cell is the record
  that the gap is known and deliberate, not an oversight.
* ``NOT_APPLICABLE`` - the guarantee is meaningless for the lane, and ``note`` says why.

``test_lane_conformance.py`` holds the checks. A new resolved variant cannot merge until it declares
a status for every guarantee, and a lane that loses a guarantee flips a cell instead of failing
silently months later.

An ``ENFORCED`` cell is only worth as much as its proof, so :func:`resolve_proof` does not stop at
"a function by that name exists". It imports the proof module and reads the real
``pytest.mark.parametrize`` marks: when the proof varies a ``lane`` argument, the cell's own lane
label must appear among the values that parametrization actually runs. Dropping a lane from a
parametrize list therefore breaks the cell that depended on it, which is the exact move that let the
four original defects through.

Test-only: nothing here ships in ``corpus_studio``, so it adds no worker-reachable bytes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# The short label the test suite parametrizes with, per RunPlan execution field. Tests read far
# better as "preference" than as "resolved_preference_execution", but the matrix has to key off the
# contract field so a new variant cannot be forgotten; this is the one place the two meet.
LANE_LABELS: dict[str, str] = {
    "resolved_execution": "training",
    "resolved_preference_execution": "preference",
    "resolved_pretraining_execution": "pretraining",
    "resolved_full_finetune_execution": "full_finetune",
    "resolved_reward_execution": "reward",
    "resolved_rollout_execution": "rollout",
}


class Status(Enum):
    """How one lane stands against one guarantee."""

    ENFORCED = "enforced"
    UNPROVEN = "unproven"
    GAP = "gap"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Cell:
    """One lane's declared status for one guarantee.

    ``note`` is mandatory and carries the evidence a reader needs: where the guarantee is applied,
    what is missing, or why it cannot apply. ``proof`` (``ENFORCED`` only) is a ``pytest`` node id
    of the form ``tests/test_module.py::test_function``. ``issue`` (``GAP`` and ``UNPROVEN``) is the
    tracking issue number.

    ``UNPROVEN`` is the honest fourth answer, and the reason this file is worth keeping: the lane
    applies the guarantee today, but no test covers it on this lane, so nothing would notice if it
    stopped. That is the state every one of #860 to #863 was in before it became a defect. Calling
    such a cell ``ENFORCED`` would make the matrix claim evidence it does not have.
    """

    status: Status
    note: str
    proof: str | None = None
    issue: int | None = None


@dataclass(frozen=True)
class Guarantee:
    """One execution guarantee the product makes about a sealed run."""

    id: str
    summary: str


def resolve_proof(node_id: str) -> Callable[..., Any]:
    """Resolve a ``tests/test_module.py::test_function`` node id to the function object.

    Imports the module rather than parsing it, so :func:`proof_lane_labels` can read the real
    parametrize marks instead of guessing at a decorator's source. Every proof module is already
    imported by a full suite run, and none of them import this one, so there is no cycle and no
    collection-order dependency. Raises ``AssertionError`` naming the node id when the module or the
    function is missing, which is exactly the signal a renamed proof should give.
    """

    assert node_id.count("::") == 1, f"proof {node_id!r} must be 'tests/<module>.py::<function>'"
    relative, function_name = node_id.split("::")
    assert relative.startswith("tests/") and relative.endswith(".py"), (
        f"proof {node_id!r} must be rooted at tests/ and name a .py module"
    )
    module_name = relative[len("tests/") : -len(".py")]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - a missing proof module is the failure signal
        raise AssertionError(f"proof {node_id!r} names a module that cannot be imported") from exc
    function = getattr(module, function_name, None)
    assert callable(function), f"proof {node_id!r} names a function that does not exist"
    return function


def _lane_values(mark: Any) -> list[str] | None:
    """The ``lane`` values of one parametrize mark, or ``None`` when it does not vary ``lane``."""

    if mark.name != "parametrize" or len(mark.args) < 2:
        return None
    argnames = mark.args[0]
    names = (
        [name.strip() for name in argnames.split(",")]
        if isinstance(argnames, str)
        else [str(name) for name in argnames]
    )
    if "lane" not in names:
        return None
    index = names.index("lane")
    values: list[str] = []
    for argvalue in mark.args[1]:
        # ``pytest.param(...)`` carries its row on ``.values``; a bare row is the value itself for a
        # single argname, and a sequence otherwise.
        if hasattr(argvalue, "values"):
            row: Any = argvalue.values
        elif len(names) == 1:
            row = (argvalue,)
        else:
            row = tuple(argvalue)
        values.append(str(row[index]))
    return values


def proof_lane_labels(function: Callable[..., Any]) -> frozenset[str] | None:
    """The lane labels a proof actually runs, or ``None`` when it is not lane-parametrized.

    ``None`` is a legitimate answer: a proof written for one lane names that lane in its own body,
    and :data:`MATRIX` may still cite it. The distinction matters because a lane-parametrized proof
    can silently stop covering a lane, and this is what makes that visible.
    """

    labels: set[str] = set()
    found = False
    for mark in getattr(function, "pytestmark", ()):
        values = _lane_values(mark)
        if values is not None:
            found = True
            labels.update(values)
    return frozenset(labels) if found else None


GUARANTEES: tuple[Guarantee, ...] = (
    Guarantee(
        "dataset-bytes",
        "The sealed dataset file is re-hashed and compared with its sealed content_sha256 before "
        "any training row is read.",
    ),
    Guarantee(
        "model-identity",
        "The pinned base model sealed in the execution config is verified against the bytes on "
        "disk before weights are loaded.",
    ),
    Guarantee(
        "tokenizer-identity",
        "The pinned tokenizer sealed in the execution config is verified before anything is "
        "tokenized.",
    ),
    Guarantee(
        "formatter-identity",
        "The sealed formatter identity, which hashes the renderer's own source, is compared with "
        "this worker's implementation before rows are rendered.",
    ),
    Guarantee(
        "loader-policy",
        "The sealed quantization, precision, attention and optimizer policy is lowered into the "
        "load rather than left to framework defaults.",
    ),
    Guarantee(
        "truncation-policy",
        "An over-length sequence is refused unless the sealed DATA policy permits truncation; the "
        "sequence flag alone never permits it.",
    ),
    Guarantee(
        "runner-lane",
        "The runner a plan is dispatched to is derived from the seal, so no other lane can consume "
        "this variant's plan.",
    ),
    Guarantee(
        "max-steps",
        "A --max-steps CLI value cannot override the sealed schedule: it is asserted equal or "
        "refused, never silently discarded.",
    ),
    Guarantee(
        "seal-inproc",
        "The in-process supervisor re-verifies the dispatched variant's execution-configuration "
        "hash before the runner is dispatched.",
    ),
    Guarantee(
        "seal-spawn",
        "The subprocess parent re-verifies the dispatched variant's execution-configuration hash "
        "before it spawns the child.",
    ),
    Guarantee(
        "accept-echo",
        "Protocol 2.0 run_accepted must echo the dispatched variant's configuration hash, and the "
        "parent compares it before accepting any run event.",
    ),
    Guarantee(
        "evidence-family",
        "A succeeded terminal must carry exactly this variant's success-evidence family; a foreign "
        "or missing family is a protocol violation.",
    ),
    Guarantee(
        "artifact-admission",
        "A succeeded terminal carries exactly one run-scoped artifact of this variant's kind whose "
        "integrity hash still matches, and its export tree policy runs before any weight byte is "
        "hashed.",
    ),
    Guarantee(
        "fit-reconstruction",
        "A claimed proven fit is reconstructed from the raw measured peak rather than trusted from "
        "the child.",
    ),
)


_SFT = "resolved_execution"
_DPO = "resolved_preference_execution"
_RM = "resolved_reward_execution"
_RL = "resolved_rollout_execution"
_FFT = "resolved_full_finetune_execution"
_PT = "resolved_pretraining_execution"

_ADAPTER_REHASH = (
    "tests/test_sealed_loader_admission.py::"
    "test_a_local_model_changed_after_sealing_is_refused_before_any_read"
)
_BOTH_REHASH = (
    "tests/test_sealed_loader_admission.py::"
    "test_local_model_and_tokenizer_bytes_are_rehashed_for_the_newer_lanes"
)
_NEWER_DATASET = (
    "tests/test_sealed_dataset_consumption.py::"
    "test_non_sft_lanes_refuse_a_changed_dataset_before_any_loader"
)
_UNLOWERABLE = (
    "tests/test_sealed_loader.py::test_each_worker_refuses_an_unlowerable_seal_before_anything_loads"
)
_SAFE_TRUNCATION = (
    "tests/test_sealed_loader.py::test_the_safe_default_seal_refuses_truncation_on_every_lane"
)
_SEAL_INPROC = (
    "tests/test_subprocess_variant_admission.py::"
    "test_execute_run_refuses_a_broken_variant_seal_in_process"
)
_SEAL_SPAWN = (
    "tests/test_subprocess_variant_admission.py::"
    "test_parent_refuses_a_broken_variant_seal_before_spawn"
)
_ACCEPT_ECHO = (
    "tests/test_subprocess_variant_admission.py::"
    "test_parent_binds_run_accepted_to_the_dispatched_variant"
)
_FORGED_TERMINAL = "tests/test_subprocess_variant_admission.py::test_forged_success_terminal_is_rejected"
_TREE_BEFORE_HASH = (
    "tests/test_subprocess_variant_admission.py::"
    "test_export_tree_policy_runs_before_any_weight_byte_is_hashed"
)
_PROVEN_FIT = (
    "tests/test_subprocess_variant_admission.py::"
    "test_genuine_success_with_a_measured_peak_is_admitted_with_its_proven_fit"
)

MATRIX: dict[str, dict[str, Cell]] = {
    "dataset-bytes": {
        _SFT: Cell(
            Status.ENFORCED,
            "verify_sealed_runtime re-hashes the file before the rows are parsed.",
            proof="tests/test_trainer.py::test_sealed_runtime_refuses_dataset_byte_drift",
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "The runner reads the dataset through _verify_sealed_dataset and the worker refuses "
            "rows not bound to its seal.",
            proof=_NEWER_DATASET,
        ),
        _RM: Cell(Status.ENFORCED, "Same runner gate and worker binding guard.", proof=_NEWER_DATASET),
        _RL: Cell(Status.ENFORCED, "Same runner gate and worker binding guard.", proof=_NEWER_DATASET),
        _FFT: Cell(
            Status.ENFORCED, "Same runner gate and worker binding guard.", proof=_NEWER_DATASET
        ),
        _PT: Cell(
            Status.GAP,
            "PretrainingShard seals a content_sha256 per shard, but the worker projects the shards "
            "to bare locations and load_corpus_documents parses them with no hashing. The runner "
            "runs neither sealed-input gate.",
            issue=921,
        ),
    },
    "model-identity": {
        _SFT: Cell(
            Status.ENFORCED,
            "The runner re-validates the sealed inputs against current local bytes, and the worker "
            "re-hashes them again after the load.",
            proof="tests/test_trainer.py::test_local_model_and_tokenizer_are_rehashed_after_loading",
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "_admit_sealed_loader verifies the non-dataset inputs before the dataset is read and "
            "before the worker module is imported.",
            proof=_ADAPTER_REHASH,
        ),
        _RM: Cell(Status.ENFORCED, "Same admission gate and post-load re-hash.", proof=_ADAPTER_REHASH),
        _RL: Cell(
            Status.UNPROVEN,
            "Wired into _admit_sealed_loader exactly like its three siblings, but every content "
            "re-hash test's lane list excludes rollout, so nothing would notice if it regressed.",
            issue=923,
        ),
        _FFT: Cell(
            Status.ENFORCED, "Same admission gate and post-load re-hash.", proof=_ADAPTER_REHASH
        ),
        _PT: Cell(
            Status.GAP,
            "This lane has no inputs model binding (random init builds from a config), but the "
            "contract requires a hash-pinned init.architecture_ref and the worker reads that ref's "
            "id as a path without ever comparing its hash.",
            issue=921,
        ),
    },
    "tokenizer-identity": {
        _SFT: Cell(
            Status.ENFORCED,
            "Loaded from the tokenizer binding's own location and revision, with the sealed "
            "chat-template digest checked before tokenization.",
            proof="tests/test_trainer.py::test_local_model_and_tokenizer_are_rehashed_after_loading",
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "load_sealed_tokenizer plus verify_sealed_chat_template run before formatting.",
            proof=_BOTH_REHASH,
        ),
        _RM: Cell(
            Status.ENFORCED,
            "Same shared loader; this proof mutates a SEPARATE tokenizer directory, so it isolates "
            "the tokenizer binding rather than sharing the model's.",
            proof=(
                "tests/test_sealed_loader_admission.py::"
                "test_a_separate_local_tokenizer_binding_is_rehashed"
            ),
        ),
        _RL: Cell(
            Status.ENFORCED,
            "Same shared loader, before prompt formatting; the proof builds its view from the "
            "rollout seal.",
            proof=(
                "tests/test_sealed_loader.py::"
                "test_load_sealed_tokenizer_uses_the_tokenizer_binding_and_verifies_the_template"
            ),
        ),
        _FFT: Cell(
            Status.ENFORCED,
            "Same shared loader, before the tokenizing preflight.",
            proof=_BOTH_REHASH,
        ),
        _PT: Cell(
            Status.ENFORCED,
            "_verify_pinned_tokenizer_content fails closed when an imported tokenizer has no "
            "tokenizer.json and compares the sealed digest. Caveat: the helper is proven, but "
            "run_pretraining is pragma: no cover, so the wiring is not.",
            proof=(
                "tests/test_pretraining_trainer.py::test_verify_pinned_tokenizer_content_enforces_the_pin"
            ),
        ),
    },
    "formatter-identity": {
        _SFT: Cell(
            Status.UNPROVEN,
            "The only lane that compares the sealed identity, and the refusal string appears in no "
            "test; run_training is pragma: no cover, so nothing drives execution to it.",
            issue=923,
        ),
        _DPO: Cell(
            Status.GAP,
            "The planner seals formatter_id and formatter_sha256, and the worker formats without "
            "ever comparing them.",
            issue=919,
        ),
        _RM: Cell(Status.GAP, "Same sealed policy, same missing comparison.", issue=919),
        _RL: Cell(
            Status.GAP,
            "The worker's only reference to the sealed identity is a comment asserting the binding; "
            "nothing checks it.",
            issue=919,
        ),
        _FFT: Cell(
            Status.GAP,
            "The loader view carries both fields to the worker, which never reads them; the one "
            "comparison lives in run_training, which this lane never calls.",
            issue=919,
        ),
        _PT: Cell(
            Status.NOT_APPLICABLE,
            "No formatter field on this lane's contract: documents are packed, never rendered.",
        ),
    },
    "loader-policy": {
        _SFT: Cell(
            Status.ENFORCED,
            "build_model_load_kwargs pins quantization, dtype, revision and placement fail-closed, "
            "and the attention policy is applied and then observed.",
            proof=(
                "tests/test_trainer.py::"
                "test_model_load_kwargs_pin_quantization_dtype_revision_and_device_map"
            ),
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "verify_loader_policy_supported refuses an unlowerable seal at the runner and again in "
            "the worker, before anything loads.",
            proof=_UNLOWERABLE,
        ),
        _RM: Cell(Status.ENFORCED, "Same shared rule.", proof=_UNLOWERABLE),
        _RL: Cell(
            Status.ENFORCED,
            "Same shared rule, plus the rollout-only requirement that the served reward base is the "
            "policy's pinned base.",
            proof=_UNLOWERABLE,
        ),
        _FFT: Cell(
            Status.ENFORCED,
            "Same shared rule, plus post-load storage and placement verification.",
            proof=_UNLOWERABLE,
        ),
        _PT: Cell(
            Status.GAP,
            "The contract seals and validates precision, attention and device_map, and the worker "
            "reads none of them: the model is built with no dtype, attention implementation or "
            "placement, so the run takes framework defaults. Only the optimizer is lowered.",
            issue=922,
        ),
    },
    "truncation-policy": {
        _SFT: Cell(
            Status.ENFORCED,
            "The full-dataset preflight refuses supervised truncation unless the sealed data policy "
            "allows it.",
            proof=(
                "tests/test_trainer.py::"
                "test_full_dataset_preflight_refuses_supervised_truncation_fail_closed"
            ),
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "sealed_truncation_permitted is the one rule: the data policy is the key and the "
            "sequence flag must agree, so it fails closed from either direction.",
            proof=_SAFE_TRUNCATION,
        ),
        _RM: Cell(Status.ENFORCED, "Same shared rule, for training and held-out eval.", proof=_SAFE_TRUNCATION),
        _RL: Cell(
            Status.ENFORCED,
            "Same shared rule, keyed off experience.truncation_policy on this lane.",
            proof=_SAFE_TRUNCATION,
        ),
        _FFT: Cell(
            Status.ENFORCED,
            "Same shared rule, through the named full-parameter wrapper.",
            proof=_SAFE_TRUNCATION,
        ),
        _PT: Cell(
            Status.NOT_APPLICABLE,
            "PretrainingDataPolicy has no truncation_policy and the worker packs rather than cuts: "
            "documents are concatenated and split with accounted coverage.",
        ),
    },
    "runner-lane": {
        _SFT: Cell(
            Status.ENFORCED,
            "required_runner_lane derives the lane from the sealed plan alone; verify_runner_lane "
            "refuses any other, on both the in-process and subprocess paths.",
            proof=(
                "tests/test_execution_config.py::test_runner_lane_is_derived_from_the_sealed_plan_only"
            ),
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "Variant-gated to the preference lane, with a typed refusal if handed to the SFT runner.",
            proof=(
                "tests/test_platform_planner.py::"
                "test_preference_dpo_resolves_to_a_sealed_config_and_routes_to_the_preference_lane"
            ),
        ),
        _RM: Cell(
            Status.ENFORCED,
            "Variant-gated to the reward lane, with a typed refusal on the SFT runner.",
            proof=(
                "tests/test_platform_planner.py::"
                "test_reward_model_resolves_to_a_sealed_config_and_routes_to_the_reward_lane"
            ),
        ),
        _RL: Cell(
            Status.ENFORCED,
            "Strictly stronger today: required_runner_lane RAISES while on_policy_rl is not "
            "workload_verified, so no runner name can pass.",
            proof=(
                "tests/test_platform_planner.py::"
                "test_on_policy_rl_resolves_to_a_sealed_config_and_is_refused_at_execution"
            ),
        ),
        _FFT: Cell(
            Status.ENFORCED,
            "Variant-gated to the full_finetune lane, with a runner-type gate behind it.",
            proof=(
                "tests/test_platform_planner.py::"
                "test_full_parameter_sft_seals_a_full_model_config_and_routes_to_the_full_finetune_lane"
            ),
        ),
        _PT: Cell(
            Status.ENFORCED,
            "Variant-gated to the pretraining lanes, with a typed refusal on the SFT runner.",
            proof=(
                "tests/test_platform_planner.py::"
                "test_pretraining_plan_is_admitted_at_execution_and_routes_to_the_pretraining_lane"
            ),
        ),
    },
    "max-steps": {
        _SFT: Cell(
            Status.ENFORCED,
            "Asserted equal to the sealed schedule on both paths, before the trainer is called.",
            proof=(
                "tests/test_platform_runners.py::"
                "test_max_steps_override_is_refused_without_calling_the_trainer"
            ),
        ),
        _DPO: Cell(
            Status.GAP,
            "The subprocess parent refuses an unequal override, but build_lane_runner forwards "
            "max_steps only to the SFT runner and execute_run has no such parameter, so the "
            "default in-process path discards it silently.",
            issue=920,
        ),
        _RM: Cell(Status.GAP, "Same silent in-process drop.", issue=920),
        _RL: Cell(Status.GAP, "Same silent in-process drop.", issue=920),
        _FFT: Cell(Status.GAP, "Same silent in-process drop.", issue=920),
        _PT: Cell(Status.GAP, "Same silent in-process drop.", issue=920),
    },
    "seal-inproc": {
        _SFT: Cell(Status.ENFORCED, "One binding-generic check covers every variant.", proof=_SEAL_INPROC),
        _DPO: Cell(Status.ENFORCED, "Same binding-generic check.", proof=_SEAL_INPROC),
        _RM: Cell(Status.ENFORCED, "Same binding-generic check.", proof=_SEAL_INPROC),
        _RL: Cell(
            Status.ENFORCED,
            "Reachable despite the lane being non-executable: the seal check precedes the lane gate.",
            proof=_SEAL_INPROC,
        ),
        _FFT: Cell(Status.ENFORCED, "Same binding-generic check.", proof=_SEAL_INPROC),
        _PT: Cell(Status.ENFORCED, "Same binding-generic check.", proof=_SEAL_INPROC),
    },
    "seal-spawn": {
        _SFT: Cell(
            Status.UNPROVEN,
            "The parent's check is binding-generic and covers this lane, but the proof's lane list "
            "omits it; the nearest candidate tampers an echo plan, so it proves the plan_hash "
            "refusal rather than the execution-configuration seal.",
            issue=923,
        ),
        _DPO: Cell(Status.ENFORCED, "Re-verified before the child is spawned.", proof=_SEAL_SPAWN),
        _RM: Cell(Status.ENFORCED, "Re-verified before the child is spawned.", proof=_SEAL_SPAWN),
        _RL: Cell(
            Status.ENFORCED,
            "Reachable for the same reason as in-process: the seal check precedes the lane gate.",
            proof=_SEAL_SPAWN,
        ),
        _FFT: Cell(Status.ENFORCED, "Re-verified before the child is spawned.", proof=_SEAL_SPAWN),
        _PT: Cell(Status.ENFORCED, "Re-verified before the child is spawned.", proof=_SEAL_SPAWN),
    },
    "accept-echo": {
        _SFT: Cell(Status.ENFORCED, "The parent compares the echoed hash before any run event.", proof=_ACCEPT_ECHO),
        _DPO: Cell(Status.ENFORCED, "Same comparison, bound to this variant.", proof=_ACCEPT_ECHO),
        _RM: Cell(Status.ENFORCED, "Same comparison, bound to this variant.", proof=_ACCEPT_ECHO),
        _RL: Cell(
            Status.NOT_APPLICABLE,
            "The comparison is binding-generic, but a rollout plan is refused before spawn and the "
            "worker rejects it too, so no run_accepted for this lane can exist. Revisit when "
            "on_policy_rl is promoted to workload_verified.",
        ),
        _FFT: Cell(Status.ENFORCED, "Same comparison, bound to this variant.", proof=_ACCEPT_ECHO),
        _PT: Cell(Status.ENFORCED, "Same comparison, bound to this variant.", proof=_ACCEPT_ECHO),
    },
    "evidence-family": {
        _SFT: Cell(
            Status.ENFORCED,
            "A foreign family on an SFT terminal is a protocol violation.",
            proof=(
                "tests/test_subprocess_variant_admission.py::"
                "test_sft_terminal_with_a_foreign_family_is_a_protocol_violation"
            ),
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "The forgery table drives foreign, extra and missing families per lane.",
            proof=_FORGED_TERMINAL,
        ),
        _RM: Cell(Status.ENFORCED, "Same forgery table.", proof=_FORGED_TERMINAL),
        _RL: Cell(Status.ENFORCED, "Same forgery table.", proof=_FORGED_TERMINAL),
        _FFT: Cell(Status.ENFORCED, "Same forgery table.", proof=_FORGED_TERMINAL),
        _PT: Cell(Status.ENFORCED, "Same forgery table.", proof=_FORGED_TERMINAL),
    },
    "artifact-admission": {
        _SFT: Cell(
            Status.GAP,
            "Count, kind, run scope and integrity all hold, but the tree policy is explicitly "
            "skipped for this lane, so the weight hash (which follows links) runs before "
            "_validate_adapter_tree can refuse a linked payload.",
            issue=918,
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "The kind's tree policy runs before any byte is hashed.",
            proof=_TREE_BEFORE_HASH,
        ),
        _RM: Cell(Status.ENFORCED, "Same ordering.", proof=_TREE_BEFORE_HASH),
        _RL: Cell(Status.ENFORCED, "Same ordering.", proof=_TREE_BEFORE_HASH),
        _FFT: Cell(Status.ENFORCED, "Same ordering, against the model-tree policy.", proof=_TREE_BEFORE_HASH),
        _PT: Cell(Status.ENFORCED, "Same ordering, against the model-tree policy.", proof=_TREE_BEFORE_HASH),
    },
    "fit-reconstruction": {
        _SFT: Cell(
            Status.UNPROVEN,
            "The parent's reconstruction is variant-generic and covers this lane, but the proofs' "
            "lane lists omit it; the nearest SFT test exercises the evidence comparison, not the "
            "final_fit reconstruction.",
            issue=923,
        ),
        _DPO: Cell(
            Status.ENFORCED,
            "A claimed fit must equal the reconstruction from the raw peak.",
            proof=_PROVEN_FIT,
        ),
        _RM: Cell(Status.ENFORCED, "Same reconstruction.", proof=_PROVEN_FIT),
        _RL: Cell(Status.ENFORCED, "Same reconstruction.", proof=_PROVEN_FIT),
        _FFT: Cell(Status.ENFORCED, "Same reconstruction.", proof=_PROVEN_FIT),
        _PT: Cell(Status.ENFORCED, "Same reconstruction.", proof=_PROVEN_FIT),
    },
}
