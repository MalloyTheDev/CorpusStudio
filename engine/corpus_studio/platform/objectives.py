"""Versioned training-objective registry and conservative compatibility checks.

An objective describes *what* is optimized. A backend describes *how* it may be executed. Keeping
those registries separate prevents a rendered config, installed package, or broad ``TaskType`` from
becoming a false support claim. Pure control-plane code: no torch, transformers, or network access.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

from .common import HashRef, Ref
from .contracts import (
    BackendManifest,
    CapabilityReport,
    ModelDescriptor,
    ObjectiveCompatibilityAxis,
    ObjectiveCompatibilityReport,
    TrainingObjective,
)
from .enums import (
    ModelExecutionKind,
    ModelTaskClass,
    ObjectiveCompatibilityStatus,
    ObjectiveSelectionMode,
    ObjectiveUpdateScope,
)

_ZERO_HASH = "0" * 64
_ALL_EXECUTION_KINDS = sorted(
    [
        "dense",
        "sparse",
        "mixture_of_experts",
        "conditional",
        "hybrid",
    ]
)
_CAUSAL_MODEL = ["causal_lm"]
_GENERATIVE_MODELS = sorted(["causal_lm", "seq2seq_lm"])
_ALL_KNOWN_MODEL_TASKS = sorted(item.value for item in ModelTaskClass if item.value != "unknown")


def _fields(**items: str) -> list[dict[str, str]]:
    return [
        {"name": name, "field_type": field_type, "semantic_role": name}
        for name, field_type in sorted(items.items())
    ]


def _variant(
    schema_id: str,
    *,
    availability: str = "builtin",
    schema_version: str | None = None,
    dataset_format: str | None = None,
    **fields: str,
) -> dict[str, object]:
    return {
        "schema_id": schema_id,
        "availability": availability,
        "schema_version": schema_version or ("0.1.0" if availability == "builtin" else "1.0.0"),
        "dataset_format": dataset_format,
        "required_fields": _fields(**fields),
    }


_RAW_TEXT = [_variant("raw_text", dataset_format="text", text="text")]
_SFT = sorted(
    [
        _variant("instruction", dataset_format="instruction", instruction="text", output="markdown"),
        _variant("chat", dataset_format="chat", messages="messages"),
        _variant("code", dataset_format="instruction", instruction="text", output="code"),
    ],
    key=lambda item: str(item["schema_id"]),
)
_COMPLETION = sorted(
    [
        _variant("instruction", dataset_format="instruction", instruction="text", output="markdown"),
        _variant("code", dataset_format="instruction", instruction="text", output="code"),
    ],
    key=lambda item: str(item["schema_id"]),
)
_PREFERENCE = [
    _variant(
        "preference",
        dataset_format="preference",
        chosen="markdown",
        prompt="text",
        rejected="markdown",
    )
]


def _dataset(variants: list[dict[str, object]], *, role: str = "train") -> list[dict[str, object]]:
    return [
        {
            "role": role,
            "variants": variants,
            "row_validation_required": True,
            "split_isolation_required": True,
        }
    ]


def _label(kind: str, source_fields: list[str], construction: str) -> list[dict[str, object]]:
    return [
        {
            "label_id": "primary_labels",
            "kind": kind,
            "source_fields": sorted(source_fields),
            "construction": construction,
            "ignore_index": -100,
        }
    ]


def _mask(kind: str, source_fields: list[str], construction: str) -> list[dict[str, object]]:
    return [
        {
            "mask_id": "primary_mask",
            "kind": kind,
            "source_fields": sorted(source_fields),
            "include_padding": False,
            "include_special_tokens": False,
            "empty_mask_action": "reject",
            "construction": construction,
        }
    ]


def _loss(
    kind: str,
    construction: str,
    *,
    component_id: str = "primary_loss",
    label_ref: str | None = "primary_labels",
    mask_ref: str | None = "primary_mask",
) -> dict[str, object]:
    return {
        "component_id": component_id,
        "kind": kind,
        "construction": construction,
        "label_ref": label_ref,
        "mask_ref": mask_ref,
        "default_weight": 1.0,
        "reduction": "token_mean" if kind == "cross_entropy" else "mean",
    }


def _model(
    task_classes: list[str],
    *,
    reference: bool = False,
    reward_head: bool = False,
    multimodal_projector: bool = False,
    tokenizer: bool = True,
    output_head: bool = True,
) -> dict[str, object]:
    return {
        "task_classes": sorted(task_classes),
        "execution_kinds": _ALL_EXECUTION_KINDS,
        "requires_tokenizer": tokenizer,
        "requires_output_head": output_head,
        "requires_reference_model": reference,
        "requires_reward_head": reward_head,
        "requires_multimodal_projector": multimodal_projector,
        "custom_code_policy": "isolated_approval",
    }


def _full_update() -> dict[str, object]:
    return {
        "scopes": ["all_parameters"],
        "selection_mode": "all",
        "stable_expert_identity": "when_expert_scoped",
        "exposure_tracking": "router_and_expert",
        "optimizer_clock": "per_component",
        "update_window_definition": "One optimizer window with component and expert exposure evidence.",
        "starvation_gate_required_when_expert_scoped": True,
        "routing_collapse_gate_required_when_routed": True,
        "notes": [
            "Dense execution has no expert exposure axis; sparse execution must retain stable expert identity."
        ],
    }


def _adapter_update() -> dict[str, object]:
    return {
        "scopes": ["adapters"],
        "selection_mode": "adapter_only",
        "stable_expert_identity": "when_expert_scoped",
        "exposure_tracking": "per_component",
        "optimizer_clock": "per_component",
        "update_window_definition": "One optimizer window per stable adapter component.",
        "notes": [
            "Expert-scoped adapters must preserve the owning expert identity in checkpoints and lineage."
        ],
    }


def _task_head_update(*, projector: bool = False) -> dict[str, object]:
    scopes = ["projector", "task_head"] if projector else ["task_head"]
    return {
        "scopes": sorted(scopes),
        "selection_mode": "task_head_only",
        "stable_expert_identity": "not_required",
        "exposure_tracking": "per_component",
        "optimizer_clock": "per_component",
        "update_window_definition": "One optimizer window per trainable task-head component.",
    }


def _no_update() -> dict[str, object]:
    return {
        "scopes": ["none"],
        "selection_mode": "none",
        "stable_expert_identity": "not_required",
        "exposure_tracking": "none",
        "optimizer_clock": "none",
        "update_window_definition": "No optimizer update is performed.",
    }


def _backend(
    task_type: str | None,
    *,
    losses: list[str] | None = None,
    adapters: list[str] | None = None,
    quantizations: list[str] | None = None,
    capabilities: list[str],
    hardware_required: bool = True,
) -> dict[str, object]:
    return {
        "task_type": task_type,
        "loss_impls": sorted(losses or []),
        "adaptation_methods": sorted(adapters or []),
        "quantization_modes": sorted(quantizations or []),
        "objective_capabilities": sorted(capabilities),
        "functional_probe_required": True,
        "hardware_verification_required": hardware_required,
    }


def _artifacts(*kinds: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = [
        {
            "kind": kind,
            "required": True,
            "component_scoped": kind in {"adapter", "checkpoint", "expert_shards", "routing_state"},
            "condition": None,
            "description": f"Required {kind.replace('_', ' ')} output with provenance.",
        }
        for kind in sorted(kinds)
    ]
    if "checkpoint" in kinds:
        conditional = {
            "expert_shards": (
                "Required when the model topology contains trainable expert groups.",
                "Stable-identity expert shard state with integrity evidence.",
            ),
            "routing_state": (
                "Required when the model topology contains trainable semantic routing.",
                "Router state, exposure counters, and optimizer-clock evidence.",
            ),
        }
        for kind, (condition, description) in conditional.items():
            if kind not in kinds:
                entries.append(
                    {
                        "kind": kind,
                        "required": False,
                        "component_scoped": True,
                        "condition": condition,
                        "description": description,
                    }
                )
    return sorted(entries, key=lambda item: str(item["kind"]))


def _resume(training: bool = True) -> dict[str, object]:
    if not training:
        return {"mode": "not_applicable", "required_state": []}
    return {
        "mode": "exact",
        "required_state": sorted(
            [
                "backend_environment_lock",
                "data_cursor",
                "dataset_version",
                "model_revision",
                "objective_state",
                "optimizer_state",
                "parameter_update_state",
                "rng_state",
                "scheduler_state",
                "trainable_component_state",
            ]
        ),
        "component_scoped_resume": True,
        "non_exact_resume_creates_lineage": True,
        "notes": [
            "Sparse objectives additionally require routing, expert exposure, and per-expert clock state."
        ],
    }


def _evaluation(*metrics: str, expert_metrics: bool = True) -> dict[str, object]:
    return {
        "before_run": False,
        "during_run": True,
        "after_run": True,
        "holdout_required": True,
        "gate_required": True,
        "metrics": sorted(metrics),
        "expert_system_metrics": (
            sorted(["expert_exposure", "load_balance", "routing_entropy", "routing_stability"])
            if expert_metrics
            else []
        ),
    }


def _hardware(
    *,
    compute: str = "high",
    device_memory: str = "high",
    host_memory: str = "medium",
    storage_io: str = "medium",
    communication: str = "unknown",
    implications: list[str] | None = None,
) -> dict[str, object]:
    return {
        "compute": compute,
        "device_memory": device_memory,
        "host_memory": host_memory,
        "storage_io": storage_io,
        "communication": communication,
        "fit_claim": "none",
        "implications": sorted(
            implications
            or [
                "Actual fit depends on the resolved model, sequence, batch, optimizer, topology, and host profile."
            ]
        ),
    }


def _make(
    *,
    objective_id: str,
    display_name: str,
    description: str,
    kind: str,
    execution_kind: str,
    task_type: str | None,
    dataset_inputs: list[dict[str, object]],
    labels: list[dict[str, object]],
    masks: list[dict[str, object]],
    losses: list[dict[str, object]],
    model: dict[str, object],
    adaptations: list[str],
    update: dict[str, object],
    backend: dict[str, object],
    artifacts: list[dict[str, object]],
    evaluation: dict[str, object],
    hardware: dict[str, object] | None = None,
    limitations: list[str] | None = None,
) -> TrainingObjective:
    raw: dict[str, object] = {
        "objective_id": objective_id,
        "objective_version": "1.0.0",
        "objective_hash": _ZERO_HASH,
        "display_name": display_name,
        "description": description,
        "kind": kind,
        "execution_kind": execution_kind,
        "coarse_task_type": task_type,
        "dataset_inputs": dataset_inputs,
        "labels": labels,
        "loss_masks": masks,
        "loss_components": sorted(losses, key=lambda item: str(item["component_id"])),
        "model_requirement": model,
        "adaptation_methods": sorted(adaptations),
        "update_policy": update,
        "backend_requirement": backend,
        "expected_artifacts": artifacts,
        "resume": _resume(execution_kind == "training"),
        "evaluation": evaluation,
        "hardware": hardware or _hardware(),
        "limitations": sorted(
            limitations
            or [
                "Registry presence is not proof that any backend implements or supports this objective."
            ]
        ),
        "verification": {
            "definition": "contract_validated",
            "implementation": "not_verified",
            "hardware": "not_verified",
            "evidence_refs": [],
        },
    }
    draft = TrainingObjective.model_validate(raw)
    sealed = draft.model_copy(update={"objective_hash": objective_hash_for(draft)})
    if not verify_objective_hash(sealed):  # pragma: no cover - defensive construction invariant
        raise ValueError(f"failed to seal objective {objective_id}")
    return sealed


def _causal_sft(
    *,
    objective_id: str,
    display_name: str,
    description: str,
    adaptations: list[str],
    update: dict[str, object],
    capability: str,
    quantizations: list[str] | None = None,
    variants: list[dict[str, object]] = _SFT,
    label_kind: str = "next_token",
    mask_kind: str = "all_non_padding",
    source_fields: list[str] | None = None,
    mask_construction: str = "Mask padding and compute loss on every remaining rendered token.",
) -> TrainingObjective:
    source = source_fields or ["rendered_sequence"]
    artifact_kind = "full_model" if "full_finetune" in adaptations else "adapter"
    return _make(
        objective_id=objective_id,
        display_name=display_name,
        description=description,
        kind="supervised_fine_tuning",
        execution_kind="training",
        task_type="sft",
        dataset_inputs=_dataset(variants),
        labels=_label(label_kind, source, "Shift the rendered target sequence by one token."),
        masks=_mask(mask_kind, source, mask_construction),
        losses=[_loss("cross_entropy", "Token-level causal language-model cross entropy.")],
        model=_model(_CAUSAL_MODEL),
        adaptations=adaptations,
        update=update,
        backend=_backend(
            "sft",
            losses=["cross_entropy", "liger_fused_ce"],
            adapters=adaptations,
            quantizations=quantizations,
            capabilities=["causal_lm_sft", capability],
        ),
        artifacts=_artifacts(artifact_kind, "checkpoint", "provenance_manifest"),
        evaluation=_evaluation("heldout_loss", "task_quality"),
    )


def _pretraining_objective(objective_id: str, display_name: str, continued: bool) -> TrainingObjective:
    capability = "continued_pretraining" if continued else "causal_lm_pretraining"
    return _make(
        objective_id=objective_id,
        display_name=display_name,
        description=(
            "Continue next-token training from an existing checkpoint on a pinned text corpus."
            if continued
            else "Train a causal language model from initialization with next-token prediction."
        ),
        kind="pretraining",
        execution_kind="training",
        task_type="pretraining",
        dataset_inputs=_dataset(_RAW_TEXT),
        labels=_label("next_token", ["text"], "Tokenize text and shift token IDs by one position."),
        masks=_mask(
            "all_non_padding", ["text"], "Mask padding and document-boundary positions as declared."
        ),
        losses=[_loss("cross_entropy", "Token-level next-token cross entropy.")],
        model=_model(_CAUSAL_MODEL),
        adaptations=["full_finetune"],
        update=_full_update(),
        backend=_backend(
            "pretraining",
            losses=["cross_entropy", "liger_fused_ce", "chunked_ce"],
            adapters=["full_finetune"],
            capabilities=[capability],
        ),
        artifacts=_artifacts("checkpoint", "full_model", "provenance_manifest"),
        evaluation=_evaluation("heldout_loss", "perplexity"),
        limitations=[
            "CorpusStudio does not currently integrate a pretraining backend.",
            "Registry presence is not proof that any backend implements or supports this objective.",
        ],
    )


def _preference_objective(
    objective_id: str, display_name: str, loss_impl: str, loss_kind: str
) -> TrainingObjective:
    reference = objective_id in {"dpo", "ipo"}
    label_construction = (
        "Split each pair into desirable and undesirable signed examples for the KTO target."
        if objective_id == "kto"
        else "Render the chosen and rejected continuations against the identical prompt."
    )
    labels = _label(
        "preference_pair", ["chosen", "prompt", "rejected"], label_construction
    )
    masks = _mask(
        "chosen_rejected",
        ["chosen", "prompt", "rejected"],
        "Mask prompt and padding positions consistently across both response branches.",
    )
    losses = [_loss(loss_kind, f"Construct the {display_name} pairwise preference loss.")]
    if objective_id == "orpo":
        labels = sorted(
            labels
            + [
                {
                    "label_id": "chosen_labels",
                    "kind": "next_token",
                    "source_fields": ["chosen", "prompt"],
                    "construction": "Shift the rendered prompt-plus-chosen response by one token.",
                    "ignore_index": -100,
                }
            ],
            key=lambda item: str(item["label_id"]),
        )
        masks = sorted(
            masks
            + [
                {
                    "mask_id": "chosen_response_mask",
                    "kind": "response_only",
                    "source_fields": ["chosen", "prompt"],
                    "include_padding": False,
                    "include_special_tokens": False,
                    "empty_mask_action": "reject",
                    "construction": "Compute supervised NLL only on the chosen response tokens.",
                }
            ],
            key=lambda item: str(item["mask_id"]),
        )
        losses = [
            _loss(
                "odds_ratio",
                "Construct the ORPO odds-ratio preference term.",
                component_id="odds_ratio_loss",
            ),
            _loss(
                "cross_entropy",
                "Construct the ORPO supervised chosen-response NLL term.",
                component_id="sft_loss",
                label_ref="chosen_labels",
                mask_ref="chosen_response_mask",
            ),
        ]
    return _make(
        objective_id=objective_id,
        display_name=display_name,
        description=f"Optimize chosen responses over rejected responses using {display_name}.",
        kind="preference_optimization",
        execution_kind="training",
        task_type="preference",
        dataset_inputs=_dataset(_PREFERENCE, role="preference"),
        labels=labels,
        masks=masks,
        losses=losses,
        model=_model(_CAUSAL_MODEL, reference=reference),
        adaptations=[],
        update=_full_update(),
        backend=_backend(
            "preference", losses=[loss_impl], capabilities=[f"preference_{objective_id}"]
        ),
        artifacts=_artifacts("checkpoint", "full_model", "provenance_manifest"),
        evaluation=_evaluation("heldout_preference_accuracy", "reward_margin"),
        limitations=[
            "Both built-in backend manifests currently declare SFT only.",
            "Registry presence is not proof that any backend implements or supports this objective.",
        ],
    )


def _distillation_objective(
    objective_id: str,
    display_name: str,
    loss_kind: str,
    planned_fields: dict[str, str],
) -> TrainingObjective:
    capability_suffix = objective_id.removesuffix("_distillation")
    variant = _variant(
        objective_id,
        availability="planned",
        dataset_format="distillation",
        **planned_fields,
    )
    return _make(
        objective_id=objective_id,
        display_name=display_name,
        description=f"Train a student from teacher evidence using {display_name.lower()}.",
        kind="distillation",
        execution_kind="training",
        task_type="distillation",
        dataset_inputs=_dataset([variant]),
        labels=_label(
            (
                "teacher_distribution"
                if objective_id in {"knowledge_distillation", "logit_distillation"}
                else "teacher_sequence"
            ),
            sorted(planned_fields),
            "Construct student targets from pinned teacher outputs and provenance.",
        ),
        masks=_mask(
            "labeled_positions",
            sorted(planned_fields),
            "Apply the loss only where teacher targets are present and valid.",
        ),
        losses=[_loss(loss_kind, f"Construct the {display_name.lower()} loss component.")],
        model=_model(_GENERATIVE_MODELS, reference=True),
        adaptations=[],
        update=_full_update(),
        backend=_backend(
            "distillation", capabilities=[f"distillation_{capability_suffix}"], losses=[]
        ),
        artifacts=_artifacts("checkpoint", "distillation_student", "provenance_manifest"),
        evaluation=_evaluation("student_teacher_gap", "task_quality"),
        limitations=[
            "The required distillation dataset shape is planned, not a built-in schema.",
            "No built-in backend currently declares this distillation capability.",
        ],
    )


def _special_training_objectives() -> list[TrainingObjective]:
    reward = _make(
        objective_id="reward_model",
        display_name="Reward model",
        description="Train a scalar reward head from chosen and rejected response pairs.",
        kind="reward_modeling",
        execution_kind="training",
        task_type="reward",
        dataset_inputs=_dataset(_PREFERENCE, role="preference"),
        labels=_label(
            "scalar_reward",
            ["chosen", "prompt", "rejected"],
            "Assign a higher scalar target to the chosen response than the rejected response.",
        ),
        masks=_mask(
            "chosen_rejected",
            ["chosen", "prompt", "rejected"],
            "Score each complete response while excluding padding.",
        ),
        losses=[_loss("reward_pairwise", "Bradley-Terry pairwise reward loss.")],
        model=_model(sorted(["causal_lm", "reward_model"]), reward_head=True),
        adaptations=[],
        update=_task_head_update(),
        backend=_backend("reward", losses=["reward_bt"], capabilities=["reward_model_pairwise"]),
        artifacts=_artifacts("checkpoint", "provenance_manifest", "reward_model"),
        evaluation=_evaluation("heldout_preference_accuracy", "reward_calibration", expert_metrics=False),
    )
    process = _make(
        objective_id="process_supervision",
        display_name="Process supervision",
        description="Supervise intermediate reasoning steps with per-step targets.",
        kind="process_supervision",
        execution_kind="training",
        task_type="classification",
        dataset_inputs=_dataset(
            [
                _variant(
                    "process_supervision",
                    availability="planned",
                    dataset_format="trace",
                    problem="text",
                    step_labels="list",
                    steps="list",
                )
            ]
        ),
        labels=_label(
            "trace_step", ["step_labels", "steps"], "Align each reasoning step with its target label."
        ),
        masks=_mask(
            "trace_steps", ["step_labels", "steps"], "Mask unlabeled or malformed trace steps."
        ),
        losses=[_loss("process_supervision", "Per-step supervised classification or scoring loss.")],
        model=_model(sorted(["causal_lm", "classification", "reward_model"])),
        adaptations=[],
        update=_full_update(),
        backend=_backend("classification", capabilities=["process_supervision"]),
        artifacts=_artifacts("checkpoint", "full_model", "provenance_manifest"),
        evaluation=_evaluation("process_accuracy", "step_calibration"),
    )
    verifier = _make(
        objective_id="verifier_training",
        display_name="Verifier training",
        description="Train a verifier to judge candidate answers or reasoning traces.",
        kind="verifier_training",
        execution_kind="training",
        task_type="classification",
        dataset_inputs=_dataset(
            [
                _variant(
                    "verifier",
                    availability="planned",
                    candidate="markdown",
                    label="string",
                    prompt="text",
                )
            ]
        ),
        labels=_label(
            "verifier_target", ["candidate", "label", "prompt"], "Map the declared verdict to a verifier target."
        ),
        masks=_mask("labeled_positions", ["label"], "Require one valid verdict for every candidate."),
        losses=[_loss("verifier", "Verifier classification or scalar scoring loss.")],
        model=_model(sorted(["classification", "reward_model"]), reward_head=True),
        adaptations=[],
        update=_task_head_update(),
        backend=_backend("classification", capabilities=["verifier_training"]),
        artifacts=_artifacts("checkpoint", "provenance_manifest", "verifier_model"),
        evaluation=_evaluation("calibration", "verifier_accuracy", expert_metrics=False),
    )
    tool_use = _make(
        objective_id="tool_use",
        display_name="Tool-use tuning",
        description="Train a causal model to emit and consume structured tool interactions.",
        kind="tool_use",
        execution_kind="training",
        task_type="sft",
        dataset_inputs=_dataset([_variant("chat", dataset_format="chat", messages="messages")]),
        labels=_label(
            "response_tokens", ["messages"], "Render assistant tool calls and final responses as targets."
        ),
        masks=_mask(
            "response_only", ["messages"], "Compute loss only on assistant tool-call and response spans."
        ),
        losses=[_loss("cross_entropy", "Response-token causal language-model cross entropy.")],
        model=_model(_CAUSAL_MODEL),
        adaptations=[],
        update=_full_update(),
        backend=_backend(
            "sft", losses=["cross_entropy"], capabilities=["response_only_mask", "tool_use_sft"]
        ),
        artifacts=_artifacts("checkpoint", "full_model", "provenance_manifest"),
        evaluation=_evaluation("argument_schema_validity", "tool_call_accuracy"),
    )
    embedding = _make(
        objective_id="embedding",
        display_name="Embedding training",
        description="Train an embedding model from positive and optional negative retrieval examples.",
        kind="embedding",
        execution_kind="training",
        task_type="embedding",
        dataset_inputs=_dataset([_variant("retrieval", positive="text", query="text")]),
        labels=_label(
            "contrastive_pair", ["negative", "positive", "query"], "Construct positive and negative query-document pairs."
        ),
        masks=_mask("labeled_positions", ["positive", "query"], "Require the query and positive document."),
        losses=[_loss("contrastive", "Contrastive embedding loss over query-document pairs.")],
        model=_model(["embedding"], output_head=False),
        adaptations=[],
        update=_full_update(),
        backend=_backend("embedding", capabilities=["embedding_contrastive"]),
        artifacts=_artifacts("checkpoint", "embedding_model", "provenance_manifest"),
        evaluation=_evaluation("mrr", "recall_at_k", expert_metrics=False),
    )
    reranker = _make(
        objective_id="reranker",
        display_name="Reranker training",
        description="Train a cross-encoder or ranking head over query-document candidates.",
        kind="reranking",
        execution_kind="training",
        task_type="embedding",
        dataset_inputs=_dataset([_variant("retrieval", negative="text", positive="text", query="text")]),
        labels=_label(
            "preference_pair", ["negative", "positive", "query"], "Assign the positive document a higher relevance target."
        ),
        masks=_mask("labeled_positions", ["positive", "query"], "Require a query and ranked candidates."),
        losses=[_loss("ranking", "Pairwise or listwise ranking loss.")],
        model=_model(["reranker"]),
        adaptations=[],
        update=_task_head_update(),
        backend=_backend("embedding", capabilities=["reranker_training"]),
        artifacts=_artifacts("checkpoint", "provenance_manifest", "reranker_model"),
        evaluation=_evaluation("mrr", "ndcg", expert_metrics=False),
    )
    classifier = _make(
        objective_id="classifier",
        display_name="Classifier training",
        description="Train a classification head from text and class labels.",
        kind="classification",
        execution_kind="training",
        task_type="classification",
        dataset_inputs=_dataset([_variant("classification", label="string", text="text")]),
        labels=_label("class_id", ["label"], "Map the pinned label vocabulary to class IDs."),
        masks=_mask("labeled_positions", ["label"], "Reject examples without a valid class label."),
        losses=[_loss("classification", "Supervised class cross entropy.")],
        model=_model(["classification"]),
        adaptations=[],
        update=_task_head_update(),
        backend=_backend("classification", capabilities=["classifier_training"]),
        artifacts=_artifacts("checkpoint", "classifier_model", "provenance_manifest"),
        evaluation=_evaluation("accuracy", "calibration", "f1", expert_metrics=False),
    )
    multimodal = _make(
        objective_id="multimodal",
        display_name="Multimodal training",
        description="Train a multimodal model from paired image and caption targets.",
        kind="multimodal",
        execution_kind="training",
        task_type="multimodal",
        dataset_inputs=_dataset([_variant("image_caption", caption="text", image="image_path")]),
        labels=_label(
            "multimodal_target", ["caption", "image"], "Encode the image and construct caption token targets."
        ),
        masks=_mask(
            "multimodal_target", ["caption", "image"], "Compute target loss only for valid image-caption pairs."
        ),
        losses=[_loss("cross_entropy", "Caption-token cross entropy conditioned on visual features.")],
        model=_model(["multimodal"], multimodal_projector=True),
        adaptations=[],
        update=_task_head_update(projector=True),
        backend=_backend("multimodal", losses=["cross_entropy"], capabilities=["multimodal_sft"]),
        artifacts=_artifacts("checkpoint", "multimodal_model", "provenance_manifest"),
        evaluation=_evaluation("caption_quality", "grounding", expert_metrics=False),
    )
    return [reward, process, verifier, tool_use, embedding, reranker, classifier, multimodal]


def _non_training_objective(
    objective_id: str,
    display_name: str,
    kind: str,
    capability: str,
    artifact: str,
) -> TrainingObjective:
    is_evaluation = kind == "evaluation"
    return _make(
        objective_id=objective_id,
        display_name=display_name,
        description=(
            "Evaluate a pinned model or artifact without updating parameters."
            if is_evaluation
            else f"Perform a provenance-preserving {display_name.lower()} artifact operation."
        ),
        kind=kind,
        execution_kind="evaluation" if is_evaluation else "artifact_operation",
        task_type="evaluation" if is_evaluation else None,
        dataset_inputs=(
            _dataset(
                [
                    _variant(
                        "evaluation",
                        dataset_format="evaluation",
                        expected_answer="markdown",
                        id="string",
                        prompt="text",
                    )
                ],
                role="evaluation",
            )
            if is_evaluation
            else []
        ),
        labels=[],
        masks=[],
        losses=[],
        model=_model(
            _ALL_KNOWN_MODEL_TASKS,
            tokenizer=is_evaluation,
            output_head=is_evaluation,
        ),
        adaptations=[],
        update=_no_update(),
        backend=_backend(
            "evaluation" if is_evaluation else None,
            capabilities=[capability],
            hardware_required=is_evaluation,
        ),
        artifacts=_artifacts(artifact, "provenance_manifest"),
        evaluation=(
            {
                "before_run": False,
                "during_run": False,
                "after_run": True,
                "holdout_required": True,
                "gate_required": True,
                "metrics": ["suite_defined"],
                "expert_system_metrics": [],
            }
            if is_evaluation
            else {
                "before_run": False,
                "during_run": False,
                "after_run": True,
                "holdout_required": False,
                "gate_required": True,
                "metrics": ["artifact_integrity", "reload_equivalence"],
                "expert_system_metrics": [],
            }
        ),
        hardware=_hardware(
            compute="medium" if is_evaluation else "low",
            device_memory="medium" if is_evaluation else "unknown",
            host_memory="medium",
            storage_io="medium",
            communication="none",
        ),
    )


def _build_catalog() -> tuple[TrainingObjective, ...]:
    objectives: list[TrainingObjective] = [
        _pretraining_objective("pretraining", "Pretraining", False),
        _pretraining_objective("continued_pretraining", "Continued pretraining", True),
        _causal_sft(
            objective_id="full_parameter_sft",
            display_name="Full-parameter SFT",
            description="Supervised causal-LM fine-tuning with all logical model parameters trainable.",
            adaptations=["full_finetune"],
            update=_full_update(),
            capability="full_parameter_sft",
        ),
        _causal_sft(
            objective_id="lora",
            display_name="LoRA",
            description="Supervised causal-LM fine-tuning through low-rank adapters.",
            adaptations=["lora"],
            update=_adapter_update(),
            capability="adapter_lora",
        ),
        _causal_sft(
            objective_id="qlora",
            display_name="QLoRA",
            description="Supervised causal-LM adapter tuning with a quantized frozen base model.",
            adaptations=["qlora"],
            update=_adapter_update(),
            capability="adapter_qlora",
            quantizations=["int4", "nf4"],
        ),
        _causal_sft(
            objective_id="other_peft",
            display_name="Other PEFT",
            description="Supervised fine-tuning through a declared non-LoRA parameter-efficient method.",
            adaptations=sorted(["dora", "ia3", "prefix_tuning", "prompt_tuning"]),
            update=_adapter_update(),
            capability="adapter_other_peft",
        ),
        _causal_sft(
            objective_id="chat_tuning",
            display_name="Chat tuning",
            description="Supervised tuning over rendered multi-turn chat transcripts.",
            adaptations=[],
            update=_full_update(),
            capability="chat_sft",
            variants=[_variant("chat", dataset_format="chat", messages="messages")],
        ),
        _causal_sft(
            objective_id="completion_only",
            display_name="Completion-only tuning",
            description="Supervised tuning whose loss excludes prompt tokens.",
            adaptations=[],
            update=_full_update(),
            capability="completion_only_mask",
            variants=_COMPLETION,
            label_kind="completion_tokens",
            mask_kind="completion_only",
            source_fields=["completion_boundary", "rendered_sequence"],
            mask_construction="Mask prompt, padding, and non-target special tokens.",
        ),
        _causal_sft(
            objective_id="response_only_loss",
            display_name="Response-only loss",
            description="Chat tuning whose loss includes assistant responses but excludes user/system/tool inputs.",
            adaptations=[],
            update=_full_update(),
            capability="response_only_mask",
            variants=[_variant("chat", dataset_format="chat", messages="messages")],
            label_kind="response_tokens",
            mask_kind="response_only",
            source_fields=["assistant_spans", "rendered_sequence"],
            mask_construction="Mask every token outside assistant response spans.",
        ),
        _preference_objective("dpo", "DPO", "dpo", "preference"),
        _preference_objective("ipo", "IPO", "ipo", "preference"),
        _preference_objective("kto", "KTO", "kto", "preference"),
        _preference_objective("orpo", "ORPO", "orpo", "odds_ratio"),
        _distillation_objective(
            "knowledge_distillation",
            "Knowledge distillation",
            "knowledge_distillation",
            {"input": "text", "teacher_targets": "object"},
        ),
        _distillation_objective(
            "sequence_distillation",
            "Sequence distillation",
            "sequence_distillation",
            {"input": "text", "teacher_sequence": "text"},
        ),
        _distillation_objective(
            "logit_distillation",
            "Logit distillation",
            "logit_distillation",
            {"input_ids": "list", "teacher_logits": "object"},
        ),
        _distillation_objective(
            "rationale_distillation",
            "Rationale distillation",
            "rationale_distillation",
            {"answer": "text", "prompt": "text", "rationale": "text"},
        ),
        _non_training_objective(
            "evaluation_only", "Evaluation-only", "evaluation", "evaluation_suite", "evaluation_result"
        ),
        _non_training_objective("merge_only", "Merge-only", "merge", "artifact_merge", "merged_model"),
        _non_training_objective(
            "conversion_only", "Conversion-only", "conversion", "artifact_conversion", "converted_model"
        ),
        _non_training_objective(
            "quantization_only",
            "Quantization-only",
            "quantization",
            "artifact_quantization",
            "quantized_model",
        ),
    ]
    objectives.extend(_special_training_objectives())
    ordered = tuple(sorted(objectives, key=lambda item: (item.objective_id, item.objective_version)))
    keys = [(item.objective_id, item.objective_version) for item in ordered]
    if len(keys) != len(set(keys)):  # pragma: no cover - static catalog construction guard
        raise ValueError("duplicate objective id/version in built-in registry")
    if len(ordered) != 29:  # pragma: no cover - guards the requested catalog surface
        raise ValueError(f"expected 29 built-in objectives, got {len(ordered)}")
    return ordered


def objective_hash_for(objective: TrainingObjective) -> str:
    """Canonical SHA-256 seal over the objective definition, excluding the seal itself."""

    payload = objective.model_dump(mode="json", exclude={"objective_hash"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_objective_hash(objective: TrainingObjective) -> bool:
    return objective.objective_hash == objective_hash_for(objective)


def validate_objective_catalog(
    objectives: Iterable[TrainingObjective],
) -> tuple[TrainingObjective, ...]:
    """Validate seals and reject duplicate id/version pairs before a catalog is accepted."""

    ordered = tuple(sorted(objectives, key=lambda item: (item.objective_id, item.objective_version)))
    keys = [(item.objective_id, item.objective_version) for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate objective id/version")
    for objective in ordered:
        if not verify_objective_hash(objective):
            raise ValueError(f"objective hash mismatch: {objective.objective_id}")
    return ordered


_BUILTIN_OBJECTIVES = validate_objective_catalog(_build_catalog())


def builtin_objectives() -> list[TrainingObjective]:
    """Return deterministic deep copies of every built-in objective definition."""

    for objective in _BUILTIN_OBJECTIVES:
        if not verify_objective_hash(objective):  # pragma: no cover - immutable module constant
            raise ValueError(f"objective hash mismatch: {objective.objective_id}")
    return [item.model_copy(deep=True) for item in _BUILTIN_OBJECTIVES]


def get_objective(objective_id: str, version: str | None = None) -> TrainingObjective | None:
    """Return one objective. With no version, return the highest registered semantic version."""

    matches = [item for item in _BUILTIN_OBJECTIVES if item.objective_id == objective_id]
    if version is not None:
        matches = [item for item in matches if item.objective_version == version]
    if not matches:
        return None
    selected = max(
        matches,
        key=lambda item: tuple(int(part) for part in item.objective_version.split(".")),
    )
    return selected.model_copy(deep=True)


def _axis(
    status: ObjectiveCompatibilityStatus,
    *,
    reasons: list[str] | None = None,
    evidence: list[str] | None = None,
) -> ObjectiveCompatibilityAxis:
    return ObjectiveCompatibilityAxis(
        status=status,
        reasons=sorted(set(reasons or [])),
        evidence=sorted(set(evidence or [])),
    )


def _check_dataset(
    objective: TrainingObjective,
    schema_id: str | None,
    schema_version: str | None,
    fields: Mapping[str, str] | None,
) -> ObjectiveCompatibilityAxis:
    if not objective.dataset_inputs:
        return _axis(ObjectiveCompatibilityStatus.not_applicable)
    if len(objective.dataset_inputs) > 1:
        roles = sorted(item.role for item in objective.dataset_inputs)
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=[
                "role-keyed evidence is required for multi-input objectives; "
                f"this checker received one flattened dataset input for roles {roles}"
            ],
        )
    if schema_id is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["dataset schema evidence was not provided"],
        )
    variants = [
        variant
        for variant in objective.dataset_inputs[0].variants
        if variant.schema_id == schema_id
    ]
    if not variants:
        accepted = sorted({variant.schema_id for variant in objective.dataset_inputs[0].variants})
        return _axis(
            ObjectiveCompatibilityStatus.incompatible,
            reasons=[f"schema '{schema_id}' is not accepted; expected one of {accepted}"],
        )
    unpinned_variants = [variant for variant in variants if variant.schema_version is None]
    pinned_versions = sorted(
        {variant.schema_version for variant in variants if variant.schema_version is not None}
    )
    if schema_version is None and not unpinned_variants:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=[f"dataset schema version evidence was not provided for '{schema_id}'"],
            evidence=[f"objective accepts schema {schema_id} versions {pinned_versions}"],
        )
    exact_versions = [
        variant for variant in variants if variant.schema_version == schema_version
    ]
    matching_variants = exact_versions or unpinned_variants
    if not matching_variants:
        return _axis(
            ObjectiveCompatibilityStatus.incompatible,
            reasons=[
                f"schema '{schema_id}' version '{schema_version}' is not accepted; "
                f"expected one of {pinned_versions}"
            ],
        )
    variant = matching_variants[0]
    if variant.availability.value == "planned":
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=[f"schema '{schema_id}' is specified but not shipped as a built-in schema"],
            evidence=[f"objective variant {schema_id}@{variant.schema_version or 'unpinned'}"],
        )
    if fields is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["dataset field evidence was not provided"],
            evidence=[f"accepted schema id {schema_id}"],
        )
    missing = sorted(item.name for item in variant.required_fields if item.name not in fields)
    mismatched = sorted(
        f"{item.name}: expected {item.field_type}, got {fields[item.name]}"
        for item in variant.required_fields
        if item.name in fields and item.field_type is not None and fields[item.name] != item.field_type
    )
    if missing or mismatched:
        reasons = [f"missing required field '{name}'" for name in missing]
        reasons.extend(f"field type mismatch for {item}" for item in mismatched)
        return _axis(ObjectiveCompatibilityStatus.incompatible, reasons=reasons)
    return _axis(
        ObjectiveCompatibilityStatus.verified_compatible,
        evidence=[f"schema {schema_id} contains every required field with matching declared types"],
    )


def _check_model(
    objective: TrainingObjective, model: ModelDescriptor | None
) -> ObjectiveCompatibilityAxis:
    if model is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["model descriptor evidence was not provided"],
        )
    requirement = objective.model_requirement
    task_classes = set(model.task_classes)
    if ModelTaskClass.unknown in task_classes:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["model task class is unknown"],
            evidence=[f"model descriptor {model.model_id}"],
        )
    if not task_classes.intersection(requirement.task_classes):
        return _axis(
            ObjectiveCompatibilityStatus.incompatible,
            reasons=[
                "model task classes do not intersect objective requirements: "
                f"model={sorted(item.value for item in task_classes)} "
                f"required={sorted(item.value for item in requirement.task_classes)}"
            ],
        )
    execution_kind = model.topology.execution_kind
    if execution_kind == ModelExecutionKind.unknown:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["model execution topology is unknown"],
            evidence=[f"model descriptor {model.model_id}"],
        )
    if execution_kind not in requirement.execution_kinds:
        return _axis(
            ObjectiveCompatibilityStatus.incompatible,
            reasons=[f"model execution kind '{execution_kind.value}' is not accepted"],
        )
    reasons: list[str] = []
    gaps: list[str] = []
    if requirement.requires_tokenizer and model.tokenizer_ref is None:
        reasons.append("objective requires a linked tokenizer descriptor")
    if requirement.requires_output_head and model.vocabulary.output_head_rows is None:
        gaps.append("output-head dimension evidence is missing")
    if requirement.requires_reference_model:
        gaps.append("reference-model descriptor evidence was not provided")
    if requirement.requires_reward_head and ModelTaskClass.reward_model not in task_classes:
        gaps.append("reward-head construction or descriptor evidence is missing")
    if requirement.requires_multimodal_projector:
        gaps.append("multimodal-projector construction or descriptor evidence is missing")
    if model.trust.custom_code_required:
        if requirement.custom_code_policy == "forbid":
            reasons.append("objective forbids the model's required custom code")
        else:
            gaps.append("custom code requires separate hash-pinned approval in an isolated worker")
    scopes = set(objective.update_policy.scopes)
    expert_scopes = {ObjectiveUpdateScope.selected_experts, ObjectiveUpdateScope.all_experts}
    if scopes.intersection(expert_scopes):
        if execution_kind == ModelExecutionKind.dense:
            reasons.append("expert-scoped updates cannot target a dense model")
        elif not model.topology.expert_groups:
            gaps.append("expert-scoped updates require parsed expert groups")
        elif any(
            not group.expert_identity_scheme and group.expert_registry_ref is None
            for group in model.topology.expert_groups
        ):
            gaps.append("expert-scoped updates require stable expert identity evidence")
        if (
            objective.update_policy.selection_mode == ObjectiveSelectionMode.routed_experts
            and model.topology.semantic_routing is None
        ):
            gaps.append("routed-expert updates require semantic routing evidence")
    if ObjectiveUpdateScope.router in scopes:
        if execution_kind == ModelExecutionKind.dense:
            reasons.append("router updates cannot target a dense model")
        elif model.topology.semantic_routing is None:
            gaps.append("router updates require semantic routing evidence")
    if reasons:
        return _axis(ObjectiveCompatibilityStatus.incompatible, reasons=reasons)
    if gaps:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=gaps,
            evidence=[f"model descriptor {model.model_id}"],
        )
    return _axis(
        ObjectiveCompatibilityStatus.verified_compatible,
        evidence=[f"model descriptor {model.model_id} satisfies structural requirements"],
    )


def _check_backend(
    objective: TrainingObjective,
    backend: BackendManifest | None,
    capability: CapabilityReport | None,
) -> ObjectiveCompatibilityAxis:
    if backend is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["backend manifest evidence was not provided"],
        )
    requirement = objective.backend_requirement
    reasons: list[str] = []
    if requirement.task_type is not None and requirement.task_type not in backend.task_types:
        reasons.append(f"backend does not declare task type '{requirement.task_type.value}'")
    if requirement.loss_impls and not set(requirement.loss_impls).intersection(backend.loss_impls):
        reasons.append("backend declares none of the objective's accepted loss implementations")
    if requirement.adaptation_methods and not set(requirement.adaptation_methods).intersection(
        backend.adapter_methods
    ):
        reasons.append("backend declares none of the objective's accepted adaptation methods")
    if requirement.quantization_modes and not set(requirement.quantization_modes).intersection(
        backend.quantization_modes
    ):
        reasons.append("backend declares none of the objective's accepted quantization modes")
    missing_capabilities = sorted(
        set(requirement.objective_capabilities) - set(backend.objective_capabilities)
    )
    if missing_capabilities:
        reasons.append(f"backend does not declare objective capabilities {missing_capabilities}")
    if reasons:
        return _axis(ObjectiveCompatibilityStatus.incompatible, reasons=reasons)
    declared_evidence = [f"backend manifest {backend.backend_id}@{backend.backend_version}"]
    if capability is None:
        return _axis(
            ObjectiveCompatibilityStatus.declared_compatible,
            evidence=declared_evidence,
        )
    if capability.backend_id != backend.backend_id:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["capability report backend does not match the selected manifest"],
            evidence=declared_evidence,
        )
    if capability.backend_version is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["capability report does not pin the selected backend version"],
            evidence=declared_evidence,
        )
    if capability.backend_version != backend.backend_version:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=[
                "capability report backend version does not match the selected manifest: "
                f"report={capability.backend_version} manifest={backend.backend_version}"
            ],
            evidence=declared_evidence,
        )
    if capability.readiness == "not_ready":
        return _axis(
            ObjectiveCompatibilityStatus.incompatible,
            reasons=["capability report says the backend is not ready on this environment"],
        )
    if capability.readiness == "cpu_toy_only":
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["capability report proves only the CPU toy runner"],
            evidence=declared_evidence,
        )
    effective = capability.effective_capabilities
    if effective is None:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=["capability report has no effective-capabilities evidence"],
            evidence=declared_evidence,
        )
    gaps: list[str] = []
    if requirement.adaptation_methods and not set(requirement.adaptation_methods).intersection(
        effective.adapter_methods
    ):
        gaps.append("required adaptation method was not functionally proven")
    if requirement.quantization_modes and not set(requirement.quantization_modes).intersection(
        effective.quantization_modes
    ):
        gaps.append("required quantization mode was not functionally proven")
    missing_effective = sorted(
        set(requirement.objective_capabilities) - set(effective.objective_capabilities)
    )
    if missing_effective:
        gaps.append(f"objective capabilities were not functionally proven: {missing_effective}")
    if gaps:
        return _axis(
            ObjectiveCompatibilityStatus.unverified,
            reasons=gaps,
            evidence=declared_evidence
            + [f"capability report for {capability.environment_ref.id}"],
        )
    return _axis(
        ObjectiveCompatibilityStatus.verified_compatible,
        evidence=declared_evidence
        + [f"capability report for {capability.environment_ref.id}"],
    )


def _overall(*axes: ObjectiveCompatibilityAxis) -> ObjectiveCompatibilityStatus:
    statuses = {axis.status for axis in axes}
    if statuses == {ObjectiveCompatibilityStatus.not_applicable}:
        return ObjectiveCompatibilityStatus.not_applicable
    if ObjectiveCompatibilityStatus.incompatible in statuses:
        return ObjectiveCompatibilityStatus.incompatible
    if ObjectiveCompatibilityStatus.unverified in statuses:
        return ObjectiveCompatibilityStatus.unverified
    if ObjectiveCompatibilityStatus.declared_compatible in statuses:
        return ObjectiveCompatibilityStatus.declared_compatible
    return ObjectiveCompatibilityStatus.verified_compatible


def check_objective_compatibility(
    objective: TrainingObjective,
    *,
    dataset_schema_id: str | None = None,
    dataset_schema_version: str | None = None,
    dataset_fields: Mapping[str, str] | None = None,
    model_descriptor: ModelDescriptor | None = None,
    backend_manifest: BackendManifest | None = None,
    capability_report: CapabilityReport | None = None,
) -> ObjectiveCompatibilityReport:
    """Check dataset, model, and backend evidence independently.

    This is a structural and capability-evidence check, not a fit prediction. Static backend metadata
    can earn only ``declared_compatible``; current capability reports intentionally do not fabricate
    end-to-end objective capabilities from package imports or lower-level kernel probes.
    """

    if not verify_objective_hash(objective):
        raise ValueError(f"objective hash mismatch: {objective.objective_id}")
    dataset_axis = _check_dataset(
        objective,
        dataset_schema_id,
        dataset_schema_version,
        dataset_fields,
    )
    model_axis = _check_model(objective, model_descriptor)
    backend_axis = _check_backend(objective, backend_manifest, capability_report)
    return ObjectiveCompatibilityReport(
        objective_ref=Ref(
            id=objective.objective_id,
            hash=HashRef(algo="sha256", value=objective.objective_hash),
        ),
        objective_version=objective.objective_version,
        dataset_schema_id=dataset_schema_id,
        dataset_schema_version=dataset_schema_version,
        model_id=model_descriptor.model_id if model_descriptor else None,
        backend_id=backend_manifest.backend_id if backend_manifest else None,
        capability_environment_ref=(
            capability_report.environment_ref if capability_report is not None else None
        ),
        dataset=dataset_axis,
        model=model_axis,
        backend=backend_axis,
        overall_status=_overall(dataset_axis, model_axis, backend_axis),
        notes=[
            "Compatibility does not predict hardware fit; only a measured run can prove fit.",
            "Dataset compatibility validates the declared schema id, version, and shape, not every dataset row.",
        ],
    )
