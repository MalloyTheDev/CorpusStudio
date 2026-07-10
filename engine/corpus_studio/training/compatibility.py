"""Training config compatibility checks.

These helpers flag likely schema/format/target mismatches before a user hands a
generated config to a trainer. They only produce advisory warnings; they never
block config export and never launch training.
"""

from __future__ import annotations

from corpus_studio.training.config_templates import TrainingConfigTarget


# The training "style" a built-in schema naturally maps to.
SCHEMA_TRAINING_STYLE: dict[str, str] = {
    "raw_text": "pretraining",
    "instruction": "sft",
    "chat": "sft",
    "code": "sft",
    "preference": "preference",
    "image_caption": "multimodal",
    "classification": "classification",
    "retrieval": "embedding",
    "evaluation": "evaluation",
}

# Format labels that are reasonable for each schema's data shape.
SCHEMA_COMPATIBLE_FORMATS: dict[str, set[str]] = {
    "raw_text": {"raw_text", "completion", "text", "pretraining"},
    "instruction": {"instruction", "alpaca", "sft"},
    "chat": {"chat", "sharegpt", "messages", "conversation"},
    "code": {"code", "instruction", "sft"},
    "preference": {"preference", "dpo", "orpo", "kto", "reward"},
    "image_caption": {"image_caption", "vision", "multimodal", "caption"},
    "classification": {"classification", "text_classification"},
    "retrieval": {"retrieval", "embedding", "sentence_transformers"},
    "evaluation": {"evaluation"},
}

# Training styles each target can express with the LoRA causal-LM template this
# module renders. Styles outside these sets need a different pipeline.
TARGET_SUPPORTED_STYLES: dict[TrainingConfigTarget, set[str]] = {
    # The first-party trainer is a TRL SFTTrainer + peft LoRA under the hood — same SFT surface as trl.
    "corpus_studio": {"sft"},
    "axolotl_yaml": {"sft", "pretraining", "preference"},
    "trl_config": {"sft", "preference"},
    "unsloth_script": {"sft", "preference"},
    "huggingface_trainer": {"sft", "classification"},
    "llama_factory": {"sft", "pretraining", "preference"},
}

# Styles that are not causal-LM fine-tuning at all.
_NON_CAUSAL_LM_STYLES = {"multimodal", "embedding", "evaluation"}


def training_compatibility_warnings(
    *,
    schema_id: str,
    dataset_format: str,
    target: TrainingConfigTarget,
) -> list[str]:
    """Return advisory warnings for a schema/format/target combination."""

    warnings: list[str] = []
    style = SCHEMA_TRAINING_STYLE.get(schema_id)

    warnings.extend(_format_warnings(schema_id, dataset_format))

    if style is None:
        return warnings

    supported_styles = TARGET_SUPPORTED_STYLES.get(target, set())

    if style in _NON_CAUSAL_LM_STYLES:
        warnings.append(
            f"The {schema_id} schema is a {style} dataset, not causal-LM fine-tuning "
            f"data. This LoRA config targets causal language models and likely needs "
            f"a different trainer."
        )
        return warnings

    if style == "preference":
        if "preference" in supported_styles:
            warnings.append(
                f"Preference datasets need a DPO/reward pipeline. Configure {target}'s "
                f"preference/DPO trainer; this template renders generic LoRA SFT fields "
                f"only."
            )
        else:
            warnings.append(
                f"{target} has no built-in preference/DPO path; a custom preference "
                f"trainer is required for the {schema_id} schema."
            )
        return warnings

    if style == "classification" and "classification" not in supported_styles:
        warnings.append(
            f"The {schema_id} schema is classification data; {target} is oriented "
            f"toward generative fine-tuning and may need a sequence-classification "
            f"trainer instead."
        )
        return warnings

    if style == "pretraining" and "pretraining" not in supported_styles:
        warnings.append(
            f"The {schema_id} schema is raw pretraining text; {target} typically "
            f"expects supervised fine-tuning data. Use a completion/pretraining "
            f"pipeline for this data."
        )
        return warnings

    return warnings


def _format_warnings(schema_id: str, dataset_format: str) -> list[str]:
    compatible = SCHEMA_COMPATIBLE_FORMATS.get(schema_id)
    if compatible is None:
        return []

    normalized = dataset_format.strip().lower().replace("-", "_")
    if not normalized or normalized in compatible:
        return []

    expected = ", ".join(sorted(compatible))
    return [
        f"Dataset format '{dataset_format}' is unusual for the {schema_id} schema; "
        f"expected one of: {expected}."
    ]
