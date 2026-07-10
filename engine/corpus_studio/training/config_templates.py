"""Training config template models.

The current module generates inspectable config data only. It does not launch
training or depend on CUDA, PyTorch, Transformers, or trainer packages.
"""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import BaseModel, Field


TrainingConfigTarget = Literal[
    # Corpus Studio's own first-party trainer (opt-in ``[train]`` extra): the config is run in-process
    # by ``train-run`` — no external trainer needed. Renders as JSON (the shape ``train-run`` reads).
    "corpus_studio",
    "axolotl_yaml",
    "trl_config",
    "unsloth_script",
    "huggingface_trainer",
    "llama_factory",
]


class TrainingConfigTemplate(BaseModel):
    """Shared training config shape for future target-specific exporters."""

    target: TrainingConfigTarget
    base_model: str
    dataset_path: str
    eval_dataset_path: str | None = None
    format: str
    # Where the trainer writes checkpoints/adapters; relative paths resolve
    # against the directory the trainer is launched from (the config's dir).
    output_dir: str = "output"
    sequence_len: int = Field(default=4096, gt=0)
    adapter: str = "lora"
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=0.0002, gt=0)
    # Emitted into every target's config so weight initialisation, data shuffling, and
    # dropout are deterministic — this is what makes a run reproducible (the run's
    # provenance manifest hashes the config, so the seed is pinned with it). All the
    # supported trainers (axolotl / TRL / transformers / unsloth / llama-factory) read a
    # top-level `seed`. A fixed default (not a random one) keeps runs reproducible by default.
    seed: int = Field(default=42, ge=0)

    def to_training_dict(self) -> dict[str, object]:
        """Return a serializable config dictionary."""

        return self.model_dump(exclude_none=True)


def build_lora_config_template(
    base_model: str,
    dataset_path: str,
    eval_dataset_path: str | None,
    dataset_format: str,
    target: TrainingConfigTarget = "axolotl_yaml",
    output_dir: str = "output",
    sequence_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 32,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 0.0002,
    seed: int = 42,
) -> TrainingConfigTemplate:
    """Build a conservative LoRA config template.

    TODO: Add richer target-specific compatibility checks in v0.4 after the
    export matrix is finalized.
    """

    return TrainingConfigTemplate(
        target=target,
        base_model=base_model,
        dataset_path=dataset_path,
        eval_dataset_path=eval_dataset_path,
        format=dataset_format,
        output_dir=output_dir,
        sequence_len=sequence_len,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
    )


def normalize_training_config_target(target: str) -> TrainingConfigTarget:
    """Normalize UI/CLI target names to the shared config target values."""

    normalized = target.strip().lower().replace("-", "_")
    aliases = {
        "corpus_studio": "corpus_studio",
        "corpusstudio": "corpus_studio",
        "corpus": "corpus_studio",
        "first_party": "corpus_studio",
        "firstparty": "corpus_studio",
        "axolotl": "axolotl_yaml",
        "axolotl_yaml": "axolotl_yaml",
        "trl": "trl_config",
        "trl_config": "trl_config",
        "unsloth": "unsloth_script",
        "unsloth_script": "unsloth_script",
        "huggingface": "huggingface_trainer",
        "huggingface_trainer": "huggingface_trainer",
        "hf_trainer": "huggingface_trainer",
        "llama_factory": "llama_factory",
        "llamafactory": "llama_factory",
    }

    try:
        return cast(TrainingConfigTarget, aliases[normalized])
    except KeyError as exc:
        supported = ", ".join(sorted(set(aliases)))
        raise ValueError(f"Unsupported training config target. Use one of: {supported}.") from exc


def training_config_file_extension(target: TrainingConfigTarget) -> str:
    """Return a default file extension for a rendered config target."""

    if target in {"axolotl_yaml", "llama_factory"}:
        return ".yaml"
    if target == "unsloth_script":
        return ".py"
    return ".json"


def render_training_config(template: TrainingConfigTemplate) -> str:
    """Render an inspectable target config without importing trainer packages."""

    payload = template.to_training_dict()
    if template.target in {"axolotl_yaml", "llama_factory"}:
        return _render_yaml(payload)
    if template.target == "unsloth_script":
        return _render_python_config(payload)
    return json.dumps(payload, indent=2) + "\n"


def _render_yaml(payload: dict[str, object]) -> str:
    return "".join(f"{key}: {_format_yaml_value(value)}\n" for key, value in payload.items())


def _format_yaml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value)


def _render_python_config(payload: dict[str, object]) -> str:
    return "TRAINING_CONFIG = " + json.dumps(payload, indent=2) + "\n"
