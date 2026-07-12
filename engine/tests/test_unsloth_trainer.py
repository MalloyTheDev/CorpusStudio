"""The Unsloth training backend. The real training needs a CUDA GPU + the unsloth package (neither in
CI), so the heavy body is user-smoke-tested only; here we prove the two things that MUST be right on a
core-only install: the config→Unsloth-args mapping (pure), and that a missing Unsloth degrades to a
clean TrainerError (which the platform classifies as ENVIRONMENT_FAILURE) rather than crashing."""

import pytest

from corpus_studio.training.trainer import TrainerError, TrainRunConfig
from corpus_studio.training.unsloth_trainer import build_unsloth_kwargs, run_unsloth_training

_CONFIG = TrainRunConfig(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="data/train.jsonl",
    output_dir="out",
    dataset_format="chat",
    sequence_len=2048,
    lora_r=16,
    lora_alpha=32,
    seed=7,
)


def test_build_unsloth_kwargs_maps_from_pretrained():
    fp = build_unsloth_kwargs(_CONFIG)["from_pretrained"]
    assert fp["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert fp["max_seq_length"] == 2048
    assert fp["load_in_4bit"] is True  # Unsloth's headline 4-bit QLoRA
    assert fp["dtype"] is None  # auto-select bf16/fp16


def test_build_unsloth_kwargs_maps_the_lora_adapter():
    peft = build_unsloth_kwargs(_CONFIG)["get_peft_model"]
    assert peft["r"] == 16
    assert peft["lora_alpha"] == 32
    assert peft["random_state"] == 7
    assert peft["use_gradient_checkpointing"] == "unsloth"
    assert peft["bias"] == "none"
    # the standard Llama/Qwen attention+MLP projection set
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= set(
        peft["target_modules"]
    )


def test_run_unsloth_training_without_unsloth_raises_a_clean_trainererror():
    # Unsloth is not installed in the engine env (dependency-light), so the lazy import fails and the
    # runner gets a clean, classifiable TrainerError — not an ImportError leaking out.
    with pytest.raises(TrainerError) as excinfo:
        run_unsloth_training(_CONFIG)
    message = str(excinfo.value)
    assert "unsloth" in message.lower()
    assert "Blackwell" in message  # points the user at the corpus_studio backend on sm_120
