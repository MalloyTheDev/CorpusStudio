import pytest

from corpus_studio.training.config_templates import (
    build_lora_config_template,
    normalize_training_config_target,
    render_training_config,
    training_config_file_extension,
)
from corpus_studio.training.estimators import build_vram_estimate, estimate_token_budget


def test_training_config_template_returns_expected_fields():
    template = build_lora_config_template(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset_path="exports/coding_tutor_v0.1/train.jsonl",
        eval_dataset_path="exports/coding_tutor_v0.1/validation.jsonl",
        dataset_format="chat",
    )
    payload = template.to_training_dict()

    assert payload["target"] == "axolotl_yaml"
    assert payload["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert payload["dataset_path"] == "exports/coding_tutor_v0.1/train.jsonl"
    assert payload["eval_dataset_path"] == "exports/coding_tutor_v0.1/validation.jsonl"
    assert payload["format"] == "chat"
    assert payload["adapter"] == "lora"
    assert payload["lora_r"] == 16


def test_config_emits_a_fixed_default_seed_for_reproducibility():
    template = build_lora_config_template(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset_path="train.jsonl",
        eval_dataset_path=None,
        dataset_format="instruction",
    )
    payload = template.to_training_dict()
    # A fixed default (not random) → runs are reproducible by default; the config hash
    # in the run provenance manifest then pins weight-init with it.
    assert payload["seed"] == 42
    assert "seed: 42" in render_training_config(template)


def test_custom_seed_threads_through_to_the_config():
    template = build_lora_config_template(
        base_model="m",
        dataset_path="train.jsonl",
        eval_dataset_path=None,
        dataset_format="instruction",
        seed=1234,
    )
    assert template.to_training_dict()["seed"] == 1234
    assert "seed: 1234" in render_training_config(template)


@pytest.mark.parametrize(
    "target", ["axolotl", "trl", "unsloth", "huggingface", "llama_factory", "corpus_studio"]
)
def test_seed_is_rendered_for_every_target(target: str):
    # The seed only makes runs reproducible if it actually reaches EVERY target's config,
    # not just the axolotl YAML — lock that in across the render formats.
    template = build_lora_config_template(
        base_model="m",
        dataset_path="train.jsonl",
        eval_dataset_path=None,
        dataset_format="instruction",
        target=normalize_training_config_target(target),
        seed=99,
    )
    rendered = render_training_config(template).lower()
    assert "seed" in rendered and "99" in rendered


def test_training_config_renderer_returns_inspectable_yaml():
    template = build_lora_config_template(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset_path="exports/coding_tutor_v0.1/train.jsonl",
        eval_dataset_path=None,
        dataset_format="instruction",
        target=normalize_training_config_target("axolotl"),
        sequence_len=2048,
        lora_r=8,
    )

    rendered = render_training_config(template)

    assert training_config_file_extension(template.target) == ".yaml"
    assert 'target: "axolotl_yaml"' in rendered
    assert 'base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"' in rendered
    assert "sequence_len: 2048" in rendered
    assert "lora_r: 8" in rendered


@pytest.mark.parametrize(
    "alias", ["corpus_studio", "corpus-studio", "corpusstudio", "corpus", "first_party", "first-party"]
)
def test_first_party_target_aliases_normalize(alias: str):
    assert normalize_training_config_target(alias) == "corpus_studio"


def test_first_party_target_renders_json_train_run_reads():
    template = build_lora_config_template(
        base_model="Qwen/Qwen2.5-7B",
        dataset_path="train.jsonl",
        eval_dataset_path=None,
        dataset_format="instruction",
        target=normalize_training_config_target("corpus_studio"),
        sequence_len=2048,
        lora_r=8,
    )
    # The first-party config is JSON — the exact shape train-run's load_run_config_from_file reads.
    assert training_config_file_extension(template.target) == ".json"
    import json as _json

    payload = _json.loads(render_training_config(template))
    assert payload["target"] == "corpus_studio"
    assert payload["base_model"] == "Qwen/Qwen2.5-7B"
    assert payload["dataset_path"] == "train.jsonl"
    assert payload["format"] == "instruction"
    assert payload["sequence_len"] == 2048
    assert payload["lora_r"] == 8


def test_training_estimators_are_lightweight():
    token_estimate = estimate_token_budget(["abcd", "abcdefgh"])
    vram_estimate = build_vram_estimate("Qwen/Qwen2.5-Coder-7B-Instruct")

    assert token_estimate.example_count == 2
    assert token_estimate.estimated_tokens == 3
    assert vram_estimate.parameter_count_billions == 7.0
    assert "planning estimate" in vram_estimate.note
