from corpus_studio.training.config_templates import (
    build_lora_config_template,
    normalize_training_config_target,
    render_training_config,
    training_config_file_extension,
)
from corpus_studio.training.estimators import describe_vram_estimate, estimate_token_budget


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


def test_training_estimators_are_lightweight_placeholders():
    token_estimate = estimate_token_budget(["abcd", "abcdefgh"])
    vram_estimate = describe_vram_estimate("Qwen/Qwen2.5-Coder-7B-Instruct")

    assert token_estimate.example_count == 2
    assert token_estimate.estimated_tokens == 3
    assert "requires model size" in vram_estimate.note
