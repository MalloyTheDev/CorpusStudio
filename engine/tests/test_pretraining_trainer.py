"""The from-scratch pretraining worker's torch-free surfaces (S3b-1a): reading the corpus, loading +
sanitizing the architecture config, and the fail-closed refusals for modes this first slice does not
implement. The torch training loop + BPE tokenizer trainer are proven by a separate CPU run."""

import json
from types import SimpleNamespace

import pytest

from corpus_studio.platform.common import Ref
from corpus_studio.training.pretraining_trainer import (
    PretrainingError,
    PretrainResult,
    _refuse_unsupported,
    load_architecture_config,
    load_corpus_documents,
)


def test_load_corpus_documents_reads_the_text_field(tmp_path):
    shard = tmp_path / "s0.jsonl"
    shard.write_text('{"text": "a"}\n\n{"text": "b"}\n{"other": 1}\n', encoding="utf-8")
    assert load_corpus_documents(["s0.jsonl"], corpus_root=tmp_path) == ["a", "b"]


def test_load_corpus_documents_refuses_a_non_json_row(tmp_path):
    (tmp_path / "s0.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(PretrainingError, match="non-JSON"):
        load_corpus_documents(["s0.jsonl"], corpus_root=tmp_path)


def test_load_architecture_config_sanitizes_provenance_keys(tmp_path):
    arch = tmp_path / "arch.json"
    arch.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 64,
                "architectures": ["LlamaForCausalLM"],
                "corpus_studio_name": "Mine",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_architecture_config(Ref(id=str(arch)))
    assert cfg == {"model_type": "llama", "hidden_size": 64}  # provenance keys stripped


def test_load_architecture_config_refuses_custom_decoder(tmp_path):
    arch = tmp_path / "arch.json"
    arch.write_text(
        json.dumps({"model_type": "custom_decoder", "corpus_studio_needs_custom_code": True}),
        encoding="utf-8",
    )
    with pytest.raises(PretrainingError, match="custom_decoder"):
        load_architecture_config(Ref(id=str(arch)))


def test_load_architecture_config_requires_a_ref():
    with pytest.raises(PretrainingError, match="requires an architecture_ref"):
        load_architecture_config(None)


def _execution(*, init_mode="random", custom_code=None, tokenizer_mode="train", tokenizer_location=None):
    # A duck-typed stand-in: _refuse_unsupported only reads these attributes (a full sealed config is heavy).
    return SimpleNamespace(
        init=SimpleNamespace(mode=init_mode, custom_code=custom_code),
        tokenizer_source=SimpleNamespace(mode=tokenizer_mode, tokenizer_location=tokenizer_location),
    )


def test_refuse_unsupported_modes():
    _refuse_unsupported(_execution())  # random + train is supported (no raise)
    _refuse_unsupported(_execution(tokenizer_mode="import", tokenizer_location="/t"))  # import (inc 3b)
    _refuse_unsupported(_execution(tokenizer_mode="freeze", tokenizer_location="/t"))  # freeze + location
    with pytest.raises(PretrainingError, match="continued"):
        _refuse_unsupported(_execution(init_mode="continued"))
    with pytest.raises(PretrainingError, match="custom-block"):
        _refuse_unsupported(_execution(custom_code=object()))
    with pytest.raises(PretrainingError, match="freeze tokenizer without a location"):
        _refuse_unsupported(_execution(tokenizer_mode="freeze", tokenizer_location=None))


def test_tokenizer_source_spec_import_requires_a_location():
    from corpus_studio.platform.contracts import TokenizerSourceSpec

    with pytest.raises(ValueError, match="no pre-existing"):  # a trained tokenizer has no location
        TokenizerSourceSpec(
            mode="train", algorithm="bpe", vocab_size=100, special_tokens=["<eos>"],
            tokenizer_location="/t",
        )
    with pytest.raises(ValueError, match="tokenizer_location"):  # import needs WHERE to load from
        TokenizerSourceSpec(mode="import", tokenizer_content_sha256="a" * 64)
    ok = TokenizerSourceSpec(mode="import", tokenizer_content_sha256="a" * 64, tokenizer_location="/tok")
    assert ok.tokenizer_location == "/tok"
    # freeze reuses the checkpoint's tokenizer - digest but no separate location
    assert TokenizerSourceSpec(mode="freeze", tokenizer_content_sha256="a" * 64).tokenizer_location is None


def test_pretrain_result_bounds_coverage():
    ok = PretrainResult(
        output_dir="/o", cpu_toy=True, vocab_size=300, num_blocks=2, coverage_ratio=1.0,
        tokenizer_source="trained",
    )
    assert ok.steps == 0
    with pytest.raises(ValueError):
        PretrainResult(
            output_dir="/o", cpu_toy=True, vocab_size=300, num_blocks=2, coverage_ratio=1.5,
            tokenizer_source="trained",
        )
