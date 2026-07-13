"""Model/tokenizer descriptor contracts and the dependency-light static inspector.

These tests intentionally use tiny fake snapshots. They prove metadata, inventory, trust, and
compatibility behavior without downloading a model or importing an ML framework.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
import corpus_studio.platform as P
from corpus_studio.platform.contracts import (
    BackendCompatibilityEntry,
    CompatibilityCheck,
    DescriptorFile,
    DescriptorSource,
    DescriptorVerification,
    DimensionEvidence,
    EmbeddingVocabulary,
    ModelDescriptor,
    ModelTokenizerCompatibility,
    TokenizerDescriptor,
    TrustRequirement,
)
from corpus_studio.platform.enums import (
    CompatibilityStatus,
    EvidenceKind,
    ModelSourceKind,
    ModelTaskClass,
    VerificationOutcome,
)
import corpus_studio.platform.model_inspector as inspector
from corpus_studio.platform.model_inspector import (
    ModelInspectionError,
    check_model_tokenizer_compatibility,
    inspect_model,
    inspect_model_bundle,
    inspect_tokenizer,
    write_inspection_bundle,
)


NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
runner = CliRunner()


def _fixed_now() -> datetime:
    return NOW


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_model(root: Path, **config_overrides: object) -> Path:
    root.mkdir(exist_ok=True)
    config: dict[str, object] = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM", "LlamaForCausalLM"],
        "vocab_size": 7,
        "tie_word_embeddings": True,
        "max_position_embeddings": 4096,
        "sliding_window": 512,
        "rope_theta": 10_000.0,
        "torch_dtype": "bfloat16",
        "license": "apache-2.0",
        "num_parameters": 1234,
        "quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True},
    }
    config.update(config_overrides)
    _write_json(root / "config.json", config)
    (root / "model.safetensors").write_bytes(b"tiny-safe-weights")
    return root


def _make_tokenizer(root: Path, *, model_max_length: int = 4096) -> Path:
    root.mkdir(exist_ok=True)
    _write_json(
        root / "tokenizer.json",
        {
            "model": {
                "type": "BPE",
                "vocab": {
                    "<pad>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "hello": 3,
                    "world": 4,
                    "unused": 5,
                },
            },
            "added_tokens": [{"id": 6, "content": "<tool>", "special": True}],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Whitespace"},
        },
    )
    _write_json(
        root / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": model_max_length,
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": {"content": "<eos>"},
            "additional_special_tokens": ["<tool>"],
            "chat_template": "{{ messages | length }}",
        },
    )
    _write_json(
        root / "special_tokens_map.json",
        {"pad_token": "<pad>", "bos_token": "<bos>", "eos_token": "<eos>"},
    )
    return root


def _bundle(tmp_path: Path, *, hash_weights: bool = False):
    model_path = _make_model(tmp_path / "model")
    tokenizer_path = _make_tokenizer(tmp_path / "tokenizer")
    return inspect_model_bundle(
        model_path,
        model_id="tiny-model",
        tokenizer_path=tokenizer_path,
        tokenizer_id="tiny-tokenizer",
        repository="example/tiny",
        requested_revision="main",
        resolved_commit="ABCDEF1",
        hash_weights=hash_weights,
        now=_fixed_now,
    )


def test_inspection_bundle_captures_static_evidence_and_compatibility(tmp_path: Path):
    bundle = _bundle(tmp_path)
    model = bundle.model
    tokenizer = bundle.tokenizer
    compatibility = bundle.compatibility

    assert tokenizer is not None and compatibility is not None
    assert model.source.repository == "example/tiny"
    assert model.source.requested_revision == "main"
    assert model.source.resolved_commit == "abcdef1"
    assert model.source.revision_pinned is True
    assert model.source.snapshot_sha256 is None  # weights were deliberately not hashed
    assert model.architectures == ["LlamaForCausalLM"]
    assert [item.value for item in model.task_classes] == ["causal_lm"]
    assert [item.value for item in model.formats] == ["safetensors"]
    assert model.parameters.kind.value == "unknown"
    assert model.parameters.components[0].storage_dtype == "bfloat16"
    assert model.parameters.components[0].quantization.value == "int4"
    assert model.parameters.counts[0].value == 1234
    assert model.topology.execution_kind.value == "unknown"
    assert model.topology.expert_groups == []
    assert model.attention_type.value == "sliding_window"
    assert model.positional_encoding.value == "rope"
    assert model.license is not None and model.license.name == "apache-2.0"
    assert model.verification.integrity == VerificationOutcome.partial
    assert model.captured_at == "2026-07-13T12:00:00Z"

    # A separate tokenizer directory does not silently inherit the model repository identity.
    assert tokenizer.source.kind.value == "local"
    assert tokenizer.source.repository is None
    assert tokenizer.format.value == "tokenizers_json"
    assert tokenizer.base_vocabulary_size == 6
    assert tokenizer.added_token_count == 1
    assert tokenizer.effective_vocabulary_size == 7
    assert tokenizer.max_token_id == 6
    assert tokenizer.normalization == {"type": "NFC"}
    assert tokenizer.pre_tokenization == {"type": "Whitespace"}
    assert tokenizer.chat_template_sha256 == hashlib.sha256(
        b"{{ messages | length }}"
    ).hexdigest()
    tokens = {item.content: item for item in tokenizer.special_tokens}
    assert tokens["<bos>"].added is False
    assert tokens["<tool>"].added is True
    assert compatibility.status == CompatibilityStatus.compatible
    assert compatibility.resize_input_embeddings is False
    assert compatibility.resize_output_head is False
    assert tokenizer.model_compatibility == [compatibility]


def test_weight_hashing_produces_a_content_bound_snapshot_identity(tmp_path: Path):
    bundle = _bundle(tmp_path, hash_weights=True)
    assert all(item.hash_status == "verified" for item in bundle.model.files)
    assert bundle.model.source.snapshot_sha256 is not None
    assert bundle.model.verification.integrity == VerificationOutcome.passed


def test_tokenizer_source_identity_is_explicit_or_inherited_only_for_same_snapshot(tmp_path: Path):
    shared = _make_model(tmp_path / "shared")
    _make_tokenizer(shared)
    same = inspect_model_bundle(
        shared,
        model_id="m",
        tokenizer_path=shared,
        tokenizer_id="t",
        repository="owner/shared",
        requested_revision="main",
        resolved_commit="abcdef1",
        now=_fixed_now,
    )
    assert same.tokenizer is not None
    assert same.tokenizer.source.repository == "owner/shared"
    assert same.tokenizer.source.resolved_commit == "abcdef1"

    model = _make_model(tmp_path / "model-explicit")
    tokenizer = _make_tokenizer(tmp_path / "tokenizer-explicit")
    separate = inspect_model_bundle(
        model,
        model_id="m2",
        tokenizer_path=tokenizer,
        tokenizer_id="t2",
        repository="owner/model",
        resolved_commit="aaaaaaa",
        tokenizer_repository="owner/tokenizer",
        tokenizer_requested_revision="v1",
        tokenizer_resolved_commit="bbbbbbb",
        now=_fixed_now,
    )
    assert separate.tokenizer is not None
    assert separate.tokenizer.source.repository == "owner/tokenizer"
    assert separate.tokenizer.source.requested_revision == "v1"
    assert separate.tokenizer.source.resolved_commit == "bbbbbbb"

    with pytest.raises(ModelInspectionError, match="source options require"):
        inspect_model_bundle(
            model,
            model_id="m3",
            tokenizer_repository="owner/orphan",
            now=_fixed_now,
        )


def test_pickle_and_custom_code_are_detected_but_never_authorized(tmp_path: Path):
    root = tmp_path / "custom"
    root.mkdir()
    _write_json(
        root / "config.json",
        {
            "model_type": "custom",
            "auto_map": {"AutoModel": "modeling_custom.CustomModel"},
        },
    )
    (root / "pytorch_model.bin").write_bytes(b"not-really-pickle")
    (root / "modeling_custom.py").write_text(
        "raise RuntimeError('static inspection must never execute this')\n", encoding="utf-8"
    )

    model = inspect_model(root, model_id="custom", now=_fixed_now)
    assert model.formats[0].value == "pytorch_pickle"
    assert any("Pickle-based" in warning for warning in model.notes)
    assert model.trust.trust_remote_code is False
    assert model.trust.custom_code_required is True
    assert model.trust.approval_required is True
    assert model.trust.isolated_execution_required is True
    assert model.trust.custom_code_files == ["modeling_custom.py"]
    assert model.verification.custom_code_policy == VerificationOutcome.partial


def test_repository_without_commit_is_explicitly_unpinned(tmp_path: Path):
    model = inspect_model(
        _make_model(tmp_path / "model"),
        model_id="m",
        repository="example/m",
        requested_revision="moving-branch",
        now=_fixed_now,
    )
    assert model.source.revision_pinned is False
    assert any("not pinned" in warning for warning in model.notes)


def test_local_source_and_invalid_resolved_commit(tmp_path: Path):
    root = _make_model(tmp_path / "model")
    assert inspect_model(root, model_id="m", now=_fixed_now).source.kind.value == "local"
    with pytest.raises(ModelInspectionError, match="7-64 character hexadecimal"):
        inspect_model(root, model_id="m", resolved_commit="not-a-commit", now=_fixed_now)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.safetensors", "safetensors"),
        ("b.bin", "pytorch_pickle"),
        ("c.gguf", "gguf"),
        ("d.onnx", "onnx"),
        ("e.torchscript", "torchscript"),
        ("f.npz", "numpy"),
    ],
)
def test_supported_weight_formats_are_component_scoped(
    tmp_path: Path, filename: str, expected: str
):
    root = tmp_path / expected
    root.mkdir()
    (root / filename).write_bytes(b"weight")
    model = inspect_model(root, model_id=f"model-{expected}", now=_fixed_now)
    assert [item.value for item in model.formats] == [expected]
    assert model.parameters.components[0].format.value == expected


def test_static_classifier_variants_do_not_claim_more_than_metadata(tmp_path: Path):
    assert inspector._task_classes(
        {
            "architectures": [
                "BertForMaskedLM",
                "T5ForConditionalGeneration",
                "BertForSequenceClassification",
                "RewardModel",
                "VisionAudioMultimodalModel",
            ]
        }
    ) == sorted(
        {
            ModelTaskClass.masked_lm,
            ModelTaskClass.seq2seq_lm,
            ModelTaskClass.classification,
            ModelTaskClass.reward_model,
            ModelTaskClass.vision,
            ModelTaskClass.speech,
            ModelTaskClass.multimodal,
        },
        key=lambda item: item.value,
    )
    assert inspector._task_classes({}) == [ModelTaskClass.unknown]
    assert inspector._attention_type({"attention_type": "full"}).value == "full"
    assert inspector._attention_type({"model_type": "mamba"}).value == "state_space"
    assert inspector._attention_type({}).value == "unknown"
    assert inspector._positional_encoding({"alibi": True}).value == "alibi"
    assert inspector._positional_encoding({"position_embedding_type": "absolute"}).value == "absolute"
    assert inspector._positional_encoding({}).value == "unknown"
    quantization, _ = inspector._quantization(
        {"quantization_config": {"quant_method": "nf4"}}
    )
    assert quantization is not None and quantization.value == "nf4"
    assert inspector._quantization({"quantization_config": {"quant_method": "future"}})[0] is None
    assert inspector._quantization({}) == (None, {})


def test_tokenizer_list_vocab_config_fallback_and_template_list(tmp_path: Path):
    root = tmp_path / "tokenizer"
    root.mkdir()
    _write_json(
        root / "tokenizer.json",
        {
            "model": {"vocab": ["a", "b"]},
            "added_tokens": [
                {"content": "<x>", "id": 3},
                {"content": "<x>", "id": 3},
                {"content": 1, "id": 4},
            ],
        },
    )
    _write_json(
        root / "tokenizer_config.json",
        {
            "additional_special_tokens": [{"content": "<x>"}],
            "chat_template": [{"name": "default", "template": "{{ x }}"}],
        },
    )
    tokenizer = inspect_tokenizer(root, tokenizer_id="tok", now=_fixed_now)
    assert tokenizer.base_vocabulary_size == 2
    assert tokenizer.added_token_count == 1
    assert tokenizer.effective_vocabulary_size == 4
    assert tokenizer.special_tokens[0].added is True
    assert isinstance(tokenizer.chat_template, list)

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _write_json(fallback / "tokenizer_config.json", {"vocab_size": 99, "tokenizer_class": "X"})
    descriptor = inspect_tokenizer(fallback, tokenizer_id="fallback", now=_fixed_now)
    assert descriptor.base_vocabulary_size == 99
    assert descriptor.effective_vocabulary_size == 99


@pytest.mark.parametrize(
    ("filename", "config", "expected"),
    [
        ("tokenizer.model", None, "sentencepiece"),
        ("vocab.tiktoken", None, "tiktoken"),
        (None, {"tokenizer_class": "Remote", "auto_map": {"AutoTokenizer": "x.Y"}}, "custom"),
        (None, {"tokenizer_class": "Known"}, "unknown"),
    ],
)
def test_tokenizer_format_detection(
    tmp_path: Path, filename: str | None, config: dict[str, object] | None, expected: str
):
    root = tmp_path / expected
    root.mkdir()
    if filename:
        (root / filename).write_bytes(b"tokenizer")
    if config is not None:
        _write_json(root / "tokenizer_config.json", config)
    descriptor = inspect_tokenizer(root, tokenizer_id=f"tok-{expected}", now=_fixed_now)
    assert descriptor.format.value == expected


def test_huggingface_max_length_sentinel_remains_unknown(tmp_path: Path):
    tokenizer = inspect_tokenizer(
        _make_tokenizer(tmp_path / "tok", model_max_length=10**30),
        tokenizer_id="tok",
        now=_fixed_now,
    )
    assert tokenizer.model_max_length is None
    assert any("sentinel" in warning for warning in tokenizer.notes)


def test_compatibility_resize_incompatible_unverified_and_context_warning(tmp_path: Path):
    bundle = _bundle(tmp_path)
    assert bundle.tokenizer is not None
    model = bundle.model
    tokenizer = bundle.tokenizer

    smaller_vocab = EmbeddingVocabulary(
        declared_vocab_size=DimensionEvidence(
            value=5, source="test", evidence=EvidenceKind.measured
        ),
        input_embedding_rows=DimensionEvidence(
            value=5, source="test", evidence=EvidenceKind.measured
        ),
        tied_embeddings=True,
    )
    resize = check_model_tokenizer_compatibility(
        model.model_copy(update={"vocabulary": smaller_vocab}), tokenizer
    )
    assert resize.status == CompatibilityStatus.resize_required
    assert resize.resize_input_embeddings is True
    assert resize.resize_output_head is True
    assert resize.required_embedding_rows == 7

    wrong_link = model.model_copy(update={"tokenizer_ref": P.Ref(id="different")})
    assert (
        check_model_tokenizer_compatibility(wrong_link, tokenizer).status
        == CompatibilityStatus.incompatible
    )

    unknown = TokenizerDescriptor(
        tokenizer_id="tiny-tokenizer",
        source=DescriptorSource(kind=ModelSourceKind.local, local_path="C:/tmp/tok"),
    )
    assert check_model_tokenizer_compatibility(model, unknown).status == CompatibilityStatus.unverified

    short_context = tokenizer.model_copy(
        update={
            "model_max_length": DimensionEvidence(
                value=2048, source="test", evidence=EvidenceKind.declared
            )
        }
    )
    context_result = check_model_tokenizer_compatibility(model, short_context)
    assert context_result.status == CompatibilityStatus.compatible
    assert any("smaller runtime limit" in warning for warning in context_result.warnings)


def test_custom_code_keeps_static_compatibility_unverified(tmp_path: Path):
    bundle = _bundle(tmp_path)
    assert bundle.tokenizer is not None
    trust = TrustRequirement(
        custom_code_required=True,
        approval_required=True,
        isolated_execution_required=True,
        custom_code_files=["tokenization_custom.py"],
    )
    tokenizer = bundle.tokenizer.model_copy(update={"trust": trust})
    result = check_model_tokenizer_compatibility(bundle.model, tokenizer)
    assert result.status == CompatibilityStatus.unverified
    custom_check = next(item for item in result.checks if item.check == "custom-code-policy")
    assert custom_check.outcome == VerificationOutcome.not_checked


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "nested:stream",
        r"nested\file",
        "nested/./file",
        "nested//file",
    ],
)
def test_descriptor_paths_are_portable_and_cannot_escape(bad_path: str):
    with pytest.raises(ValidationError):
        DescriptorFile(path=bad_path, size_bytes=0)
    assert DescriptorFile(path=".metadata/file.json", size_bytes=0).path == ".metadata/file.json"


def test_descriptor_hash_and_link_evidence_is_consistent():
    with pytest.raises(ValidationError):
        DescriptorFile(path="x", size_bytes=1, hash_status="verified")
    with pytest.raises(ValidationError):
        DescriptorFile(path="x", size_bytes=1, sha256="a" * 64)
    with pytest.raises(ValidationError):
        DescriptorFile(path="x", size_bytes=1, is_link=True)
    linked = DescriptorFile(path="x", size_bytes=1, is_link=True, hash_status="skipped_unsafe")
    assert linked.is_link is True


def test_source_and_custom_code_contracts_fail_closed():
    with pytest.raises(ValidationError):
        DescriptorSource(kind="huggingface")
    with pytest.raises(ValidationError):
        DescriptorSource(kind="local")
    with pytest.raises(ValidationError):
        DescriptorSource(kind="artifact")
    with pytest.raises(ValidationError):
        DescriptorSource(kind="local", local_path="x", revision_pinned=True)
    assert DescriptorSource(kind="artifact", artifact_ref=P.Ref(id="artifact-1")).artifact_ref

    with pytest.raises(ValidationError):
        TrustRequirement(trust_remote_code=True)
    with pytest.raises(ValidationError):
        TrustRequirement(custom_code_required=True)
    with pytest.raises(ValidationError):
        TrustRequirement(approval_required=True)
    trust = TrustRequirement(
        custom_code_required=True,
        approval_required=True,
        isolated_execution_required=True,
        custom_code_files=["modeling.py"],
    )
    assert trust.trust_remote_code is False


def test_moe_contract_is_component_scoped_and_has_no_dense_only_scalar():
    model = ModelDescriptor(
        model_id="moe-model",
        source=DescriptorSource(kind="local", local_path="C:/models/moe"),
        formats=["safetensors"],
        parameters={
            "kind": "mixture_of_experts",
            "components": [
                {
                    "component_id": "experts",
                    "scope": "expert_group",
                    "format": "safetensors",
                    "storage_dtype": "bfloat16",
                },
                {
                    "component_id": "router",
                    "scope": "router",
                    "format": "safetensors",
                    "storage_dtype": "float32",
                },
            ],
            "counts": [
                {
                    "kind": "active_token",
                    "value": 2_000,
                    "scope": "experts_selected_per_token",
                    "measurement_window": "one_token",
                    "source": "model_card",
                    "evidence": "declared",
                },
                {
                    "kind": "logical",
                    "value": 20_000,
                    "scope": "all_experts_and_shared",
                    "measurement_window": "static_model",
                    "source": "model_card",
                    "evidence": "declared",
                },
            ],
        },
        topology={
            "execution_kind": "mixture_of_experts",
            "semantic_routing": {
                "router_type": "top-k",
                "selection_policy": "learned_logits",
                "top_k": 2,
                "metadata_source": "config",
            },
            "expert_groups": [
                {
                    "group_id": "decoder",
                    "layer_indices": [1, 2],
                    "expert_count": 8,
                    "experts_per_token": 2,
                }
            ],
        },
    )
    payload = model.model_dump(mode="json")
    assert "parameter_count" not in payload
    assert len(payload["parameters"]["counts"]) == 2
    assert payload["topology"]["physical_scheduler_owner"] == "run_plan"
    assert ModelDescriptor.model_validate_json(model.model_dump_json()) == model


def test_contract_ordering_tying_and_tokenizer_size_invariants():
    source = {"kind": "local", "local_path": "C:/models/x"}
    with pytest.raises(ValidationError, match="architectures"):
        ModelDescriptor(model_id="m", source=source, architectures=["Z", "A"])
    with pytest.raises(ValidationError, match="tied embeddings"):
        EmbeddingVocabulary(
            tied_embeddings=True,
            input_embedding_rows={"value": 2, "source": "x", "evidence": "declared"},
            output_head_rows={"value": 3, "source": "x", "evidence": "declared"},
        )
    with pytest.raises(ValidationError, match="below base"):
        TokenizerDescriptor(
            tokenizer_id="t", source=source, base_vocabulary_size=2, effective_vocabulary_size=1
        )
    with pytest.raises(ValidationError, match="below effective"):
        TokenizerDescriptor(
            tokenizer_id="t", source=source, effective_vocabulary_size=2, max_token_id=2
        )
    with pytest.raises(ValidationError, match="set together"):
        TokenizerDescriptor(tokenizer_id="t", source=source, chat_template="x")


def test_manual_support_and_compatibility_claims_require_evidence():
    with pytest.raises(ValidationError, match="capability evidence"):
        BackendCompatibilityEntry(backend_ref=P.Ref(id="backend"), status="compatible")
    supported = BackendCompatibilityEntry(
        backend_ref=P.Ref(id="backend"),
        environment_ref=P.Ref(id="environment"),
        capability_report_ref=P.Ref(id="report"),
        status="compatible",
    )
    assert supported.status == "compatible"
    with pytest.raises(ValidationError, match="require reasons"):
        BackendCompatibilityEntry(backend_ref=P.Ref(id="backend"), status="incompatible")

    model_ref = P.Ref(id="model")
    tokenizer_ref = P.Ref(id="tokenizer")
    with pytest.raises(ValidationError, match="passed checks"):
        ModelTokenizerCompatibility(
            model_ref=model_ref,
            tokenizer_ref=tokenizer_ref,
            status=CompatibilityStatus.compatible,
        )
    with pytest.raises(ValidationError, match="explicit resize"):
        ModelTokenizerCompatibility(
            model_ref=model_ref,
            tokenizer_ref=tokenizer_ref,
            status=CompatibilityStatus.resize_required,
            checks=[
                CompatibilityCheck(
                    check="input-vocabulary", outcome=VerificationOutcome.failed
                )
            ],
        )
    with pytest.raises(ValidationError, match="not_checked"):
        ModelTokenizerCompatibility(
            model_ref=model_ref,
            tokenizer_ref=tokenizer_ref,
            status=CompatibilityStatus.unverified,
            checks=[
                CompatibilityCheck(
                    check="input-vocabulary", outcome=VerificationOutcome.passed
                )
            ],
        )


def test_manual_inventory_claims_are_internally_consistent():
    source = {"kind": "local", "local_path": "C:/models/x"}
    with pytest.raises(ValidationError, match="recorded file sizes"):
        ModelDescriptor.model_validate(
            {
                "model_id": "m",
                "source": source,
                "files": [{"path": "config.json", "size_bytes": 2}],
                "storage_size_bytes": 1,
            }
        )
    with pytest.raises(ValidationError, match="exist in the model inventory"):
        ModelDescriptor.model_validate(
            {
                "model_id": "m",
                "source": source,
                "formats": ["safetensors"],
                "parameters": {
                    "components": [
                        {
                            "component_id": "weights",
                            "format": "safetensors",
                            "file_refs": ["missing.safetensors"],
                        }
                    ]
                },
            }
        )
    with pytest.raises(ValidationError, match="fully hashed"):
        ModelDescriptor.model_validate(
            {
                "model_id": "m",
                "source": {**source, "snapshot_sha256": "a" * 64},
                "files": [{"path": "config.json", "size_bytes": 2}],
                "storage_size_bytes": 2,
            }
        )
    with pytest.raises(ValidationError, match="sorted and unique"):
        DescriptorVerification(warnings=["z", "a"])


def test_invalid_missing_and_oversized_metadata_fail_boundedly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ModelInspectionError, match="neither config"):
        inspect_model(empty, model_id="m")
    with pytest.raises(ModelInspectionError, match="no recognized tokenizer"):
        inspect_tokenizer(empty, tokenizer_id="t")

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "config.json").write_text("{", encoding="utf-8")
    with pytest.raises(ModelInspectionError, match="invalid JSON"):
        inspect_model(malformed, model_id="m")

    array_root = tmp_path / "array"
    array_root.mkdir()
    _write_json(array_root / "config.json", [])
    with pytest.raises(ModelInspectionError, match="root must be an object"):
        inspect_model(array_root, model_id="m")

    oversized = _make_model(tmp_path / "oversized")
    monkeypatch.setattr(inspector, "MAX_JSON_BYTES", 1)
    with pytest.raises(ModelInspectionError, match="exceeds the 1-byte limit"):
        inspect_model(oversized, model_id="m")


def test_inventory_limit_and_hash_failure_are_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _make_model(tmp_path / "model")
    monkeypatch.setattr(inspector, "MAX_INVENTORY_FILES", 1)
    with pytest.raises(ModelInspectionError, match="1 file inspection limit"):
        inspect_model(root, model_id="m")

    monkeypatch.setattr(inspector, "MAX_INVENTORY_FILES", 100_000)
    original = inspector._sha256_file_stable

    def fail_config(path: Path) -> str:
        if path.name == "config.json":
            raise ModelInspectionError("source changed while hashing: config.json")
        return original(path)

    monkeypatch.setattr(inspector, "_sha256_file_stable", fail_config)
    descriptor = inspect_model(root, model_id="m")
    config_file = next(item for item in descriptor.files if item.path == "config.json")
    assert config_file.hash_status == "unreadable"
    assert descriptor.inventory_complete is False
    assert descriptor.verification.integrity == VerificationOutcome.failed


def test_parsed_metadata_must_match_the_verified_inventory_digest(tmp_path: Path):
    root = _make_model(tmp_path / "model")
    with pytest.raises(ModelInspectionError, match="changed after inventory"):
        inspector._load_json(root, "config.json", expected_sha256="0" * 64)


def test_linked_files_and_directories_are_not_followed(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    root = _make_model(tmp_path / "model")
    try:
        os.symlink(outside / "secret.txt", root / "linked-file.txt")
        os.symlink(outside, root / "linked-dir", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    model = inspect_model(root, model_id="m", now=_fixed_now)
    linked = next(item for item in model.files if item.path == "linked-file.txt")
    assert linked.is_link is True
    assert linked.hash_status == "skipped_unsafe"
    assert all(not item.path.startswith("linked-dir/") for item in model.files)
    assert model.inventory_complete is False

    with pytest.raises(ModelInspectionError, match="root cannot be"):
        inspect_model(root / "linked-dir", model_id="m")


def test_safe_root_rejects_missing_and_regular_files(tmp_path: Path):
    with pytest.raises(ModelInspectionError, match="does not exist"):
        inspect_model(tmp_path / "missing", model_id="m")
    regular = tmp_path / "file"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(ModelInspectionError, match="not a directory"):
        inspect_model(regular, model_id="m")


def test_cli_json_output_is_atomic_and_human_output_is_ascii(tmp_path: Path):
    model = _make_model(tmp_path / "model")
    tokenizer = _make_tokenizer(tmp_path / "tokenizer")
    output = tmp_path / "descriptors"
    result = runner.invoke(
        app,
        [
            "model-inspect",
            str(model),
            "--tokenizer",
            str(tokenizer),
            "--out",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["model"]["model_id"] == "model"
    assert payload["tokenizer"]["tokenizer_id"] == "model-tokenizer"
    assert payload["compatibility"]["status"] == "compatible"
    assert len(payload["written_files"]) == 3
    assert not list(output.glob("*.tmp"))
    assert all(Path(path).is_file() for path in payload["written_files"])

    human = runner.invoke(app, ["model-inspect", str(model), "--hash-weights"])
    assert human.exit_code == 0
    assert "Model: model" in human.stdout
    assert "trust_remote_code=false" in human.stdout
    human.stdout.encode("ascii")


def test_cli_failure_is_structured_and_output_target_is_validated(tmp_path: Path):
    model = _make_model(tmp_path / "model")
    result = runner.invoke(
        app,
        ["model-inspect", str(model), "--resolved-commit", "bad", "--json"],
    )
    assert result.exit_code == 2
    assert "MODEL_INSPECTION_FAILED" in result.output

    bundle = inspect_model_bundle(model, model_id="m", now=_fixed_now)
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("x", encoding="utf-8")
    with pytest.raises(ModelInspectionError, match="regular directory"):
        write_inspection_bundle(bundle, output_file)


def test_static_inspector_imports_without_ml_frameworks():
    code = """
import json, sys
import corpus_studio.platform.model_inspector
print(json.dumps({name: name in sys.modules for name in ['torch','transformers','tokenizers','trl']}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "torch": False,
        "transformers": False,
        "tokenizers": False,
        "trl": False,
    }
