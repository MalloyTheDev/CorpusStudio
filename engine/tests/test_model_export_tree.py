"""The full-parameter model export tree policy (pretraining / full-parameter SFT).

The subprocess parent admits a model-kind artifact only when its directory holds exactly one root
``model.safetensors`` weights payload and no link: the weight content hash follows linked files, so
without this policy an admitted hash could bind bytes that live outside the run-scoped artifact.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from corpus_studio.platform.artifacts import (
    _MAX_AUXILIARY_METADATA_BYTES,
    _validate_model_tree,
)


def _model_export(root: Path) -> Path:
    """The shape save_pretrained + tokenizer.save_pretrained write for a single-file export."""

    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"safetensors-bytes")
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.model").write_bytes(b"sentencepiece")
    return root


def test_genuine_model_export_and_benign_trainer_metadata_are_accepted(tmp_path):
    root = _model_export(tmp_path / "model")
    (root / "training_args.bin").write_bytes(b"\x80\x05\x95NOT-A-VALID-PICKLE")  # never parsed
    _validate_model_tree(root)


@pytest.mark.parametrize(
    "weight_name",
    [
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
        "model.bin",
        "model-00001-of-00002.safetensors",
        "tf_model.h5",
        "weights.h5",
        "flax_model.msgpack",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "weights.gguf",
        "model.onnx",
        "state.ckpt",
        "arbitrary.bin",
    ],
)
def test_model_tree_rejects_alternate_weight_payloads(tmp_path, weight_name):
    root = _model_export(tmp_path / "model")
    (root / weight_name).write_bytes(b"x")
    with pytest.raises(ValueError, match="^model artifact contains an alternate or nested"):
        _validate_model_tree(root)


def test_model_tree_rejects_nested_weights_and_nested_training_args(tmp_path):
    nested = _model_export(tmp_path / "nested")
    (nested / "sub").mkdir()
    (nested / "sub" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="alternate or nested"):
        _validate_model_tree(nested)
    nested_args = _model_export(tmp_path / "nested-args")
    (nested_args / "sub").mkdir()
    (nested_args / "sub" / "training_args.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="alternate or nested"):
        _validate_model_tree(nested_args)


def test_model_tree_rejects_checkpoints_and_links(tmp_path):
    checkpoint = _model_export(tmp_path / "checkpoint")
    (checkpoint / "checkpoint-10").mkdir()
    with pytest.raises(ValueError, match="^model artifact contains an intermediate checkpoint"):
        _validate_model_tree(checkpoint)

    outside = tmp_path / "outside-secret.bin"
    outside.write_bytes(b"bytes living outside the run scope")
    linked_file = _model_export(tmp_path / "linked-file")
    os.symlink(outside, linked_file / "notes.txt")
    with pytest.raises(ValueError, match="^model artifact contains a linked or irregular file"):
        _validate_model_tree(linked_file)

    linked_weights = _model_export(tmp_path / "linked-weights")
    (linked_weights / "model.safetensors").unlink()
    os.symlink(outside, linked_weights / "model.safetensors")
    with pytest.raises(ValueError, match="linked or irregular file"):
        _validate_model_tree(linked_weights)

    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    linked_dir = _model_export(tmp_path / "linked-dir")
    os.symlink(outside_dir, linked_dir / "extra", target_is_directory=True)
    with pytest.raises(ValueError, match="^model artifact contains a linked or irregular directory"):
        _validate_model_tree(linked_dir)


def test_model_tree_rejects_unusable_roots(tmp_path):
    with pytest.raises(ValueError, match="^model artifact directory is unavailable"):
        _validate_model_tree(tmp_path / "missing")
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="^model artifact must be a regular, non-link directory"):
        _validate_model_tree(file_root)
    real_root = _model_export(tmp_path / "real-root")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="non-link directory"):
        _validate_model_tree(linked_root)


@pytest.mark.parametrize("mode", ["hard_link", "oversized"])
def test_model_tree_applies_the_auxiliary_metadata_policy(tmp_path, mode):
    root = _model_export(tmp_path / "model")
    if mode == "hard_link":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"x")
        os.link(outside, root / "training_args.bin")
        expected = "^model artifact auxiliary metadata is hard-linked"
    else:
        (root / "training_args.bin").write_bytes(b"x" * (_MAX_AUXILIARY_METADATA_BYTES + 1))
        expected = "^model artifact auxiliary metadata exceeds the permitted size"
    with pytest.raises(ValueError, match=expected):
        _validate_model_tree(root)


def test_model_tree_normalizes_unexpected_resolution_failures(tmp_path, monkeypatch):
    root = _model_export(tmp_path / "model")
    unsafe = root / "unsafe"
    unsafe.mkdir()
    real_resolve = Path.resolve

    def fail_for_child(path, *, strict=False):
        if path == unsafe:
            raise RuntimeError("synthetic resolution failure")
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_for_child)
    with pytest.raises(ValueError, match="^model artifact tree is unsafe or changed"):
        _validate_model_tree(root)
