"""Platform slice 5 — ArtifactManifest persistence + two-tier integrity. Pure tests (no torch): a
real temp 'adapter' directory exercises the cheap fingerprint, the byte-exact content hash, and the
live re-check (ok / modified / missing). Also covers execute_run surfacing + writing the manifests."""

from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import struct

import pytest

import corpus_studio.platform as P
from corpus_studio.platform.artifacts import (
    _MAX_AUXILIARY_METADATA_BYTES,
    _stable_bounded_file_bytes,
    _validate_adapter_tree,
    build_artifact_manifest,
    canonical_adapter_config_sha256,
    recheck_artifact_integrity,
    validate_sealed_adapter_artifact,
    write_artifact_manifest,
)
from corpus_studio.platform.supervisor import (
    EchoRunner,
    ProducedArtifact,
    RunContext,
    demo_run_plan,
    execute_run,
)
from corpus_studio.training.artifact_registry import compute_weight_content_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


def _make_adapter(directory):
    """A minimal on-disk adapter: a descriptor + a weight file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"weights-v1")
    return directory


def _make_sealed_adapter(directory, *, tensor_names=None):
    from corpus_studio.platform.parameter_accounting import canonical_tensor_state_sha256
    from corpus_studio.platform.runners import demo_training_plan

    execution = demo_training_plan().resolved_execution
    assert execution is not None
    directory.mkdir(parents=True, exist_ok=True)
    names = tensor_names or [
        "base_model.model.layers.0.q_proj.lora_A.weight",
        "base_model.model.layers.0.q_proj.lora_B.weight",
    ]
    data = struct.pack(
        "<" + "f" * len(names),
        *(float(index + 1) for index in range(len(names))),
    )
    header = json.dumps(
        {
            name: {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [index * 4, (index + 1) * 4],
            }
            for index, name in enumerate(names)
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    (directory / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + data
    )
    config = {
        "peft_type": "LORA",
        "task_type": execution.adapter_task_type,
        "r": execution.adapter.lora_r,
        "lora_alpha": execution.adapter.lora_alpha,
        "lora_dropout": execution.adapter.lora_dropout,
        "bias": execution.adapter.bias,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": execution.inputs.model.location,
        "inference_mode": True,
        "peft_version": next(
            item.version
            for item in execution.trainer_interface.package_versions
            if item.name == "peft"
        ),
        "use_dora": False,
        "use_rslora": False,
    }
    (directory / "adapter_config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    records = [
        {
            "name": name,
            "dtype": "F32",
            "shape": [1],
            "content_sha256": hashlib.sha256(
                data[index * 4 : (index + 1) * 4]
            ).hexdigest(),
        }
        for index, name in enumerate(names)
    ]
    evidence = P.AdapterExportStateEvidence(
        before_sha256="a" * 64,
        after_sha256=canonical_tensor_state_sha256(records),
        tensor_count=len(names),
        tensor_names=names,
        changed_tensor_count=1,
        changed_tensor_names=[names[-1]],
        adapter_config_semantic_sha256=canonical_adapter_config_sha256(config),
    )
    return execution, evidence, config


def test_adapter_config_canonicalization_handles_enums_sets_and_order():
    class Value(str, Enum):
        one = "one"

    first = {"target_modules": {"b", "a"}, "value": Value.one, "ordered": (2, 1)}
    second = {"ordered": [2, 1], "value": "one", "target_modules": ["a", "b"]}
    assert canonical_adapter_config_sha256(first) == canonical_adapter_config_sha256(second)
    with pytest.raises(ValueError, match="non-JSON"):
        canonical_adapter_config_sha256({"bad": object()})


def test_bounded_metadata_reader_rejects_empty_and_linked_files(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="size"):
        _stable_bounded_file_bytes(empty, limit=10)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="non-link"):
        _stable_bounded_file_bytes(linked, limit=10)


def test_bounded_metadata_reader_detects_an_open_file_identity_change(tmp_path, monkeypatch):
    import corpus_studio.platform.artifacts as artifacts_module

    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    real_fstat = artifacts_module.os.fstat
    calls = 0

    def changed_fstat(file_descriptor):
        nonlocal calls
        calls += 1
        observed = real_fstat(file_descriptor)
        if calls == 2:
            return type(
                "ChangedStat",
                (),
                {
                    "st_dev": observed.st_dev,
                    "st_ino": observed.st_ino,
                    "st_size": observed.st_size,
                    "st_mtime_ns": observed.st_mtime_ns + 1,
                },
            )()
        return observed

    monkeypatch.setattr(artifacts_module.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="changed while it was read"):
        _stable_bounded_file_bytes(path, limit=10)


def test_adapter_tree_rejects_missing_roots_links_and_alternate_weights(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        _validate_adapter_tree(tmp_path / "missing")
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        _validate_adapter_tree(file_root)
    linked_root = tmp_path / "linked-root"
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="non-link"):
        _validate_adapter_tree(linked_root)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "weights").mkdir()
    (nested / "weights" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="alternate or nested"):
        _validate_adapter_tree(nested)

    linked_directory_root = tmp_path / "linked-directory-root"
    linked_directory_root.mkdir()
    (linked_directory_root / "linked-child").symlink_to(
        real_root, target_is_directory=True
    )
    with pytest.raises(ValueError, match="linked or irregular directory"):
        _validate_adapter_tree(linked_directory_root)

    linked_file_root = tmp_path / "linked-file-root"
    linked_file_root.mkdir()
    (linked_file_root / "linked-file").symlink_to(file_root)
    with pytest.raises(ValueError, match="linked or irregular file"):
        _validate_adapter_tree(linked_file_root)


def test_adapter_tree_normalizes_unexpected_resolution_failures(tmp_path, monkeypatch):
    root = tmp_path / "adapter"
    unsafe = root / "unsafe"
    unsafe.mkdir(parents=True)
    real_resolve = Path.resolve

    def fail_for_child(path, *, strict=False):
        if path == unsafe:
            raise RuntimeError("synthetic resolution failure")
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_for_child)
    with pytest.raises(ValueError, match="tree is unsafe"):
        _validate_adapter_tree(root)


# ---- training_args.bin auxiliary-metadata admission (precise, fail-closed) --------------------------
# Regression for the observed ARTIFACT_FAILURE: TRL Trainer.save_model writes a benign training_args.bin
# next to the adapter, and the blanket ".bin == weight" rule rejected it. The adapter itself was clean.


def _canonical_adapter(root: Path) -> Path:
    """A realistic PEFT/TRL adapter output tree (the shape our worker actually writes)."""
    root.mkdir(parents=True)
    (root / "adapter_model.safetensors").write_bytes(b"safetensors-bytes")
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    return root


def test_adapter_tree_accepts_training_args_bin_and_normal_metadata(tmp_path):
    root = _canonical_adapter(tmp_path / "adapter")
    # exactly what TRL Trainer.save_model emits alongside the adapter
    (root / "training_args.bin").write_bytes(b"\x80\x04}\x94.")  # arbitrary bytes; NEVER unpickled
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
    (root / "README.md").write_text("card", encoding="utf-8")
    (root / "MODEL_CARD.md").write_text("card", encoding="utf-8")
    _validate_adapter_tree(root)  # must not raise


def test_adapter_tree_never_deserializes_training_args_bin(tmp_path):
    # Bytes that would raise if unpickled; validation must still pass (it only stats, never parses).
    root = _canonical_adapter(tmp_path / "adapter")
    (root / "training_args.bin").write_bytes(b"\x80\x05\x95GARBAGE-NOT-A-VALID-PICKLE\xff\xff")
    _validate_adapter_tree(root)


@pytest.mark.parametrize(
    "weight_name",
    [
        "adapter_model.bin",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "model.bin",
        "model-00001-of-00002.bin",
        "arbitrary.bin",
        "optimizer.pt",
        "scheduler.pt",
        "extra.safetensors",
        "second_model.safetensors",
        "weights.gguf",
        "model.onnx",
        "state.ckpt",
    ],
)
def test_adapter_tree_still_rejects_weight_payloads(tmp_path, weight_name):
    root = _canonical_adapter(tmp_path / f"adapter-{weight_name.replace('.', '_')}")
    (root / weight_name).write_bytes(b"x")
    with pytest.raises(ValueError, match="alternate or nested"):
        _validate_adapter_tree(root)


def test_adapter_tree_rejects_nested_training_args_bin(tmp_path):
    # training_args.bin is permitted ONLY at the artifact root; a nested copy is a weight payload.
    root = _canonical_adapter(tmp_path / "adapter")
    (root / "sub").mkdir()
    (root / "sub" / "training_args.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="alternate or nested"):
        _validate_adapter_tree(root)


def test_adapter_tree_rejects_checkpoint_directory(tmp_path):
    root = _canonical_adapter(tmp_path / "adapter")
    (root / "checkpoint-100").mkdir()
    with pytest.raises(ValueError, match="intermediate checkpoint"):
        _validate_adapter_tree(root)


def test_adapter_tree_rejects_oversized_training_args_bin(tmp_path):
    root = _canonical_adapter(tmp_path / "adapter")
    (root / "training_args.bin").write_bytes(b"x" * (_MAX_AUXILIARY_METADATA_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds the permitted size"):
        _validate_adapter_tree(root)


def test_adapter_tree_rejects_hardlinked_training_args_bin(tmp_path):
    root = _canonical_adapter(tmp_path / "adapter")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    os.link(outside, root / "training_args.bin")  # a hard link can alias content outside the tree
    with pytest.raises(ValueError, match="hard-linked"):
        _validate_adapter_tree(root)


def test_adapter_tree_rejects_symlinked_training_args_bin(tmp_path):
    root = _canonical_adapter(tmp_path / "adapter")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    (root / "training_args.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="linked or irregular file"):
        _validate_adapter_tree(root)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"r": 99}, "field 'r' differs"),
        ({"lora_dropout": True}, "dropout is invalid"),
        ({"lora_dropout": 0.2}, "dropout differs"),
        ({"target_modules": []}, "target_modules are invalid"),
        ({"target_modules": ["q_proj", "k_proj"]}, "no saved LoRA"),
        ({"base_model_name_or_path": "wrong/model"}, "base-model linkage"),
        ({"inference_mode": False}, "inference-mode"),
        ({"peft_version": "wrong"}, "PEFT version"),
        ({"use_dora": True}, "semantics differ"),
    ],
)
def test_sealed_adapter_rejects_config_semantic_deviations(tmp_path, mutation, expected):
    adapter = tmp_path / "adapter"
    execution, evidence, config = _make_sealed_adapter(adapter)
    config.update(mutation)
    (adapter / "adapter_config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=expected):
        validate_sealed_adapter_artifact(adapter, execution, evidence)


@pytest.mark.parametrize("payload", [b"not json", b"[]", b'{"r":1,"r":2}'])
def test_sealed_adapter_rejects_malformed_config_documents(tmp_path, payload):
    adapter = tmp_path / "adapter"
    execution, evidence, _config = _make_sealed_adapter(adapter)
    (adapter / "adapter_config.json").write_bytes(payload)
    with pytest.raises(ValueError):
        validate_sealed_adapter_artifact(adapter, execution, evidence)


def test_sealed_adapter_rejects_missing_config_and_invalid_tensor_file(tmp_path):
    adapter = tmp_path / "adapter"
    execution, evidence, _config = _make_sealed_adapter(adapter)
    (adapter / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_sealed_adapter_artifact(adapter, execution, evidence)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Safetensors is invalid"):
        validate_sealed_adapter_artifact(adapter, execution, evidence)


def test_sealed_adapter_rejects_targets_that_differ_from_an_explicit_seal(tmp_path):
    adapter = tmp_path / "adapter"
    execution, evidence, _config = _make_sealed_adapter(adapter)
    explicit = execution.model_copy(
        update={
            "adapter": execution.adapter.model_copy(
                update={"target_modules": ["k_proj"]}
            )
        }
    )
    with pytest.raises(ValueError, match="target_modules differ from the explicit seal"):
        validate_sealed_adapter_artifact(adapter, explicit, evidence)


@pytest.mark.parametrize(
    ("tensor_names", "expected"),
    [
        (["base_model.model.layers.0.q_proj.weight"], "non-LoRA"),
        (["base_model.model.layers.0.q_proj.lora_A.weight"], "complete LoRA"),
    ],
)
def test_sealed_adapter_requires_only_complete_lora_tensor_pairs(
    tmp_path, tensor_names, expected
):
    adapter = tmp_path / "adapter"
    execution, evidence, _config = _make_sealed_adapter(
        adapter, tensor_names=tensor_names
    )
    with pytest.raises(ValueError, match=expected):
        validate_sealed_adapter_artifact(adapter, execution, evidence)


# ---- build ------------------------------------------------------------------


def test_build_manifest_for_a_real_adapter_computes_two_tier_integrity(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(
        artifact_id="run-1-adapter", path=str(adapter), run_id="run-1",
        base_model="tiny/model", now="t",
    )

    assert manifest.artifact_id == "run-1-adapter"
    assert manifest.producer_run_ref.id == "run-1"
    assert manifest.kind == "adapter"
    assert manifest.status == "candidate"
    assert manifest.base_model == "tiny/model"
    assert manifest.integrity is not None
    assert manifest.integrity.current_integrity == "ok"
    assert _SHA256.match(manifest.integrity.content_hash or "")
    assert _SHA256.match(manifest.integrity.metadata_hash or "")
    assert manifest.integrity.cheap_fingerprint  # size:mtime present
    assert P.ArtifactManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_build_manifest_for_a_missing_path_is_missing(tmp_path):
    manifest = build_artifact_manifest(
        artifact_id="run-2-adapter", path=str(tmp_path / "nope"), run_id="run-2", now="t",
    )
    assert manifest.integrity is not None
    assert manifest.integrity.current_integrity == "missing"
    assert manifest.integrity.content_hash is None


def test_required_weight_hash_never_falls_back_to_adapter_config(tmp_path):
    descriptor_only = tmp_path / "descriptor-only"
    descriptor_only.mkdir()
    (descriptor_only / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")

    assert compute_weight_content_hash(str(descriptor_only)) is None
    adapter = _make_adapter(tmp_path / "adapter-with-weights")
    assert _SHA256.match(compute_weight_content_hash(str(adapter)) or "")


def test_unknown_kind_is_recorded_as_other(tmp_path):
    adapter = _make_adapter(tmp_path / "a")
    manifest = build_artifact_manifest(
        artifact_id="x", path=str(adapter), run_id="r", kind="frobnicator", now="t",
    )
    assert manifest.kind == "other"


# ---- re-check ---------------------------------------------------------------


def test_recheck_is_ok_when_unchanged(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    rechecked = recheck_artifact_integrity(manifest, now="t2")
    assert rechecked.integrity is not None
    assert rechecked.integrity.current_integrity == "ok"
    assert rechecked.updated_at == "t2"


def test_recheck_detects_modified_weights(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    # Byte-swap the weights (same-ish size) — the content hash must catch it.
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")
    rechecked = recheck_artifact_integrity(manifest)
    assert rechecked.integrity is not None
    assert rechecked.integrity.current_integrity == "modified"


def test_recheck_detects_modified_adapter_config_independently_of_weights(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    (adapter / "adapter_config.json").write_text('{"r": 8}', encoding="utf-8")
    rechecked = recheck_artifact_integrity(manifest)
    assert rechecked.integrity is not None
    assert rechecked.integrity.current_integrity == "modified"


def test_recheck_detects_missing_weights(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    for entry in adapter.iterdir():
        entry.unlink()
    adapter.rmdir()
    rechecked = recheck_artifact_integrity(manifest)
    assert rechecked.integrity is not None
    assert rechecked.integrity.current_integrity == "missing"


def test_recheck_falls_back_to_cheap_fingerprint_when_no_content_hash(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    built = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    # An artifact registered with only the cheap fingerprint (content hash skipped).
    cheap_only = built.model_copy(
        update={"integrity": built.integrity.model_copy(update={"content_hash": None})}
    )
    assert recheck_artifact_integrity(cheap_only).integrity.current_integrity == "ok"
    # A resize/touch changes size+mtime → the cheap fingerprint catches it.
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-much-longer-now")
    assert recheck_artifact_integrity(cheap_only).integrity.current_integrity == "modified"


def test_recheck_is_unknown_when_no_hashes_were_stored(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    built = build_artifact_manifest(artifact_id="a", path=str(adapter), run_id="r", now="t")
    no_hashes = built.model_copy(
        update={
            "integrity": built.integrity.model_copy(
                update={"content_hash": None, "cheap_fingerprint": None}
            )
        }
    )
    # The path still exists (fingerprint computable) but nothing was stored to compare against.
    assert recheck_artifact_integrity(no_hashes).integrity.current_integrity == "unknown"


def test_recheck_without_stored_integrity_is_a_noop():
    manifest = P.ArtifactManifest(
        artifact_id="a", producer_run_ref=P.Ref(id="r"), created_at="t", updated_at="t",
        path="somewhere", integrity=None,
    )
    assert recheck_artifact_integrity(manifest) is manifest


# ---- persistence ------------------------------------------------------------


def test_write_artifact_manifest_atomically(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")
    manifest = build_artifact_manifest(artifact_id="run-1-adapter", path=str(adapter), run_id="r", now="t")
    written = write_artifact_manifest(manifest, tmp_path / "out")

    assert written == tmp_path / "out" / "artifacts" / "run-1-adapter.json"
    assert written.is_file()
    reloaded = P.ArtifactManifest.model_validate_json(written.read_text(encoding="utf-8"))
    assert reloaded == manifest
    assert list((tmp_path / "out" / "artifacts").glob(".*tmp")) == []


# ---- execute_run integration ------------------------------------------------


def test_execute_run_surfaces_and_writes_artifact_manifests(tmp_path):
    adapter = _make_adapter(tmp_path / "adapter")

    class _AdapterRunner(EchoRunner):
        name = "echo"

        def run(self, ctx: RunContext):
            art = ProducedArtifact(
                artifact_id=f"{ctx.run_id}-adapter", kind="adapter", path=str(adapter)
            )
            ctx.emit_artifact(art)
            return [art]

    out = tmp_path / "out"
    plan = demo_run_plan()
    result = execute_run(plan, _AdapterRunner(), run_id="run-9", out_dir=out, clock=_CLOCK)

    assert result.manifest.state == "succeeded"
    assert len(result.artifacts) == 1
    art_manifest = result.artifacts[0]
    assert art_manifest.artifact_id == "run-9-adapter"
    assert art_manifest.integrity is not None
    assert art_manifest.integrity.current_integrity == "ok"
    assert art_manifest.base_model == plan.base_model
    # It was persisted next to the RunManifest.
    on_disk = out / "runs" / "run-9" / "artifacts" / "run-9-adapter.json"
    assert on_disk.is_file()
    assert P.ArtifactManifest.model_validate_json(on_disk.read_text(encoding="utf-8")) == art_manifest


def test_echo_run_produces_no_artifacts():
    from corpus_studio.platform.supervisor import EchoRunner, demo_run_plan

    result = execute_run(demo_run_plan(), EchoRunner(), clock=_CLOCK)
    assert result.artifacts == []
