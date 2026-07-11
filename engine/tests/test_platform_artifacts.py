"""Platform slice 5 — ArtifactManifest persistence + two-tier integrity. Pure tests (no torch): a
real temp 'adapter' directory exercises the cheap fingerprint, the byte-exact content hash, and the
live re-check (ok / modified / missing). Also covers execute_run surfacing + writing the manifests."""

import re

import corpus_studio.platform as P
from corpus_studio.platform.artifacts import (
    build_artifact_manifest,
    recheck_artifact_integrity,
    write_artifact_manifest,
)
from corpus_studio.platform.runners import demo_training_plan
from corpus_studio.platform.supervisor import ProducedArtifact, RunContext, execute_run

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLOCK = lambda: "2026-07-11T00:00:00+00:00"  # noqa: E731


def _make_adapter(directory):
    """A minimal on-disk adapter: a descriptor + a weight file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"weights-v1")
    return directory


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
    assert manifest.integrity.cheap_fingerprint  # size:mtime present
    assert P.ArtifactManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_build_manifest_for_a_missing_path_is_missing(tmp_path):
    manifest = build_artifact_manifest(
        artifact_id="run-2-adapter", path=str(tmp_path / "nope"), run_id="run-2", now="t",
    )
    assert manifest.integrity is not None
    assert manifest.integrity.current_integrity == "missing"
    assert manifest.integrity.content_hash is None


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

    class _AdapterRunner:
        name = "training"

        def run(self, ctx: RunContext):
            art = ProducedArtifact(
                artifact_id=f"{ctx.run_id}-adapter", kind="adapter", path=str(adapter)
            )
            ctx.emit_artifact(art)
            return [art]

    out = tmp_path / "out"
    result = execute_run(
        demo_training_plan(), _AdapterRunner(), run_id="run-9", out_dir=out, clock=_CLOCK
    )

    assert result.manifest.state == "succeeded"
    assert len(result.artifacts) == 1
    art_manifest = result.artifacts[0]
    assert art_manifest.artifact_id == "run-9-adapter"
    assert art_manifest.integrity is not None
    assert art_manifest.integrity.current_integrity == "ok"
    assert art_manifest.base_model == demo_training_plan().base_model
    # It was persisted next to the RunManifest.
    on_disk = out / "artifacts" / "run-9-adapter.json"
    assert on_disk.is_file()
    assert P.ArtifactManifest.model_validate_json(on_disk.read_text(encoding="utf-8")) == art_manifest


def test_echo_run_produces_no_artifacts():
    from corpus_studio.platform.supervisor import EchoRunner, demo_run_plan

    result = execute_run(demo_run_plan(), EchoRunner(), clock=_CLOCK)
    assert result.artifacts == []
