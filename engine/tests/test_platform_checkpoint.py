"""Deterministic tests for exact checkpoint + resume lineage (#440).

No GPU and no torch: the worker-populated state is synthesized, files are plain bytes. Covers the
sealed manifest contract + validators, canonical hashing, fail-closed integrity (incomplete / missing
/ externally-changed / tampered / malformed), resume compatibility (parent admission), and lineage.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from corpus_studio.platform import checkpoint as ck
from corpus_studio.platform.contracts import (
    CheckpointFileEntry,
    CheckpointManifest,
    CheckpointResumeRequest,
    HashRef,
    Ref,
    RunManifest,
    SealedTrainingState,
)
from corpus_studio.platform.runners import demo_training_plan


# --------------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------------
def _entry(path: str, role: str, data: bytes) -> CheckpointFileEntry:
    return CheckpointFileEntry(
        path=path, role=role, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


def _state(**over: object) -> SealedTrainingState:
    base: dict[str, object] = dict(
        scheduler_captured=True, scaler_captured=False, rng_captured=True,
        sampler_state_captured=True, rng_algorithm="philox", epoch=0.5,
        global_optimizer_step=6, microstep_within_step=0, gradient_accumulation_steps=4,
        consumed_microsteps=24,
    )
    base.update(over)
    return SealedTrainingState(**base)  # type: ignore[arg-type]


def _build_sealed_checkpoint(directory: Path, plan=None) -> CheckpointManifest:
    directory.mkdir(parents=True, exist_ok=True)
    opt = directory / "optimizer.pt"
    opt.write_bytes(b"optimizer-state")
    rng = directory / "rng.pt"
    rng.write_bytes(b"rng-state")
    files = sorted(
        [_entry("optimizer.pt", "optimizer", b"optimizer-state"), _entry("rng.pt", "rng", b"rng-state")],
        key=lambda e: e.path,
    )
    bound = ck.bound_identities_from_plan(plan or demo_training_plan(plan_id="demo-ckpt"))
    draft = CheckpointManifest(
        checkpoint_id="ckpt-000001",
        checkpoint_manifest_hash="0" * 64,
        source_run_id="run-parent01",
        created_at="2026-07-15T00:00:00+00:00",
        bound=bound,
        state=_state(),
        files=files,
    )
    sealed = ck.seal_checkpoint_manifest(draft)
    ck.write_checkpoint_manifest(sealed, directory)
    return sealed


# --------------------------------------------------------------------------------------------------
# Torch-free
# --------------------------------------------------------------------------------------------------
def test_checkpoint_import_is_torch_free() -> None:
    for heavy in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        assert heavy not in sys.modules, f"checkpoint pulled {heavy}"


# --------------------------------------------------------------------------------------------------
# Contract validators
# --------------------------------------------------------------------------------------------------
def test_checkpoint_files_must_be_sorted_unique_and_include_optimizer() -> None:
    bound = ck.bound_identities_from_plan(demo_training_plan())
    with pytest.raises(ValueError, match="sorted and unique"):
        CheckpointManifest(
            checkpoint_id="ckpt-1", checkpoint_manifest_hash="0" * 64, source_run_id="run-1",
            created_at="t", bound=bound, state=_state(),
            files=[_entry("b.pt", "optimizer", b"x"), _entry("a.pt", "rng", b"y")],
        )
    with pytest.raises(ValueError, match="optimizer state file"):
        CheckpointManifest(
            checkpoint_id="ckpt-1", checkpoint_manifest_hash="0" * 64, source_run_id="run-1",
            created_at="t", bound=bound, state=_state(),
            files=[_entry("a.pt", "rng", b"y")],
        )


def test_checkpoint_parent_id_and_hash_are_paired() -> None:
    bound = ck.bound_identities_from_plan(demo_training_plan())
    with pytest.raises(ValueError, match="provided together"):
        CheckpointManifest(
            checkpoint_id="ckpt-1", checkpoint_manifest_hash="0" * 64, source_run_id="run-1",
            created_at="t", bound=bound, state=_state(), parent_checkpoint_id="ckpt-0",
            files=[_entry("a.pt", "optimizer", b"y")],
        )


def test_sealed_state_rejects_microstep_past_accumulation_and_rng_without_algo() -> None:
    with pytest.raises(ValueError, match="less than gradient_accumulation_steps"):
        _state(microstep_within_step=4, gradient_accumulation_steps=4)
    with pytest.raises(ValueError, match="name its algorithm"):
        _state(rng_captured=True, rng_algorithm=None)


def test_checkpoint_file_path_must_be_canonical_relative() -> None:
    for bad in ("/abs/optimizer.pt", "../escape.pt", "a/./b.pt"):
        with pytest.raises(ValueError, match="canonical, non-escaping relative path"):
            CheckpointFileEntry(path=bad, role="optimizer", sha256="0" * 64, size_bytes=1)


def test_run_manifest_carries_optional_resume_lineage() -> None:
    manifest = RunManifest(
        run_id="run-resumed", plan_ref=Ref(id="plan", hash=HashRef(value="a" * 64)),
        created_at="t", updated_at="t", state="prepared",
    )
    assert manifest.resume_lineage is None  # ordinary from-scratch run
    assert "resume_lineage" in RunManifest.model_fields


def test_resume_request_round_trips() -> None:
    req = CheckpointResumeRequest(
        checkpoint_id="ckpt-1", checkpoint_manifest_hash="a" * 64, checkpoint_dir="/x/ckpt-1"
    )
    assert CheckpointResumeRequest.model_validate_json(req.model_dump_json()) == req


# --------------------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------------------
def test_seal_produces_verifiable_hash_excluding_itself(tmp_path: Path) -> None:
    sealed = _build_sealed_checkpoint(tmp_path / "c")
    assert sealed.complete is True
    assert ck.verify_checkpoint_manifest_hash(sealed) is True
    # The hash covers the body, not the hash field: zeroing it changes nothing in the payload.
    payload = ck.checkpoint_manifest_hash_payload(sealed)
    assert "checkpoint_manifest_hash" not in payload
    assert ck.compute_checkpoint_manifest_hash(payload) == sealed.checkpoint_manifest_hash


def test_tampered_manifest_body_fails_hash(tmp_path: Path) -> None:
    sealed = _build_sealed_checkpoint(tmp_path / "c")
    tampered = sealed.model_copy(update={"state": _state(global_optimizer_step=999)})
    assert ck.verify_checkpoint_manifest_hash(tampered) is False


# --------------------------------------------------------------------------------------------------
# Integrity (crash-recovery / external change) - fail closed
# --------------------------------------------------------------------------------------------------
def test_integrity_verifies_a_sealed_checkpoint(tmp_path: Path) -> None:
    cdir = tmp_path / "c"
    _build_sealed_checkpoint(cdir)
    verified = ck.verify_checkpoint_integrity(cdir)
    assert verified.complete and verified.checkpoint_id == "ckpt-000001"


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    cdir = tmp_path / "c"
    cdir.mkdir()
    (cdir / "optimizer.pt").write_bytes(b"optimizer-state")
    bound = ck.bound_identities_from_plan(demo_training_plan())
    draft = CheckpointManifest(
        checkpoint_id="ckpt-1", checkpoint_manifest_hash="0" * 64, source_run_id="run-1",
        created_at="t", bound=bound, state=_state(),
        files=[_entry("optimizer.pt", "optimizer", b"optimizer-state")],
    )  # complete defaults False
    ck.write_checkpoint_manifest(draft, cdir)
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(cdir)
    assert exc.value.reason == "incomplete"


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    cdir = tmp_path / "c"
    _build_sealed_checkpoint(cdir)
    (cdir / "rng.pt").unlink()
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(cdir)
    assert exc.value.reason == "missing_file"


def test_externally_changed_file_is_rejected(tmp_path: Path) -> None:
    cdir = tmp_path / "c"
    _build_sealed_checkpoint(cdir)
    (cdir / "optimizer.pt").write_bytes(b"different-bytes-same-name")
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(cdir)
    assert exc.value.reason == "external_change"


def test_hash_mismatch_manifest_is_rejected(tmp_path: Path) -> None:
    cdir = tmp_path / "c"
    sealed = _build_sealed_checkpoint(cdir)
    # Rewrite the manifest with an altered body but the old (now-wrong) hash.
    forged = sealed.model_copy(update={"source_run_id": "run-forged1"})
    (cdir / ck.MANIFEST_FILENAME).write_text(forged.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(cdir)
    assert exc.value.reason == "hash_mismatch"


def test_missing_and_malformed_manifest_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(tmp_path / "nope")
    assert exc.value.reason == "missing_file"
    bad = tmp_path / "c"
    bad.mkdir()
    (bad / ck.MANIFEST_FILENAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_checkpoint_integrity(bad)
    assert exc.value.reason == "malformed"


# --------------------------------------------------------------------------------------------------
# Resume compatibility (parent admission) - fail closed
# --------------------------------------------------------------------------------------------------
def test_bound_identities_require_resolved_execution() -> None:
    from corpus_studio.platform.supervisor import demo_run_plan

    echo_plan = demo_run_plan()  # no resolved_execution
    with pytest.raises(ck.CheckpointError) as exc:
        ck.bound_identities_from_plan(echo_plan)
    assert exc.value.reason == "ambiguous"


def test_resumable_into_admits_the_same_plan(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    _build_sealed_checkpoint(tmp_path / "c", plan=plan)
    lineage = ck.admit_resume(plan, tmp_path / "c", resumed_run_id="run-resumed01")
    assert lineage.parent_run_id == "run-parent01"
    assert lineage.parent_checkpoint_id == "ckpt-000001"
    assert lineage.resumed_from_global_step == 6


def test_resumable_into_rejects_incompatible_plan(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    sealed = _build_sealed_checkpoint(tmp_path / "c", plan=plan)
    # A different plan hash is the canonical incompatibility.
    other = plan.model_copy(update={"plan_hash": "b" * 64})
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_resumable_into(sealed, other)
    assert exc.value.reason == "incompatible"


def test_resumable_into_rejects_mutated_bound_identity(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    sealed = _build_sealed_checkpoint(tmp_path / "c", plan=plan)
    # Forge the manifest to bind a different objective, then it must not admit the real plan.
    forged_bound = sealed.bound.model_copy(update={"objective_ref": Ref(id="different-objective")})
    forged = sealed.model_copy(update={"bound": forged_bound})
    with pytest.raises(ck.CheckpointError) as exc:
        ck.verify_resumable_into(forged, plan)
    assert exc.value.reason == "incompatible"


def test_admit_resume_requires_a_fresh_run_id(tmp_path: Path) -> None:
    plan = demo_training_plan(plan_id="demo-ckpt")
    _build_sealed_checkpoint(tmp_path / "c", plan=plan)
    with pytest.raises(ck.CheckpointError) as exc:
        ck.admit_resume(plan, tmp_path / "c", resumed_run_id="run-parent01")  # == source_run_id
    assert exc.value.reason == "ambiguous"


# --------------------------------------------------------------------------------------------------
# CLI (GPU-free)
# --------------------------------------------------------------------------------------------------
def test_cli_checkpoint_verify_ok_and_incompatible(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from corpus_studio.cli import app

    runner = CliRunner()
    plan = demo_training_plan(plan_id="demo-ckpt")
    cdir = tmp_path / "c"
    _build_sealed_checkpoint(cdir, plan=plan)

    ok = runner.invoke(app, ["checkpoint-verify", str(cdir)])
    assert ok.exit_code == 0, ok.output
    assert "verified" in ok.output and "resume from optimizer step 6" in ok.output

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    compat = runner.invoke(app, ["checkpoint-verify", str(cdir), "--plan", str(plan_path)])
    assert compat.exit_code == 0 and "compatible" in compat.output

    other = tmp_path / "other.json"
    other.write_text(plan.model_copy(update={"plan_hash": "b" * 64}).model_dump_json(), encoding="utf-8")
    bad = runner.invoke(app, ["checkpoint-verify", str(cdir), "--plan", str(other)])
    assert bad.exit_code == 1 and "NOT a compatible resume source" in bad.output


def test_cli_checkpoint_verify_fails_closed_on_tamper(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from corpus_studio.cli import app

    runner = CliRunner()
    cdir = tmp_path / "c"
    _build_sealed_checkpoint(cdir)
    (cdir / "optimizer.pt").write_bytes(b"tampered")
    result = runner.invoke(app, ["checkpoint-verify", str(cdir)])
    assert result.exit_code == 1 and "integrity FAILED (external_change)" in result.output
