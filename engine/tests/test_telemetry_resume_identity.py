"""A resumed trial must be distinguishable from an uninterrupted one in paper evidence (#440 + §11).

Torch-free: the resume distinction is carried on ``TelemetryIdentity`` and derived straight from
``RunManifest.resume_lineage`` so aggregation never conflates a resumed trial with a from-scratch one.
"""

from __future__ import annotations

from corpus_studio.platform.contracts import (
    HashRef,
    Ref,
    ResumeLineage,
    RunManifest,
    TelemetryIdentity,
)
from corpus_studio.platform.telemetry import identity_from_plan


def _manifest(resume: ResumeLineage | None) -> RunManifest:
    return RunManifest(
        run_id="run-child01",
        plan_ref=Ref(id="plan", hash=HashRef(value="a" * 64)),
        created_at="t",
        updated_at="t",
        state="succeeded",
        resume_lineage=resume,
    )


def test_from_scratch_run_is_not_marked_resumed() -> None:
    identity = identity_from_plan(None, _manifest(None))
    assert identity.resumed is False
    assert identity.parent_run_id is None
    assert identity.resumed_from_global_step is None


def test_resumed_run_carries_parent_lineage() -> None:
    lineage = ResumeLineage(
        parent_run_id="run-parent01",
        parent_checkpoint_id="ckpt-000006",
        parent_checkpoint_hash="b" * 64,
        resumed_from_global_step=6,
    )
    identity = identity_from_plan(None, _manifest(lineage))
    assert identity.resumed is True
    assert identity.parent_run_id == "run-parent01"
    assert identity.resumed_from_global_step == 6


def test_resumed_from_global_step_requires_at_least_one() -> None:
    # The contract forbids a nonsensical "resumed from step 0"; a resume always continues after >=1.
    import pytest

    with pytest.raises(ValueError):
        TelemetryIdentity(resumed=True, resumed_from_global_step=0)
