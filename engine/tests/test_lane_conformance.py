"""Checks over the lane conformance matrix (``lane_conformance.MATRIX``).

The matrix declares, per sealed execution variant and per execution guarantee, whether the
guarantee is enforced, a tracked gap, or not applicable. These tests keep that declaration honest:

* every variant in ``execution_config.RESOLVED_EXECUTION_FIELDS`` declares every guarantee, so a NEW
  variant cannot ship without stating where it stands (the defect class behind #860 to #863);
* an ``ENFORCED`` cell names a test that exists AND, when that test is lane-parametrized, one whose
  parametrization actually runs this lane - so removing a lane from a parametrize list fails here
  instead of leaving a cell claiming a guarantee nothing checks for it;
* a ``GAP`` cell names a tracking issue, so a known gap stays visible work rather than folklore;
* a ``NOT_APPLICABLE`` cell explains itself.
"""

from __future__ import annotations

import pytest
from lane_conformance import GUARANTEES, LANE_LABELS, MATRIX, Status, proof_lane_labels, resolve_proof

from corpus_studio.platform.execution_config import RESOLVED_EXECUTION_FIELDS

GUARANTEE_IDS = tuple(guarantee.id for guarantee in GUARANTEES)
CELLS = tuple(
    (guarantee_id, plan_field, MATRIX[guarantee_id][plan_field])
    for guarantee_id in GUARANTEE_IDS
    for plan_field in sorted(MATRIX[guarantee_id])
)


def test_guarantee_ids_are_unique_and_documented():
    assert len(GUARANTEE_IDS) == len(set(GUARANTEE_IDS))
    for guarantee in GUARANTEES:
        assert guarantee.summary.strip(), f"guarantee {guarantee.id!r} has no summary"
        assert guarantee.summary.isascii()


def test_every_variant_has_a_lane_label():
    # The label is how a cell reaches the test suite's own vocabulary; a new variant that adds no
    # label would silently skip the lane-coverage check below rather than fail it.
    assert set(LANE_LABELS) == set(RESOLVED_EXECUTION_FIELDS)
    assert len(set(LANE_LABELS.values())) == len(LANE_LABELS)


def test_every_variant_declares_every_guarantee():
    # The point of the matrix: a new resolved execution variant cannot reach execution without
    # stating, for every guarantee, that it is enforced, a tracked gap, or not applicable.
    expected = set(RESOLVED_EXECUTION_FIELDS)
    assert expected, "the execution-variant table is empty"
    for guarantee_id in GUARANTEE_IDS:
        declared = set(MATRIX[guarantee_id])
        missing = expected - declared
        assert not missing, (
            f"guarantee {guarantee_id!r} does not declare a status for {sorted(missing)}; "
            "add a cell to lane_conformance.MATRIX"
        )
        unknown = declared - expected
        assert not unknown, (
            f"guarantee {guarantee_id!r} declares {sorted(unknown)}, which is not a resolved "
            "execution variant"
        )


def test_matrix_declares_no_guarantee_outside_the_registry():
    assert set(MATRIX) == set(GUARANTEE_IDS)


@pytest.mark.parametrize(
    ("guarantee_id", "plan_field", "cell"),
    [pytest.param(*cell, id=f"{cell[0]}-{cell[1]}") for cell in CELLS],
)
def test_cell_is_self_consistent(guarantee_id, plan_field, cell):
    assert cell.note.strip(), f"{guarantee_id}/{plan_field} has no note"
    assert cell.note.isascii(), f"{guarantee_id}/{plan_field} note must be ASCII"
    if cell.status is Status.ENFORCED:
        assert cell.proof, f"{guarantee_id}/{plan_field} is enforced but names no proof"
        assert cell.issue is None, f"{guarantee_id}/{plan_field} is enforced but tracks an issue"
        function = resolve_proof(cell.proof)
        labels = proof_lane_labels(function)
        if labels is not None:
            # A lane-parametrized proof must actually run THIS lane. A single-lane proof (labels is
            # None) names its lane in its own body and cannot drift this way.
            label = LANE_LABELS[plan_field]
            assert label in labels, (
                f"{guarantee_id}/{plan_field} cites {cell.proof}, which parametrizes lanes "
                f"{sorted(labels)} and does not run {label!r}"
            )
    elif cell.status in (Status.GAP, Status.UNPROVEN):
        # Both mean work is outstanding - a missing guarantee, or a guarantee nothing watches - so
        # both owe a tracking issue and neither may claim a proof.
        assert isinstance(cell.issue, int) and cell.issue > 0, (
            f"{guarantee_id}/{plan_field} is {cell.status.value} but names no tracking issue"
        )
        assert cell.proof is None, (
            f"{guarantee_id}/{plan_field} is {cell.status.value} but claims a proof"
        )
    else:
        assert cell.proof is None and cell.issue is None, (
            f"{guarantee_id}/{plan_field} is not applicable but claims a proof or an issue"
        )
