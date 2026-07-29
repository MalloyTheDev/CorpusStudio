"""SupportLevel projection (Training Systems P0a, #481).

``SupportLevel`` (``enums.py``) is a coarse, single-value capability-support rollup. It COEXISTS with
the shipped multi-axis verification ladders - it never replaces them, and it never states WHICH axis
is proven. This module holds the **lossy partial projection** from a multi-axis record onto one
``SupportLevel``: it is defined only for the affirmatively-proven axes, and it CARRIES the unproven
ones (``not_verified`` / ``not_applicable``) verbatim instead of collapsing them into a misleading
higher level.

The objective-verification -> SupportLevel mapping (lossy; the objective ladder can evidence only this
subset - ``config_generation_only`` / ``installed`` / ``production_supported`` / ``refused`` come from
environment / manifest / operational evidence it does not carry):

    hardware axis == hardware_verified            -> workload_verified   (a measured run)
    else implementation axis functionally proven  -> probed              (a passed functional probe)
    else                                          -> declared            (a claim, no proven path)

Control-plane only: stdlib + platform contracts, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from corpus_studio.platform.contracts import ObjectiveVerification
from corpus_studio.platform.enums import ObjectiveVerificationStatus, SupportLevel

# A single objective-verification axis carries no affirmative proof in these states, so the projection
# CARRIES them (names them in ``carried``) rather than promoting the rollup on their behalf.
_UNPROVEN: frozenset[ObjectiveVerificationStatus] = frozenset(
    {ObjectiveVerificationStatus.not_verified, ObjectiveVerificationStatus.not_applicable}
)
# The implementation axis is "functionally proven" (has passed a functional probe) in these states -
# the evidence a ``probed`` rollup requires. ``hardware_verified`` is included because it is strictly
# STRONGER than ``functional_verified`` on the ladder, so an implementation axis that reached it has
# certainly passed a functional probe. It does NOT promote the rollup to ``workload_verified``:
# ``workload_verified`` is gated on the HARDWARE axis specifically, so an implementation axis at
# ``hardware_verified`` while the hardware axis is unproven is still only ``probed`` (with the unproven
# hardware axis carried) - the projection never re-labels one axis's proof as another axis's.
_FUNCTIONALLY_PROVEN: frozenset[ObjectiveVerificationStatus] = frozenset(
    {ObjectiveVerificationStatus.functional_verified, ObjectiveVerificationStatus.hardware_verified}
)


class CarriedAxis(NamedTuple):
    """One verification axis the projection did NOT fold into the rollup level, carried verbatim so the
    coarse ``level`` can never be mistaken for full multi-axis proof. ``status`` is the ladder status
    value, kept as a string so ``carried`` stays ladder-agnostic as more projections are added."""

    axis: str
    status: str


@dataclass(frozen=True)
class SupportLevelRollup:
    """A lossy partial projection of a multi-axis verification record onto one ``SupportLevel``.

    ``level`` is derived ONLY from the affirmatively-proven axes. ``carried`` names the axes that were
    NOT projected (``not_verified`` / ``not_applicable``) - the projection carries them so a reader can
    never mistake the coarse ``level`` for full multi-axis proof. The multi-axis record stays
    authoritative for WHICH axis is proven; this is a summary, never a replacement."""

    level: SupportLevel
    carried: tuple[CarriedAxis, ...] = ()


def project_objective_verification(verification: ObjectiveVerification) -> SupportLevelRollup:
    """Project the 3-axis :class:`ObjectiveVerification` onto a single :class:`SupportLevel`.

    Lossy, partial, anti-collapse (the full mapping is in the module docstring):

    * ``workload_verified`` ONLY when the hardware axis is ``hardware_verified`` (a measured run);
    * ``probed`` ONLY when the implementation axis is functionally proven (a passed functional probe);
    * ``declared`` otherwise - a claim without a proven runnable path.

    ``not_verified`` / ``not_applicable`` on any axis are CARRIED (a :class:`CarriedAxis` in
    ``carried``), never promoted. The objective ladder cannot evidence ``config_generation_only`` /
    ``installed`` / ``production_supported`` / ``refused``, so this projection never emits them.
    """

    carried = tuple(
        CarriedAxis(axis=axis, status=status.value)
        for axis, status in (
            ("definition", verification.definition),
            ("implementation", verification.implementation),
            ("hardware", verification.hardware),
        )
        if status in _UNPROVEN
    )
    if verification.hardware == ObjectiveVerificationStatus.hardware_verified:
        level = SupportLevel.workload_verified
    elif verification.implementation in _FUNCTIONALLY_PROVEN:
        level = SupportLevel.probed
    else:
        level = SupportLevel.declared
    return SupportLevelRollup(level=level, carried=carried)
