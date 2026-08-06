"""The pure helpers of the DPO config-consuming worker (run_preference's torch training path is
# pragma, proven by a GPU run): the objective-derived EOS masking + the sealed epoch->steps conversion."""

from types import SimpleNamespace

from corpus_studio.training.preference_worker import (
    concrete_max_steps,
    paged_adamw_requested,
    score_response_eos_for,
)


def test_score_response_eos_derived_from_the_objective_primary_mask() -> None:
    # dpo_qlora's primary mask sets include_special_tokens=False -> the appended EOS is NOT scored.
    off = SimpleNamespace(
        primary_mask_ref="primary_mask",
        token_masks=[SimpleNamespace(mask_id="primary_mask", include_special_tokens=False)],
    )
    assert score_response_eos_for(off) is False
    on = SimpleNamespace(
        primary_mask_ref="primary_mask",
        token_masks=[SimpleNamespace(mask_id="primary_mask", include_special_tokens=True)],
    )
    assert score_response_eos_for(on) is True
    # an objective with no matching mask fails safe (do not score the EOS)
    assert score_response_eos_for(SimpleNamespace()) is False


def test_concrete_max_steps_uses_sealed_max_steps_verbatim() -> None:
    execution = SimpleNamespace(
        schedule=SimpleNamespace(max_steps=15, num_train_epochs=None),
        batching=SimpleNamespace(fallback_grad_accumulation_steps=1),
    )
    assert concrete_max_steps(execution, pair_count=100) == 15


def test_concrete_max_steps_converts_epochs_using_the_pair_count() -> None:
    # epoch mode: ceil(pairs / grad_accum) steps per epoch * epochs (one pair per microbatch).
    execution = SimpleNamespace(
        schedule=SimpleNamespace(max_steps=None, num_train_epochs=3),
        batching=SimpleNamespace(fallback_grad_accumulation_steps=2),
    )
    assert concrete_max_steps(execution, pair_count=10) == 15  # ceil(10/2)=5 * 3
    partial = SimpleNamespace(
        schedule=SimpleNamespace(max_steps=None, num_train_epochs=1),
        batching=SimpleNamespace(fallback_grad_accumulation_steps=2),
    )
    assert concrete_max_steps(partial, pair_count=9) == 5  # ceil(9/2)=5, rounds up


def test_paged_adamw_requested_reads_the_sealed_impl() -> None:
    # the DPO worker builds the optimizer the seal names; paged 8-bit AdamW is what fits seq 4096 in 12 GB.
    assert paged_adamw_requested(SimpleNamespace(impl=SimpleNamespace(value="paged_adamw_8bit"))) is True
    assert paged_adamw_requested(SimpleNamespace(impl=SimpleNamespace(value="adamw_torch"))) is False
    # tolerant of a bare-string impl (no .value); anything that is not paged 8-bit is not paged
    assert paged_adamw_requested(SimpleNamespace(impl="paged_adamw_8bit")) is True
    assert paged_adamw_requested(SimpleNamespace(impl="adamw_torch")) is False
