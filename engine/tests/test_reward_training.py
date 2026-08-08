"""The pairwise reward-modeling primitive's PURE + fail-closed logic (RL slice S5a): the deterministic
held-out split, the backbone/score-head accessor, the Bradley-Terry loss, and the concrete-step schedule.
The live GPU loop (run_reward_training / evaluate_reward_accuracy / _score_reward_branch / run_reward) is
pragma, proven by a GPU run in the runner slice."""

import types

import pytest

from corpus_studio.training.reward_worker import concrete_reward_max_steps
from corpus_studio.training.trainer import (
    TrainerError,
    _seqcls_backbone_and_score_head,
    reward_heldout_split,
    reward_pairwise_loss,
)


# --- reward_heldout_split: the deterministic seeded train / held-out carve-out (PURE) -----------------


def test_reward_heldout_split_is_deterministic_disjoint_and_total() -> None:
    train, heldout = reward_heldout_split(20, data_seed=42)
    # same (n, seed, ratio) -> identical split, so the sealed run is reproducible
    assert reward_heldout_split(20, data_seed=42) == (train, heldout)
    assert set(train).isdisjoint(heldout)
    assert sorted(train + heldout) == list(range(20))  # partitions every pair
    assert heldout == sorted(heldout)  # stable order
    assert len(heldout) == 4  # round(20 * 0.2)
    assert train and heldout  # both non-empty


def test_reward_heldout_split_varies_with_seed() -> None:
    assert reward_heldout_split(50, data_seed=1) != reward_heldout_split(50, data_seed=2)


def test_reward_heldout_split_always_holds_out_one_and_keeps_one() -> None:
    # a tiny dataset still yields at least one held-out ranking pair AND at least one training pair
    train, heldout = reward_heldout_split(2, data_seed=7)
    assert len(heldout) == 1 and len(train) == 1
    # a large ratio can never consume the last training pair
    train_big, heldout_big = reward_heldout_split(3, data_seed=7, heldout_ratio=0.99)
    assert len(train_big) == 1 and len(heldout_big) == 2


def test_reward_heldout_split_refuses_below_two_pairs() -> None:
    with pytest.raises(TrainerError, match="needs >= 2 preference pairs"):
        reward_heldout_split(1, data_seed=42)


def test_reward_heldout_split_refuses_a_ratio_outside_the_open_unit_interval() -> None:
    with pytest.raises(TrainerError, match="open interval"):
        reward_heldout_split(10, data_seed=42, heldout_ratio=0.0)
    with pytest.raises(TrainerError, match="open interval"):
        reward_heldout_split(10, data_seed=42, heldout_ratio=1.0)


# --- _seqcls_backbone_and_score_head: architecture-tolerant accessor, fail-closed -------------------


class _FakeModule:
    pass


class _FakeSeqClsBase:
    def __init__(self, *, backbone_attr: str | None, head_attr: str | None) -> None:
        if backbone_attr is not None:
            setattr(self, backbone_attr, _FakeModule())
        if head_attr is not None:
            setattr(self, head_attr, _FakeModule())


class _FakePeftModel:
    def __init__(self, base: _FakeSeqClsBase) -> None:
        self._base = base

    def get_base_model(self) -> _FakeSeqClsBase:
        return self._base


def test_seqcls_accessor_locates_backbone_and_score_head() -> None:
    base = _FakeSeqClsBase(backbone_attr="model", head_attr="score")
    backbone, head = _seqcls_backbone_and_score_head(_FakePeftModel(base))
    assert backbone is base.model and head is base.score


def test_seqcls_accessor_tolerates_alternate_naming() -> None:
    base = _FakeSeqClsBase(backbone_attr="transformer", head_attr="classifier")
    backbone, head = _seqcls_backbone_and_score_head(_FakePeftModel(base))
    assert backbone is base.transformer and head is base.classifier


def test_seqcls_accessor_fails_closed_on_missing_backbone() -> None:
    base = _FakeSeqClsBase(backbone_attr=None, head_attr="score")
    with pytest.raises(TrainerError, match="could not locate a decoder backbone"):
        _seqcls_backbone_and_score_head(_FakePeftModel(base))


def test_seqcls_accessor_fails_closed_on_missing_score_head() -> None:
    base = _FakeSeqClsBase(backbone_attr="model", head_attr=None)
    with pytest.raises(TrainerError, match="could not locate a decoder backbone"):
        _seqcls_backbone_and_score_head(_FakePeftModel(base))


# --- concrete_reward_max_steps: sealed schedule -> concrete optimizer steps over TRAINING pairs -------


def _schedule_stub(*, max_steps, num_train_epochs, grad_accum):
    return types.SimpleNamespace(
        schedule=types.SimpleNamespace(max_steps=max_steps, num_train_epochs=num_train_epochs),
        batching=types.SimpleNamespace(fallback_grad_accumulation_steps=grad_accum),
    )


def test_concrete_reward_max_steps_uses_an_explicit_step_count_verbatim() -> None:
    execution = _schedule_stub(max_steps=25, num_train_epochs=3.0, grad_accum=4)
    assert concrete_reward_max_steps(execution, train_pair_count=100) == 25


def test_concrete_reward_max_steps_converts_epochs_over_the_training_split() -> None:
    execution = _schedule_stub(max_steps=None, num_train_epochs=2.0, grad_accum=4)
    # ceil(10 / 4) = 3 steps/epoch * 2 epochs = 6
    assert concrete_reward_max_steps(execution, train_pair_count=10) == 6


def test_concrete_reward_max_steps_is_at_least_one() -> None:
    execution = _schedule_stub(max_steps=None, num_train_epochs=1.0, grad_accum=8)
    assert concrete_reward_max_steps(execution, train_pair_count=1) == 1


# --- reward_pairwise_loss: the Bradley-Terry pairwise loss (torch-guarded) --------------------------


def test_reward_pairwise_loss_is_log2_at_equal_scores_and_falls_as_chosen_wins() -> None:
    torch = pytest.importorskip("torch")

    equal = torch.tensor(0.5)
    loss, margin = reward_pairwise_loss(equal, equal)
    assert float(loss) == pytest.approx(0.6931471805599453, abs=1e-6)  # -log sigmoid(0) == log 2
    assert float(margin) == pytest.approx(0.0)

    # as the chosen score beats the rejected, the loss falls
    better, better_margin = reward_pairwise_loss(torch.tensor(3.0), torch.tensor(-1.0))
    assert float(better) < float(loss)
    # the RETURNED margin is chosen - rejected WITHOUT the hyperparameter subtracted (evidence consistency)
    assert float(better_margin) == pytest.approx(4.0)


def test_reward_pairwise_loss_margin_hyperparameter_raises_the_bar() -> None:
    torch = pytest.importorskip("torch")

    chosen, rejected = torch.tensor(1.0), torch.tensor(0.0)
    base, base_margin = reward_pairwise_loss(chosen, rejected, margin=0.0)
    demanding, demanding_margin = reward_pairwise_loss(chosen, rejected, margin=0.5)
    # requiring a bigger gap makes the same scores cost more, but the recorded margin is unchanged
    assert float(demanding) > float(base)
    assert float(demanding_margin) == pytest.approx(float(base_margin)) == pytest.approx(1.0)
