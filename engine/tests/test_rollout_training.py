"""The on-policy RL / GRPO math core (RL slice S5b): the PURE group-relative advantage + the torch-thin
clipped surrogate loss, so GRPO correctness is locked in the base gate (not deferred to the GPU run). The
live rollout-generation + reward-serving loop (run_rollout_training) is pragma, proven by a GPU run."""

import pytest

from corpus_studio.training.trainer import TrainerError, grpo_group_advantage, grpo_policy_loss


# --- grpo_group_advantage: the critic-free, group-relative advantage (PURE) -------------------------


def test_grpo_group_advantage_is_group_relative_and_zero_mean() -> None:
    adv = grpo_group_advantage([1.0, 2.0, 3.0])
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)  # normalized => zero group mean
    assert adv[0] < 0 < adv[2]  # below-average suppressed, above-average reinforced
    assert adv[1] == pytest.approx(0.0, abs=1e-6)  # the group average is neutral


def test_grpo_group_advantage_uniform_group_teaches_nothing() -> None:
    # a group the reward source could not separate (std 0) yields ~zero advantages - nothing to learn.
    adv = grpo_group_advantage([0.5, 0.5, 0.5])
    assert all(a == pytest.approx(0.0, abs=1e-3) for a in adv)


def test_grpo_group_advantage_refuses_a_degenerate_group() -> None:
    with pytest.raises(TrainerError, match="group of at least two rollouts"):
        grpo_group_advantage([1.0])


def test_grpo_group_advantage_refuses_non_finite_rewards() -> None:
    with pytest.raises(TrainerError, match="finite numbers"):
        grpo_group_advantage([1.0, float("inf")])


# --- grpo_policy_loss: the PPO-clipped surrogate + k3 KL (torch-guarded) ----------------------------


def test_grpo_policy_loss_reduces_to_reinforce_on_the_first_on_policy_update() -> None:
    torch = pytest.importorskip("torch")

    logp = torch.tensor([-0.5, -1.0, -2.0])
    adv = torch.tensor([1.0, 0.0, -1.0])
    # first update: old == policy (ratio 1) and ref == policy (KL 0)
    loss, kl = grpo_policy_loss(logp, logp, logp, adv, kl_coefficient=0.05)
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert float(loss) == pytest.approx(float(-(adv * logp).mean()), abs=1e-6)


def test_grpo_policy_loss_kl_penalizes_reference_drift() -> None:
    torch = pytest.importorskip("torch")

    policy = torch.tensor([-0.5, -0.5])
    ref = torch.tensor([-1.5, -1.5])  # reference differs -> positive KL
    adv = torch.tensor([0.0, 0.0])  # isolate the KL term (no policy-gradient contribution)
    loss_lo, kl = grpo_policy_loss(policy, policy, ref, adv, kl_coefficient=0.1)
    loss_hi, _ = grpo_policy_loss(policy, policy, ref, adv, kl_coefficient=1.0)
    assert float(kl) > 0.0  # k3 estimator is non-negative and non-zero on drift
    assert float(loss_hi) > float(loss_lo)  # a bigger KL coefficient costs more


def test_grpo_policy_loss_clips_a_positive_advantage_when_the_ratio_grows() -> None:
    torch = pytest.importorskip("torch")

    old = torch.tensor([-1.0])
    policy = torch.tensor([-0.2])  # policy moved up a lot -> ratio = exp(0.8) ~ 2.2, beyond 1+clip
    ref = policy
    adv = torch.tensor([1.0])  # positive advantage: the clip caps the surrogate
    loss, _ = grpo_policy_loss(old_logprobs=old, policy_logprobs=policy, ref_logprobs=ref,
                              advantages=adv, clip_range=0.2, kl_coefficient=0.0)
    # clipped surrogate uses (1 + clip_range) * A, so loss == -(1.2 * 1.0)
    assert float(loss) == pytest.approx(-1.2, abs=1e-4)
