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


# --- grpo_policy_loss: the PPO-clipped surrogate + k3 KL, per-completion masked (torch-guarded) -----


def test_grpo_policy_loss_ratio_one_surrogate_value_is_minus_mean_advantage() -> None:
    torch = pytest.importorskip("torch")

    logp = torch.tensor([[-0.5, -1.0, -2.0]])  # [G=1, T=3]
    mask = torch.ones_like(logp)
    adv = torch.tensor([1.0])  # [G=1]
    # first update: old == policy (ratio 1) and ref == policy (KL 0). The surrogate VALUE is -mean(A),
    # NOT -mean(A*logprob) (that identity is the GRADIENT). Assert the value.
    loss, kl = grpo_policy_loss(logp, logp, logp, adv, mask, kl_coefficient=0.05)
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert float(loss) == pytest.approx(float(-adv.mean()), abs=1e-6)


def test_grpo_policy_loss_kl_is_fp32_stable_and_penalizes_reference_drift() -> None:
    torch = pytest.importorskip("torch")

    policy = torch.tensor([[-0.5, -0.5]])
    ref = torch.tensor([[-1.5, -1.5]])  # reference differs -> positive KL
    mask = torch.ones_like(policy)
    adv = torch.tensor([0.0])  # isolate the KL term (no policy-gradient contribution)
    loss_lo, kl = grpo_policy_loss(policy, policy, ref, adv, mask, kl_coefficient=0.1)
    loss_hi, _ = grpo_policy_loss(policy, policy, ref, adv, mask, kl_coefficient=1.0)
    assert float(kl) > 0.0 and kl.dtype == torch.float32  # k3 estimator, computed in float32
    assert float(loss_hi) > float(loss_lo)  # a bigger KL coefficient costs more

    # bf16 cancellation guard: a tiny log-ratio still yields a strictly positive k3 KL (not 0 / negative).
    tiny = torch.tensor([[0.01, 0.01]], dtype=torch.bfloat16)
    _, kl_tiny = grpo_policy_loss(tiny, tiny, torch.zeros_like(tiny), torch.tensor([0.0]),
                                  torch.ones_like(tiny))
    assert float(kl_tiny) > 0.0


def test_grpo_policy_loss_clips_a_positive_advantage_when_the_ratio_grows() -> None:
    torch = pytest.importorskip("torch")

    old = torch.tensor([[-1.0]])
    policy = torch.tensor([[-0.2]])  # ratio = exp(0.8) ~ 2.2, beyond 1+clip
    adv = torch.tensor([1.0])  # positive advantage: the clip caps the surrogate at (1+clip)*A
    loss, _ = grpo_policy_loss(policy, old, policy, adv, torch.ones_like(old),
                              clip_range=0.2, kl_coefficient=0.0)
    assert float(loss) == pytest.approx(-1.2, abs=1e-4)


def test_grpo_policy_loss_masks_padding_without_nan_poisoning() -> None:
    torch = pytest.importorskip("torch")
    import math

    # a pad position carries an EXTREME reference divergence (log-ratio 100 -> expm1 == inf); the mask must
    # REPLACE it with 0, not multiply (0 * inf == NaN), so the group KL + loss stay FINITE and reflect only
    # the valid tokens (ref == policy there -> KL 0; ratio 1 -> -mean(A) = -1.0).
    policy = torch.tensor([[-0.5, -0.5, 0.0]])
    old = torch.tensor([[-0.5, -0.5, 0.0]])
    ref = torch.tensor([[-0.5, -0.5, 100.0]])  # pad log-ratio 100 -> expm1 overflows to inf
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, kl = grpo_policy_loss(policy, old, ref, torch.tensor([1.0]), mask, kl_coefficient=1.0)
    assert math.isfinite(float(loss)) and math.isfinite(float(kl))
    assert float(kl) == pytest.approx(0.0, abs=1e-6)
    assert float(loss) == pytest.approx(-1.0, abs=1e-4)


def test_grpo_policy_loss_refuses_an_empty_completion_row() -> None:
    torch = pytest.importorskip("torch")

    logp = torch.tensor([[-0.5, -0.5], [-0.5, -0.5]])  # G=2
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # the second completion has NO valid tokens
    with pytest.raises(TrainerError, match="at least one valid"):
        grpo_policy_loss(logp, logp, logp, torch.tensor([1.0, 1.0]), mask)


def test_grpo_policy_loss_refuses_a_nonzero_bonus_without_an_entropy_tensor() -> None:
    torch = pytest.importorskip("torch")

    logp = torch.tensor([[-0.5, -0.5]])
    with pytest.raises(TrainerError, match="entropy_bonus requires an entropy tensor"):
        grpo_policy_loss(logp, logp, logp, torch.tensor([0.0]), torch.ones_like(logp),
                        entropy_bonus=0.1, entropy=None)
