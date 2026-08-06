"""The shared sealed-``OptimizerSpec`` -> optimizer lowering (S0). The PURE surfaces - paged-8bit detection
and the full HF ``TrainingArguments`` optimizer kwargs - are gate-tested here; ``build_torch_optimizer``'s
torch/bitsandbytes path is ``# pragma`` (GPU-proven). The kwargs test is the ANTI-DRIFT regression: it fails
the moment any sealed optimizer field stops being threaded (the audit's F1/F3 defect class)."""

from types import SimpleNamespace

from corpus_studio.training.optimizer_config import (
    hf_training_arguments_optimizer_kwargs,
    paged_adamw_requested,
)

_KEYS = {
    "optim", "learning_rate", "weight_decay", "adam_beta1", "adam_beta2",
    "adam_epsilon", "max_grad_norm", "lr_scheduler_type", "warmup_ratio",
}


def _spec(**over):
    base = dict(
        impl=SimpleNamespace(value="adamw_torch"), learning_rate=2e-4, weight_decay=0.01,
        adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, max_grad_norm=1.0,
        lr_scheduler="cosine", warmup_ratio=0.03,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_paged_adamw_requested_reads_the_sealed_impl() -> None:
    # paged 8-bit AdamW is what fits 4B QLoRA DPO in 12 GB at seq 4096 - the worker must build the seal's.
    assert paged_adamw_requested(_spec(impl=SimpleNamespace(value="paged_adamw_8bit"))) is True
    assert paged_adamw_requested(_spec(impl=SimpleNamespace(value="adamw_torch"))) is False
    # tolerant of a bare-string impl (no .value); anything that is not paged 8-bit is not paged
    assert paged_adamw_requested(SimpleNamespace(impl="paged_adamw_8bit")) is True
    assert paged_adamw_requested(SimpleNamespace(impl="adamw_torch")) is False


def test_hf_kwargs_thread_every_sealed_optimizer_field() -> None:
    # anti-F3 regression: ALL nine optimizer fields present + carrying the sealed value, exact-match.
    assert hf_training_arguments_optimizer_kwargs(_spec()) == {
        "optim": "adamw_torch",
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
    }
    # the paged 8-bit impl flows through as the optim string (never silently downgraded)
    paged = hf_training_arguments_optimizer_kwargs(_spec(impl=SimpleNamespace(value="paged_adamw_8bit")))
    assert paged["optim"] == "paged_adamw_8bit"


def test_hf_kwargs_apply_documented_defaults_for_unset_fields() -> None:
    # None weight_decay / scheduler / warmup fall back to HF's documented defaults, never crash or drop.
    kw = hf_training_arguments_optimizer_kwargs(
        _spec(weight_decay=None, lr_scheduler=None, warmup_ratio=None)
    )
    assert kw["weight_decay"] == 0.0
    assert kw["lr_scheduler_type"] == "linear"
    assert kw["warmup_ratio"] == 0.0


def test_hf_kwargs_accept_the_real_optimizer_spec_contract() -> None:
    # field-name drift guard: the helper reads a genuine sealed OptimizerSpec, not just a namespace.
    from corpus_studio.platform.contracts import OptimizerSpec

    kw = hf_training_arguments_optimizer_kwargs(
        OptimizerSpec(impl="paged_adamw_8bit", learning_rate=1e-4)
    )
    assert kw["optim"] == "paged_adamw_8bit" and kw["learning_rate"] == 1e-4
    assert set(kw) == _KEYS
