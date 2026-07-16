"""Real pinned-stack integration proof for the training_step token observer (the v7 fix).

Skipped wherever the training stack is absent (the torch-free CI gate), so it never runs in the
dependency-light lane. Where the PINNED stack IS present (trl 1.8.0 / transformers 5.13.1 /
torch 2.11.0+cu128 / datasets 5.0.0) it builds a tiny Llama FROM CONFIG (no network) plus a local
tokenizer, runs a real ``SFTTrainer`` for a few CPU steps, and proves:

  * the SHIPPED ``_TokenAccumulator`` + ``count_batch_tokens`` observe POSITIVE non-padding AND
    supervised token counts at EVERY optimizer step, when fed ``inputs`` at ``training_step`` - the
    trainer's own consumption boundary; and
  * the v6 mechanism (reassigning ``collate_fn`` on the dataloader ``get_train_dataloader`` returns)
    observes ZERO batches on this exact stack, because the accelerate-prepared shard does not honor the
    reassignment. That is the root cause of the v6 ``tokens_per_sec == 0.0`` gap, and it is why the fix
    had to move to ``training_step``.

This is a CPU-only, no-network, no-GPU test. Run it against the pinned stack directly with, e.g.::

    PYTHONPATH=engine <pinned-env>/bin/python engine/tests/test_token_observer_integration.py

which drives the same ``_run_observer_integration`` the pytest case asserts on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# A local tokenizer keeps the test offline. Default to the host's Qwen2.5-0.5B assets; override via env.
_DEFAULT_TOKENIZER = (
    "/mnt/training-nvme/models/Qwen2.5-0.5B-Instruct/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
_TOKENIZER_PATH = os.environ.get("CORPUS_STUDIO_TEST_TOKENIZER", _DEFAULT_TOKENIZER)


def _run_observer_integration(output_dir: str, tokenizer_path: str) -> dict[str, Any]:
    """Train a tiny model on the real stack; return what each boundary observed.

    Returns a dict with ``per_step`` (list of ``(nonpadding, supervised, observed_microbatches)`` from
    the SHIPPED accumulator, one entry per optimizer step) and ``collate_fired`` (how many batches the
    v6 collate-wrap saw - expected 0 on the pinned stack)."""

    from datasets import Dataset
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    # SHIPPED code under test - the exact helpers the worker uses for token accounting.
    from corpus_studio.training.trainer import _TokenAccumulator, count_batch_tokens

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # A genuinely tiny Llama, constructed from config so nothing is downloaded.
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=tok.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)

    rows = [
        {"messages": [{"role": "user", "content": "hello there"},
                      {"role": "assistant", "content": "hi, how can I help you today?"}]},
        {"messages": [{"role": "user", "content": "what is 2+2"},
                      {"role": "assistant", "content": "2 + 2 equals 4."}]},
        {"messages": [{"role": "user", "content": "name a color"},
                      {"role": "assistant", "content": "blue is a nice color."}]},
        {"messages": [{"role": "user", "content": "say bye"},
                      {"role": "assistant", "content": "goodbye, take care!"}]},
    ]
    ds = Dataset.from_list(rows)

    accumulator = _TokenAccumulator()
    per_step: list[tuple[int, int, int]] = []
    collate_fired = {"n": 0}

    class _FlushOnStepEnd(TrainerCallback):
        # Mirrors the shipped run_training wiring: flush the accumulator once per optimizer step.
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            per_step.append(accumulator.flush())

    class _ObservingTrainer(SFTTrainer):  # type: ignore[misc, valid-type]
        # (B) the fix: observe the collated batch the trainer actually consumes.
        def training_step(self, *args: Any, **kwargs: Any) -> Any:
            inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
            from collections.abc import Mapping

            if isinstance(inputs, Mapping):
                accumulator.observe(inputs)
            return super().training_step(*args, **kwargs)

        # (A) the v6 mechanism: reassign collate_fn on the returned (accelerate-prepared) loader.
        def get_train_dataloader(self) -> Any:
            loader = super().get_train_dataloader()
            try:
                inner = loader.collate_fn

                def _wrapped(features: Any, _inner: Any = inner) -> Any:
                    batch = _inner(features)
                    collate_fired["n"] += 1
                    count_batch_tokens(batch)
                    return batch

                loader.collate_fn = _wrapped
            except Exception:  # noqa: BLE001 - matches the v6 best-effort install
                pass
            return loader

    args = SFTConfig(
        output_dir=output_dir,
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        max_length=256,
    )
    trainer = _ObservingTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok,
        callbacks=[_FlushOnStepEnd()],
    )
    trainer.train()
    return {"per_step": per_step, "collate_fired": collate_fired["n"]}


def _assert_integration(result: dict[str, Any]) -> None:
    """Shared assertions so the pytest case and the __main__ driver check the identical invariants."""
    per_step = result["per_step"]
    # One flushed record per optimizer step (max_steps=3, ga=1 -> exactly 3).
    assert len(per_step) == 3, per_step
    for nonpadding, supervised, observed in per_step:
        # The observer FIRED and saw real tokens: never the v6 "captured nothing -> 0.0".
        assert observed == 1, per_step
        assert nonpadding > 0, per_step
        assert supervised > 0, per_step
        # TRL default SFT labels the full non-padding sequence, so supervised == non-padding here.
        assert supervised == nonpadding, per_step
    # Root-cause guard: the v6 collate-wrap never fires on this pinned stack (why the fix moved to
    # training_step). If a future stack DID honor it this assertion would flag the change for review.
    assert result["collate_fired"] == 0, result


# The pytest surface exists only when pytest is importable. Guarded so the __main__ driver runs under a
# worker env that has the pinned stack but no pytest, while CI collection still skips the torch-free lane.
try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - worker envs run this via __main__
    pytest = None  # type: ignore[assignment]

if pytest is not None:
    # Module is inert in the dependency-light lane: skip collection unless the real stack is present.
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")

    @pytest.mark.skipif(
        not Path(_TOKENIZER_PATH).exists(),
        reason=f"offline tokenizer not present at {_TOKENIZER_PATH}",
    )
    def test_training_step_observer_counts_positive_tokens_on_pinned_stack(tmp_path: Path) -> None:
        _assert_integration(_run_observer_integration(str(tmp_path / "out"), _TOKENIZER_PATH))


if __name__ == "__main__":  # Direct pinned-stack execution (no pytest needed in the worker env).
    import json
    import tempfile

    out = os.environ.get("CORPUS_STUDIO_TEST_OUT") or tempfile.mkdtemp(prefix="token_observer_")
    res = _run_observer_integration(out, _TOKENIZER_PATH)
    print(json.dumps(res, indent=2))
    try:
        _assert_integration(res)
    except AssertionError as exc:
        print(f"INTEGRATION_FAIL: {exc}")
        raise SystemExit(1) from exc
    print("INTEGRATION_PASS")
    raise SystemExit(0)
