"""Reference resumable training loop for the checkpoint/resume equivalence proof (#440).

Run as a SUBPROCESS in three modes so every run - including the resume - is a genuinely fresh
process, which is the honest test of cross-process determinism:

    python _checkpoint_reference.py uninterrupted --workdir DIR --k K --n N
    python _checkpoint_reference.py firsthalf     --workdir DIR --k K --n N
    python _checkpoint_reference.py resume        --workdir DIR --k K --n N

``uninterrupted`` trains N steps and writes ``uninterrupted.pt`` (final params) + ``losses_full.json``.
``firsthalf`` trains K steps then checkpoints via the real engine. ``resume`` restores that checkpoint
and trains the remaining N-K steps, writing ``resumed.pt`` + ``losses_tail.json``. The loop is
deterministic (fixed seed, single thread, restored RNG), so ``resumed.pt`` must match
``uninterrupted.pt`` and the resumed step numbering must continue from K+1.

This module carries a top-level torch import ON PURPOSE - it is executed only under a torch-bearing
interpreter (the test that drives it skips when torch is absent), never imported by the engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from corpus_studio.platform.checkpoint import bound_identities_from_plan
from corpus_studio.platform.contracts import CheckpointResumeRequest
from corpus_studio.platform.runners import demo_training_plan
from corpus_studio.training import checkpoint_io as cio

SEED = 1234
PLAN = demo_training_plan(plan_id="ckpt-integration")
SOURCE_RUN_ID = "run-integration-parent"
CHECKPOINT_ID = "run-integration-parent-ckpt"


def _deterministic() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _build_model() -> torch.nn.Module:
    # Dropout makes the forward RNG-dependent, so a correct resume MUST restore the RNG streams.
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(16, 4),
    )
    model.train()
    return model


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999))


def _build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)


def _batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    # A fixed per-step batch derived from the step index: deterministic and order-stable, so the
    # "sampler cursor" is just the number of consumed batches.
    generator = torch.Generator().manual_seed(10_000 + step)
    inputs = torch.randn(4, 8, generator=generator)
    targets = torch.randint(0, 4, (4,), generator=generator)
    return inputs, targets


def _train_range(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    start_step: int,
    end_step: int,
) -> list[dict[str, float]]:
    loss_fn = torch.nn.CrossEntropyLoss()
    losses: list[dict[str, float]] = []
    for step in range(start_step, end_step + 1):
        inputs, targets = _batch(step)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append({"optimizer_step": step, "loss": float(loss.detach())})
    return losses


def _final_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _run_uninterrupted(workdir: Path, n: int) -> None:
    _deterministic()
    model = _build_model()
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    losses = _train_range(model, optimizer, scheduler, 1, n)
    torch.save(_final_state(model), workdir / "uninterrupted.pt")
    (workdir / "losses_full.json").write_text(json.dumps(losses), encoding="utf-8")


def _run_firsthalf(workdir: Path, k: int, n: int) -> None:
    _deterministic()
    model = _build_model()
    optimizer = _build_optimizer(model)
    scheduler = _build_scheduler(optimizer)
    losses = _train_range(model, optimizer, scheduler, 1, k)
    cio.save_checkpoint(
        torch_module=torch,
        final_dir=workdir / "checkpoint",
        adapter_state={name: param.detach().clone() for name, param in model.named_parameters()},
        optimizer=optimizer,
        lr_scheduler=scheduler,
        position=cio.StepPosition(
            epoch=float(k),
            global_optimizer_step=k,
            gradient_accumulation_steps=1,
            consumed_microsteps=k,
        ),
        bound=bound_identities_from_plan(PLAN),
        source_run_id=SOURCE_RUN_ID,
        checkpoint_id=CHECKPOINT_ID,
        created_at="2026-07-15T00:00:00+00:00",
        rng_state=cio.capture_rng_state(torch),
        sampler_state={"cursor": k},
    )
    (workdir / "losses_first.json").write_text(json.dumps(losses), encoding="utf-8")


def _run_resume(workdir: Path, n: int) -> None:
    _deterministic()
    model = _build_model()  # fresh weights; restore overwrites them
    from corpus_studio.platform.checkpoint import verify_checkpoint_integrity

    checkpoint_dir = workdir / "checkpoint"
    manifest = verify_checkpoint_integrity(checkpoint_dir)
    request = CheckpointResumeRequest(
        checkpoint_id=manifest.checkpoint_id,
        checkpoint_manifest_hash=manifest.checkpoint_manifest_hash,
        checkpoint_dir=str(checkpoint_dir),
    )

    def _apply_adapter(m: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in m.named_parameters():
                param.copy_(state[name])

    result = cio.restore_checkpoint(
        torch_module=torch,
        request=request,
        plan=PLAN,
        model=model,
        apply_adapter_state=_apply_adapter,
        build_optimizer=lambda: _build_optimizer(model),
        build_lr_scheduler=_build_scheduler,
    )
    start = result.resumed_from_global_step + 1  # the exact next optimizer step
    losses = _train_range(model, result.optimizer, result.lr_scheduler, start, n)
    torch.save(_final_state(model), workdir / "resumed.pt")
    (workdir / "losses_tail.json").write_text(json.dumps(losses), encoding="utf-8")
    (workdir / "resume_meta.json").write_text(
        json.dumps({"cursor": result.sampler_state, "start_step": start}), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["uninterrupted", "firsthalf", "resume"])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "uninterrupted":
        _run_uninterrupted(args.workdir, args.n)
    elif args.mode == "firsthalf":
        _run_firsthalf(args.workdir, args.k, args.n)
    else:
        _run_resume(args.workdir, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
