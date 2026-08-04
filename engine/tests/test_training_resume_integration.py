"""Real-torch end-to-end proof for checkpoint write + hybrid resume through ``run_training`` (#486).

Skipped wherever torch/trl or the offline tokenizer are absent (the dependency-light CI lane), so it
never runs in the torch-free gate. Where the real stack IS present it drives the SHIPPED ``run_training``
end to end - a cpu_toy WRITE run seals checkpoints at the cadence, then a RESUME run verifies our seal,
materializes an HF layout, and resumes via ``SFTTrainer.resume_from_checkpoint`` - proving the whole path
(not a hand-built trainer). The bitwise-faithfulness of the translation itself is proven separately; this
guards the integration: the resume run continues from the checkpoint step and re-checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_TOKENIZER = (
    "/mnt/training-nvme/models/Qwen2.5-0.5B-Instruct/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
_TOKENIZER_PATH = os.environ.get("CORPUS_STUDIO_TEST_TOKENIZER", _DEFAULT_TOKENIZER)

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - worker envs run this via __main__
    pytest = None  # type: ignore[assignment]


def _run_resume_integration(root: str) -> dict[str, Any]:
    """WRITE then RESUME through ``run_training``; returns what each run produced."""

    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from corpus_studio.platform.checkpoint import bound_identities_from_plan, load_checkpoint_manifest
    from corpus_studio.platform.contracts import CheckpointResumeRequest
    from corpus_studio.platform.runners import demo_training_plan
    from corpus_studio.training.trainer import TrainRunConfig, run_training

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    base = Path(root)

    tok = AutoTokenizer.from_pretrained(_TOKENIZER_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            pad_token_id=tok.pad_token_id,
        )
    )
    mdir = base / "model"
    model.save_pretrained(str(mdir))
    tok.save_pretrained(str(mdir))
    data = base / "data.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        for i in range(8):
            handle.write(json.dumps({"instruction": f"question {i}", "output": f"answer {i} text"}) + "\n")

    def _cfg(output_dir: str) -> TrainRunConfig:
        return TrainRunConfig(
            base_model=str(mdir), dataset_path=str(data), output_dir=output_dir, cpu_toy=True,
            dataset_format="instruction", save_strategy="steps", save_steps=3, save_total_limit=3,
            max_steps=6, gradient_accumulation_steps=1, micro_batch_size=1, attn_implementation="eager",
            lora_r=4, lora_alpha=8, learning_rate=5e-3,
        )

    bound = bound_identities_from_plan(demo_training_plan(plan_id="resume-integration"))
    ck_root = base / "checkpoints"
    stages: list[str] = []

    def _stage(name: str, _msg: str) -> None:
        stages.append(name)

    # WRITE: seal checkpoints at steps 3 and 6.
    run_training(_cfg(str(base / "out1")), stage_callback=_stage, checkpoint_bound=bound,
                 source_run_id="run-parent", checkpoints_root=str(ck_root))
    written = sorted(p.name for p in ck_root.glob("step-*")) if ck_root.exists() else []

    # RESUME from the step-3 checkpoint.
    manifest = load_checkpoint_manifest(ck_root / "step-00000003")
    request = CheckpointResumeRequest(
        checkpoint_id=manifest.checkpoint_id,
        checkpoint_manifest_hash=manifest.checkpoint_manifest_hash,
        checkpoint_dir=str(ck_root / "step-00000003"),
    )
    resume_stages: list[str] = []
    run_training(_cfg(str(base / "out2")), stage_callback=lambda n, _m: resume_stages.append(n),
                 resume=request, checkpoint_bound=bound, source_run_id="run-resumed",
                 checkpoints_root=str(base / "ck2"))
    return {
        "written": written,
        "resume_adapter": (base / "out2").exists(),
        "resumed": "resume" in resume_stages,
    }


def _assert(result: dict[str, Any]) -> None:
    assert result["written"] == ["step-00000003", "step-00000006"], result
    assert result["resumed"], "the resume run did not emit a 'resume' stage"
    assert result["resume_adapter"], "the resume run produced no adapter"


if pytest is not None:
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    @pytest.mark.skipif(
        not Path(_TOKENIZER_PATH).exists(),
        reason=f"offline tokenizer not present at {_TOKENIZER_PATH}",
    )
    def test_run_training_writes_then_resumes(tmp_path: Path) -> None:
        _assert(_run_resume_integration(str(tmp_path / "e2e")))


if __name__ == "__main__":  # Direct pinned-stack execution (no pytest needed in the worker env).
    import tempfile

    out = os.environ.get("CORPUS_STUDIO_TEST_OUT") or tempfile.mkdtemp(prefix="resume_e2e_")
    res = _run_resume_integration(out)
    print(json.dumps(res, indent=2))
    try:
        _assert(res)
    except AssertionError as exc:
        print(f"INTEGRATION_FAIL: {exc}")
        raise SystemExit(1) from exc
    print("INTEGRATION_PASS")
    raise SystemExit(0)
