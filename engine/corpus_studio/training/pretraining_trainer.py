"""``run_pretraining`` - the from-scratch full-parameter causal-LM pretraining worker (S3b-1a).

It consumes the sealed :class:`ResolvedPretrainingExecutionConfiguration` DIRECTLY (faithful by
construction - it cannot drop a sealed field it never copied). It mirrors ``run_training``'s lifecycle
(corpus -> tokenizer -> ``from_config`` random init -> pack -> train -> save) but is full-parameter plain
HF ``Trainer`` (not TRL ``SFTTrainer``), over PACKED sequences (``corpus_packing``).

Torch + tokenizers are imported LAZILY, so the torch-free helpers (corpus reading, the architecture-config
load, the unsupported-mode refusals, the result model) are unit-tested in the base gate; the training loop
and BPE tokenizer trainer are ``# pragma: no cover`` and proven by a CPU run. This slice is UNROUTED: the
runner still refuses the sealed pretraining path, so nothing runs it in production and no worker wheel is
needed until the (gated) Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import ResolvedPretrainingExecutionConfiguration
from corpus_studio.training.corpus_packing import pack_documents

# Keys create-model writes that are provenance, not transformers config fields - stripped before build.
_NON_HF_CONFIG_KEYS = ("architectures", "corpus_studio_name", "corpus_studio_needs_custom_code")


class PretrainingError(RuntimeError):
    """A pretraining run the worker cannot honor (fail-closed, a clean typed error)."""


class PretrainResult(BaseModel):
    """The outcome of a pretraining run (mirrors TrainResult; no adapter - a full model was trained)."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    cpu_toy: bool
    steps: int = Field(default=0, ge=0)
    final_loss: float | None = None
    vocab_size: int = Field(ge=1)
    num_blocks: int = Field(ge=0)
    coverage_ratio: float = Field(ge=0, le=1)
    tokenizer_source: str


def load_corpus_documents(
    shard_locations: list[str], *, corpus_root: str | Path = ".", text_field: str = "text"
) -> list[str]:
    """Read the ``text`` field of every row across the corpus shards (JSONL). Torch-free + unit-tested;
    the memory-bounded streaming version is the S3b-1c loader."""
    documents: list[str] = []
    for location in shard_locations:
        path = Path(corpus_root) / location
        try:
            handle = path.open(encoding="utf-8")
        except OSError as exc:
            raise PretrainingError(f"cannot read corpus shard {path}: {exc}") from exc
        with handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except ValueError as exc:
                    raise PretrainingError(f"corpus shard {path} has a non-JSON row: {exc}") from exc
                text = row.get(text_field) if isinstance(row, dict) else None
                if isinstance(text, str) and text:
                    documents.append(text)
    return documents


def load_architecture_config(
    architecture_ref: Ref | None, *, corpus_root: str | Path = "."
) -> dict[str, Any]:
    """Load + sanitize the from-scratch architecture config (the create-model JSON) into transformers
    config kwargs. Refuses a custom_decoder here (the sandbox is a separate, gated slice)."""
    if architecture_ref is None:
        raise PretrainingError("random-init pretraining requires an architecture_ref")
    path = Path(architecture_ref.id)
    if not path.is_absolute():
        path = Path(corpus_root) / architecture_ref.id
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PretrainingError(f"cannot read architecture config {path}: {exc}") from exc
    if not isinstance(config, dict) or "model_type" not in config:
        raise PretrainingError("architecture config must be a JSON object with a model_type")
    if config.get("model_type") == "custom_decoder" or config.get("corpus_studio_needs_custom_code"):
        raise PretrainingError(
            "a custom_decoder architecture needs the gated custom-block worker sandbox (not built); refuse"
        )
    return {k: v for k, v in config.items() if k not in _NON_HF_CONFIG_KEYS}


def _refuse_unsupported(execution: ResolvedPretrainingExecutionConfiguration) -> None:
    """Fail-closed on the modes this first worker slice does not implement (torch-free guard)."""
    init = execution.init
    if init.mode == "continued":
        raise PretrainingError("continued pretraining (checkpoint resume) is a later worker slice")
    if init.custom_code is not None:
        raise PretrainingError("custom-block execution needs the gated worker sandbox (not built); refuse")
    if execution.tokenizer_source.mode in ("import", "freeze"):
        raise PretrainingError("the import/freeze tokenizer path is a later slice (S3b-1a inc 3b)")


def _train_bpe_tokenizer(documents: list[str], tokenizer_source: Any) -> Any:  # pragma: no cover
    """Train a byte-level BPE tokenizer from the corpus (the truly-from-scratch tokenizer path)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    specials = list(tokenizer_source.special_tokens or ["<unk>", "<bos>", "<eos>", "<pad>"])

    def _pick(*candidates: str) -> str | None:
        return next((token for token in candidates if token in specials), None)

    unk = _pick("<unk>", "[UNK]") or specials[0]
    tokenizer = Tokenizer(models.BPE(unk_token=unk))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        documents,
        trainers.BpeTrainer(
            vocab_size=tokenizer_source.vocab_size,
            special_tokens=specials,
            min_frequency=tokenizer_source.min_frequency or 1,
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk,
        bos_token=_pick("<bos>", "<s>"),
        eos_token=_pick("<eos>", "</s>"),
        pad_token=_pick("<pad>", "<eos>", "</s>"),
    )


def run_pretraining(  # pragma: no cover - torch/tokenizers integration; proven by a CPU run
    execution: ResolvedPretrainingExecutionConfiguration,
    *,
    corpus_root: str | Path = ".",
    output_dir: str | None = None,
) -> PretrainResult:
    """Train a from-scratch full-parameter causal-LM per the sealed config. cpu_toy runs on CPU for the
    plumbing proof; a real run is Phase 2 (gated)."""
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _refuse_unsupported(execution)
    cpu_toy = execution.runtime_mode == "cpu_toy"
    out = Path(output_dir or execution.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(execution.seed)

    documents = load_corpus_documents(
        [shard.location for shard in execution.data.shards], corpus_root=corpus_root
    )
    if not documents:
        raise PretrainingError("no corpus documents found in the sealed shards")

    tokenizer = _train_bpe_tokenizer(documents, execution.tokenizer_source)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    arch = load_architecture_config(execution.init.architecture_ref, corpus_root=corpus_root)
    model_type = arch.pop("model_type")
    arch["vocab_size"] = int(tokenizer.vocab_size)  # from-scratch: size the embedding to THIS tokenizer
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **arch))

    packed = pack_documents(
        [tokenizer.encode(document) for document in documents],
        sequence_len=execution.sequence.max_sequence_len,
        eos_id=int(eos_id),
    )
    if not packed.blocks:
        raise PretrainingError("the corpus is too small to fill one block at the sealed sequence length")
    dataset = Dataset.from_dict({"input_ids": packed.blocks})

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}  # causal LM shifts internally

    optimizer = execution.optimizer
    arguments = TrainingArguments(
        output_dir=str(out),
        max_steps=execution.schedule.max_steps or 1,
        per_device_train_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay or 0.0,
        max_grad_norm=optimizer.max_grad_norm,
        lr_scheduler_type=(optimizer.lr_scheduler or "linear"),
        warmup_ratio=optimizer.warmup_ratio or 0.0,
        seed=execution.seed,
        data_seed=execution.data_seed,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=cpu_toy,
        gradient_checkpointing=execution.gradient_checkpointing and not cpu_toy,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=_collate)
    train_output = trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    metrics = getattr(train_output, "metrics", {}) or {}
    return PretrainResult(
        output_dir=str(out),
        cpu_toy=cpu_toy,
        steps=int(getattr(train_output, "global_step", 0) or 0),
        final_loss=metrics.get("train_loss"),
        vocab_size=int(tokenizer.vocab_size),
        num_blocks=packed.coverage.num_blocks,
        coverage_ratio=packed.coverage.coverage_ratio,
        tokenizer_source="trained",
    )
