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

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corpus_studio.platform.common import Ref
from corpus_studio.platform.contracts import (
    PretrainingExecutionEvidence,
    PretrainingSuccessEvidence,
    ResolvedPretrainingExecutionConfiguration,
)
from corpus_studio.platform.enums import StageMarker
from corpus_studio.training.corpus_packing import pack_documents
from corpus_studio.training.pretraining_evidence import (
    PretrainingExecutionTracker,
    register_full_model_gradient_hooks,
)
from corpus_studio.training.trainer import (
    capture_adapter_export_state,
    capture_trainable_state,
)

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
    # Sealed proof that a real full-parameter optimization happened (optimizer + per-step loss + gradient
    # coverage + before != after model bytes). None only for a run that predates evidence capture.
    execution_evidence: PretrainingSuccessEvidence | None = None


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
    tokenizer_source = execution.tokenizer_source
    if tokenizer_source.mode == "freeze" and tokenizer_source.tokenizer_location is None:
        # freeze with no location = the checkpoint's tokenizer (continued init) - refuse cleanly rather
        # than let the load path do Path(None).
        raise PretrainingError(
            "a freeze tokenizer without a location comes from a continued checkpoint (not yet supported)"
        )


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


def _verify_pinned_tokenizer_content(location: Path, expected_sha256: str) -> None:
    """PURE fail-closed integrity gate for an imported tokenizer. The seal REQUIRES a
    tokenizer_content_sha256 for an import (TokenizerSourceSpec validator), and a fast tokenizer's canonical
    bytes live in tokenizer.json. FAIL CLOSED when there is no tokenizer.json to verify against - a missing
    file must never silently skip the integrity pin (a swapped tokenizer redefines the whole vocab) - and on
    any content mismatch. A slow/SentencePiece-only tokenizer is refused on purpose: it cannot be
    content-verified by this digest."""
    tokenizer_json = location / "tokenizer.json" if location.is_dir() else location
    if not tokenizer_json.is_file():
        raise PretrainingError(
            "cannot verify the sealed tokenizer_content_sha256: no tokenizer.json at the pinned location "
            f"{location} - an imported tokenizer must ship a tokenizer.json to be content-pinned."
        )
    observed = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise PretrainingError(
            "the imported tokenizer content does not match the sealed tokenizer_content_sha256"
        )


def _load_imported_tokenizer(tokenizer_source: Any) -> Any:  # pragma: no cover
    """Load a pre-built tokenizer pinned by location + content digest (the bring-your-own path)."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    location = Path(tokenizer_source.tokenizer_location)
    tokenizer = AutoTokenizer.from_pretrained(str(location))
    _verify_pinned_tokenizer_content(location, tokenizer_source.tokenizer_content_sha256)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """A stable semantic digest of the from-scratch model architecture config (order-independent)."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _build_success_evidence(  # pragma: no cover - filesystem/torch integration; proven by a run
    out: Path, execution: PretrainingExecutionEvidence
) -> PretrainingSuccessEvidence:
    """Parse-validate the saved model artifacts and seal the pretraining success evidence. This is
    stronger than an existence check - the model.safetensors must open as a non-empty tensor archive and
    the config as JSON - so the success flags are earned. The full reload-and-compare-to-trained-state
    verification (SFT's runner reload-verify) is the slice-2b runner-admission step."""
    from safetensors import safe_open  # noqa: PLC0415

    model_safetensors = out / "model.safetensors"
    config_json = out / "config.json"
    if not model_safetensors.is_file():
        raise PretrainingError("training completed without a saved model.safetensors")
    if not config_json.is_file():
        raise PretrainingError("training completed without a saved config.json")
    try:
        with safe_open(str(model_safetensors), framework="pt") as handle:
            if not list(handle.keys()):
                raise PretrainingError("the saved model.safetensors contains no tensors")
    except PretrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any safetensors parse failure to fail-closed
        raise PretrainingError(
            f"the saved model.safetensors is not a valid safetensors archive: {exc}"
        ) from exc
    try:
        json.loads(config_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PretrainingError(f"the saved config.json is not valid JSON: {exc}") from exc
    return PretrainingSuccessEvidence(
        execution=execution,
        output_path_verified=True,
        model_bytes_verified=True,
        artifact_integrity_verified=True,
        model_safetensors_sha256=hashlib.sha256(model_safetensors.read_bytes()).hexdigest(),
        model_config_sha256=hashlib.sha256(config_json.read_bytes()).hexdigest(),
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
        TrainerCallback,
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

    if execution.tokenizer_source.mode == "train":
        tokenizer = _train_bpe_tokenizer(documents, execution.tokenizer_source)
        tokenizer_source = "trained"
    else:  # import / freeze - bring your own pinned tokenizer
        tokenizer = _load_imported_tokenizer(execution.tokenizer_source)
        tokenizer_source = "imported"
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    arch = load_architecture_config(execution.init.architecture_ref, corpus_root=corpus_root)
    model_type = arch.pop("model_type")
    arch["vocab_size"] = int(tokenizer.vocab_size)  # from-scratch: size the embedding to THIS tokenizer
    model_config_semantic_sha256 = _canonical_config_sha256({"model_type": model_type, **arch})
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

    # Execution-evidence capture: register gradient-observation hooks and snapshot the trainable state
    # BEFORE training, so the finalized evidence proves a real optimizer stepped the complete inventory,
    # gradients materialized on trainable tensors, and both the trainable state and the exported model
    # tensors changed. Full-parameter throughout (no adapter, no nf4 / master-dtype enforcement).
    gradient_tracker = register_full_model_gradient_hooks(model, torch)
    # The sealed schedule is exactly one of max_steps XOR num_train_epochs. In step mode the ceiling is
    # the sealed max_steps; in epoch mode it is the Trainer's data-derived total-step plan for the sealed
    # epochs over the sealed packed corpus, bound AFTER trainer.train() computes it (deterministic in the
    # sealed inputs). A hardcoded `or 1` silently collapsed an epoch plan to a single optimizer step.
    sealed_max_steps = execution.schedule.max_steps
    epoch_mode = sealed_max_steps is None
    sealed_epochs = execution.schedule.num_train_epochs
    if epoch_mode and sealed_epochs is None:  # pragma: no cover - contract guarantees exactly-one-of
        raise PretrainingError("the sealed schedule set neither max_steps nor num_train_epochs")
    execution_tracker = PretrainingExecutionTracker(
        expected_steps=sealed_max_steps if sealed_max_steps is not None else 0,
        gradients=gradient_tracker,
    )

    def _trainable_mapping() -> dict[str, Any]:
        return {n: p for n, p in model.named_parameters() if p.requires_grad}

    before_trainable = capture_trainable_state(model, torch, stage=StageMarker.optimizer_step)
    before_export = capture_adapter_export_state(
        _trainable_mapping(), torch, stage=StageMarker.optimizer_step
    )

    class _EvidenceCallback(TrainerCallback):  # type: ignore[misc]
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            execution_tracker.on_train_begin(kwargs.get("optimizer"))

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            execution_tracker.on_step_end(int(state.global_step), kwargs.get("optimizer"))

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Mapping[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            execution_tracker.on_log(int(state.global_step), logs)

    optimizer = execution.optimizer
    arguments = TrainingArguments(
        output_dir=str(out),
        # max_steps=-1 hands control to num_train_epochs (HF's epoch mode); a sealed max_steps overrides
        # epochs. Either way the sealed schedule - not a hardcoded 1 - drives how long training runs.
        max_steps=sealed_max_steps if sealed_max_steps is not None else -1,
        num_train_epochs=(
            float(sealed_epochs) if epoch_mode and sealed_epochs is not None else 1.0
        ),
        per_device_train_batch_size=execution.batching.micro_batch_size,
        gradient_accumulation_steps=execution.batching.fallback_grad_accumulation_steps or 1,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay or 0.0,
        max_grad_norm=optimizer.max_grad_norm,
        lr_scheduler_type=(optimizer.lr_scheduler or "linear"),
        warmup_ratio=optimizer.warmup_ratio or 0.0,
        # Honor the FULL sealed optimizer - impl (adamw_torch vs paged_adamw_8bit) + betas + epsilon - not
        # just the lr/decay: silently defaulting HF's optim would drop the sealed choice (the same
        # trainer-field-filtering the SFT path avoids via optim=impl.value).
        optim=optimizer.impl.value,
        adam_beta1=optimizer.adam_beta1,
        adam_beta2=optimizer.adam_beta2,
        adam_epsilon=optimizer.adam_epsilon,
        seed=execution.seed,
        data_seed=execution.data_seed,
        logging_steps=1,
        logging_nan_inf_filter=False,  # log one loss per step, even a non-finite one, for the evidence
        save_strategy="no",
        report_to=[],
        use_cpu=cpu_toy,
        gradient_checkpointing=execution.gradient_checkpointing and not cpu_toy,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=_collate,
        callbacks=[_EvidenceCallback()],
    )
    train_output = trainer.train()
    steps = int(getattr(train_output, "global_step", 0) or 0)
    if epoch_mode:
        # Now that the Trainer has computed its total optimizer-step plan for the sealed epochs, bind it
        # as the finalize ceiling (the placeholder above was 0). Deterministic in the sealed inputs.
        planned_steps = int(getattr(trainer.state, "max_steps", 0) or 0)
        if planned_steps < 1:
            raise PretrainingError(
                "epoch-scheduled pretraining computed a non-positive optimizer-step plan"
            )
        execution_tracker.expected_steps = planned_steps

    # After training: verify the trainable inventory is unchanged, snapshot the trained state, and seal
    # the execution evidence BEFORE saving. Finalizing first keeps a tampered export from being admitted;
    # the saved model.safetensors / config.json are then hashed into the success evidence.
    gradient_tracker.verify_model_inventory(model, stage=StageMarker.optimizer_step)
    after_trainable = capture_trainable_state(model, torch, stage=StageMarker.optimizer_step)
    after_export = capture_adapter_export_state(
        _trainable_mapping(), torch, stage=StageMarker.optimizer_step
    )
    execution_detail = execution_tracker.finalize(
        steps=steps,
        before=before_trainable,
        after=after_trainable,
        before_export=before_export,
        after_export=after_export,
        model_config_semantic_sha256=model_config_semantic_sha256,
    )

    # Save the full model as a SINGLE safetensors file (no size-based sharding) so the single-file
    # success-evidence and the supervisor's independent reload-verify hold for large from-scratch /
    # continued models too - safetensors serves a multi-GB single file via mmap. save_model delegates to
    # save_pretrained; we call it directly only to pass an effectively-unbounded max_shard_size.
    trainer.model.save_pretrained(str(out), safe_serialization=True, max_shard_size="1000GB")
    tokenizer.save_pretrained(str(out))
    success_evidence = _build_success_evidence(out, execution_detail)

    metrics = getattr(train_output, "metrics", {}) or {}
    return PretrainResult(
        output_dir=str(out),
        cpu_toy=cpu_toy,
        steps=steps,
        final_loss=metrics.get("train_loss"),
        vocab_size=int(tokenizer.vocab_size),
        num_blocks=packed.coverage.num_blocks,
        coverage_ratio=packed.coverage.coverage_ratio,
        tokenizer_source=tokenizer_source,
        execution_evidence=success_evidence,
    )
