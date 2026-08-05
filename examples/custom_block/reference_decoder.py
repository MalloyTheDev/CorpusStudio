"""A reference custom-block for CorpusStudio's mode-3 "your own model code" path.

Copy this as a starting point for your OWN model implementation (the only not-borrowed mode - a family
or a composed-on-standard-blocks design needs no custom code). To be admitted by ``vet-model-code`` a
custom block must:

1. subclass a recognized model base (torch ``nn.Module`` / HF ``PreTrainedModel`` / ``CustomDecoderBase``),
2. define ``__init__(self, config)`` that builds the model from an architecture config, and
3. define ``forward(...)`` that accepts ``input_ids`` and returns ``loss`` + ``logits``.

The static screen (``vet-model-code``) rejects the obvious dangerous surface (process/network/reflective
escapes) and pins the exact bytes. It is NOT a safety proof: your code runs only inside the gated worker
sandbox, and admission stays human-gated. This path never uses ``trust_remote_code``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReferenceDecoderForCausalLM(nn.Module):
    """A minimal, honestly-simple causal-LM decoder built from a from-scratch architecture config."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                batch_first=True,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # attention_mask (padding) is accepted for interface compatibility with HF-style trainers; this
        # minimal reference trains on packed sequences and relies on the causal mask alone.
        seq_len = input_ids.size(1)
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device), diagonal=1
        )
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels)
        return {"loss": loss, "logits": logits}
