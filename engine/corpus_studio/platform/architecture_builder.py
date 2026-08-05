"""Model architecture builder (S3b front-end) - the torch-free producer of the ``architecture_ref`` a
from-scratch pretraining run's ``from_config`` init consumes. Two HONEST modes:

* **base on a family** (``build_from_family``) - configure a KNOWN architecture (Llama, Mistral, Qwen2,
  Gemma, GPT-2, GPT-NeoX). The result is *based on* that family - it is NOT "your own model".
* **compose** (``build_composed``) - YOU pick the building blocks (positions, MLP shape, norm, attention
  bias, grouped-query attention) into a novel configuration + your own name. That design is YOURS; it is
  realized on the matching reference block IMPLEMENTATION (the same blocks Llama/Mistral share). If your
  combination matches NO reference implementation, ``needs_custom_code`` is set: a truly-novel design
  needs the custom-block path (real model code), not a config.

No model is instantiated here (no torch / transformers) - only a config dict + an honest STATIC parameter
ESTIMATE (validated against reference models; a measured count is the worker's job).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

_HEAD_DIM = 64  # heads are sized at 64-d (heads = hidden // 64) unless given explicitly.


class ArchitectureBuilderError(ValueError):
    """A from-scratch architecture request the builder cannot honor (fail-closed, a clean typed error)."""


@dataclass(frozen=True)
class ArchitectureTraits:
    """The BUILDING BLOCKS of a decoder-LM architecture - the design choices, independent of dimensions.
    A named family is a preset of these (honestly "based on" that family); composing your own means
    choosing them yourself. ``intermediate_ratio`` is only a default sizing (explicit dims override it)."""

    positions: Literal["rope", "learned"]
    gated_mlp: bool
    activation: str
    norm: Literal["rmsnorm", "layernorm"]
    attention_bias: bool
    mlp_bias: bool
    tie_embeddings: bool
    intermediate_ratio: float

    def structural_signature(self) -> tuple:
        """What determines the trainable IMPLEMENTATION (activation / tie / ratio are config knobs of it)."""
        return (self.positions, self.gated_mlp, self.norm, self.attention_bias, self.mlp_bias)


# The known families, as trait presets - used ONLY for the honest "base on a family" mode.
_FAMILIES: dict[str, ArchitectureTraits] = {
    "llama": ArchitectureTraits("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3),
    "mistral": ArchitectureTraits("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3),
    "qwen2": ArchitectureTraits("rope", True, "silu", "rmsnorm", True, False, False, 8 / 3),
    "gemma": ArchitectureTraits("rope", True, "gelu_pytorch_tanh", "rmsnorm", False, False, True, 8.0),
    "gpt2": ArchitectureTraits("learned", False, "gelu_new", "layernorm", True, True, True, 4.0),
    "gpt_neox": ArchitectureTraits("rope", False, "gelu", "layernorm", True, True, False, 4.0),
}
KNOWN_FAMILIES: tuple[str, ...] = tuple(_FAMILIES)

# A reference IMPLEMENTATION (a real transformers model_type) per structural signature. A composed design
# whose signature is here trains on that implementation; one that is absent needs the custom-block path.
_IMPL_BY_SIGNATURE: dict[tuple, str] = {
    _FAMILIES["llama"].structural_signature(): "llama",  # gated / RMSNorm / RoPE / no bias
    _FAMILIES["qwen2"].structural_signature(): "qwen2",  # + attention bias
    _FAMILIES["gpt2"].structural_signature(): "gpt2",  # standard / LayerNorm / learned / bias
    _FAMILIES["gpt_neox"].structural_signature(): "gpt_neox",  # standard / LayerNorm / RoPE / bias
}
_HF_STANDARD_TYPES = {"llama", "qwen2", "gemma", "mistral"}  # hidden_size-style config fields


@dataclass(frozen=True)
class BuiltArchitecture:
    """A built architecture: the HF-style config dict + the static (approximate) parameter estimate, the
    chosen dims, and HONEST provenance - ``design_source`` ('family:<x>' or 'composed'), the reference
    ``realizing_family`` (the implementation it runs on), and ``needs_custom_code`` (a novel design that
    no reference implementation builds)."""

    name: str
    config: dict[str, Any]
    estimated_parameters: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    design_source: str
    realizing_family: str | None
    needs_custom_code: bool


def _intermediate_for(traits: ArchitectureTraits, hidden_size: int) -> int:
    raw = int(traits.intermediate_ratio * hidden_size)
    if traits.gated_mlp:  # round gated MLPs to a multiple of 256 (the standard convention)
        return max(256, ((raw + 255) // 256) * 256)
    return max(1, raw)


def estimate_parameters(
    traits: ArchitectureTraits,
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> int:
    """Static estimate of the trainable parameter count from the building blocks + dims. Validated against
    reference models (GPT-2 small ~124M, Llama-7B ~6.7B); NOT a measured count."""
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim
    embeddings = vocab_size * hidden_size
    if traits.positions == "learned":
        embeddings += max_position_embeddings * hidden_size
    attention = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_dim  # Q,O + K,V (GQA-aware)
    if traits.attention_bias:
        attention += 2 * hidden_size + 2 * kv_dim
    if traits.gated_mlp:
        mlp = 3 * hidden_size * intermediate_size
        if traits.mlp_bias:
            mlp += 2 * intermediate_size + hidden_size
    else:
        mlp = 2 * hidden_size * intermediate_size
        if traits.mlp_bias:
            mlp += intermediate_size + hidden_size
    norm_params = hidden_size * (2 if traits.norm == "layernorm" else 1)
    per_layer = attention + mlp + 2 * norm_params
    total = embeddings + num_hidden_layers * per_layer + norm_params  # + final norm
    if not tie_word_embeddings:
        total += vocab_size * hidden_size
    return total


_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4},
    "small": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "base": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "large": {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16},
}


def solve_for_target(
    traits: ArchitectureTraits,
    target_parameters: int,
    *,
    num_key_value_heads: int | None,
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> dict[str, int]:
    """Search valid ``{hidden_size, num_hidden_layers}`` whose ESTIMATE lands closest to the target (heads
    at 64-d, depth ~ hidden/64). Never claims the target is hit exactly."""
    if target_parameters <= 0:
        raise ArchitectureBuilderError("target_parameters must be positive")
    best: tuple[dict[str, int], int] | None = None
    for hidden_size in range(128, 8192 + 1, 64):
        heads = hidden_size // _HEAD_DIM
        kv = num_key_value_heads if num_key_value_heads is not None else heads
        if heads < 1 or hidden_size % heads != 0 or heads % kv != 0:
            continue
        layers = max(2, round(hidden_size / _HEAD_DIM))
        intermediate = _intermediate_for(traits, hidden_size)
        params = estimate_parameters(
            traits,
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv,
            intermediate_size=intermediate,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
        )
        dims = {
            "hidden_size": hidden_size,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "intermediate_size": intermediate,
        }
        if best is None or abs(params - target_parameters) < abs(best[1] - target_parameters):
            best = (dims, params)
    assert best is not None
    return best[0]


def _config_dict(
    traits: ArchitectureTraits,
    model_type: str,
    *,
    name: str,
    dims: dict[str, int],
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """The HF-style architecture config dict (a plain dict the worker's ``from_config`` materializes)."""
    if model_type == "gpt2":
        return {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "corpus_studio_name": name,
            "n_embd": dims["hidden_size"],
            "n_layer": dims["num_hidden_layers"],
            "n_head": dims["num_attention_heads"],
            "n_inner": dims["intermediate_size"],
            "n_positions": max_position_embeddings,
            "vocab_size": vocab_size,
            "tie_word_embeddings": tie_word_embeddings,
            "activation_function": traits.activation,
            "layer_norm_epsilon": 1e-5,
        }
    config = {
        "model_type": model_type,
        "architectures": [f"{model_type.capitalize()}ForCausalLM"],
        "corpus_studio_name": name,
        "hidden_size": dims["hidden_size"],
        "num_hidden_layers": dims["num_hidden_layers"],
        "num_attention_heads": dims["num_attention_heads"],
        "num_key_value_heads": dims["num_key_value_heads"],
        "intermediate_size": dims["intermediate_size"],
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "tie_word_embeddings": tie_word_embeddings,
        "hidden_act": traits.activation,
    }
    if traits.norm == "rmsnorm":
        config["rms_norm_eps"] = 1e-5
    if traits.positions == "rope":
        config["rope_theta"] = 10000.0
    if traits.attention_bias:
        config["attention_bias"] = True
    return config


def _finalize(
    traits: ArchitectureTraits,
    *,
    name: str,
    design_source: str,
    preset: str | None,
    target_parameters: int | None,
    hidden_size: int | None,
    num_hidden_layers: int | None,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
    intermediate_size: int | None,
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> BuiltArchitecture:
    if vocab_size < 1 or max_position_embeddings < 1:
        raise ArchitectureBuilderError("vocab_size and max_position_embeddings must be positive")
    if sum(x is not None for x in (preset, target_parameters, hidden_size)) != 1:
        raise ArchitectureBuilderError(
            "specify exactly one of: a preset, a target parameter count, or explicit hidden_size"
        )
    if preset is not None:
        if preset not in _PRESETS:
            raise ArchitectureBuilderError(f"unknown preset '{preset}'; choose from: {', '.join(_PRESETS)}")
        dims = dict(_PRESETS[preset])
        dims["num_key_value_heads"] = num_key_value_heads or dims["num_attention_heads"]
        dims["intermediate_size"] = _intermediate_for(traits, dims["hidden_size"])
    elif target_parameters is not None:
        dims = solve_for_target(
            traits,
            target_parameters,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
        )
    else:
        assert hidden_size is not None
        heads = num_attention_heads if num_attention_heads is not None else max(1, hidden_size // _HEAD_DIM)
        if heads < 1 or hidden_size % heads != 0:
            raise ArchitectureBuilderError("hidden_size must be divisible by num_attention_heads")
        kv = num_key_value_heads or heads
        if heads % kv != 0:
            raise ArchitectureBuilderError("num_attention_heads must be a multiple of num_key_value_heads")
        dims = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers if num_hidden_layers is not None else max(2, round(hidden_size / _HEAD_DIM)),
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "intermediate_size": intermediate_size if intermediate_size is not None else _intermediate_for(traits, hidden_size),
        }
        if dims["num_hidden_layers"] < 1 or dims["intermediate_size"] < 1:
            raise ArchitectureBuilderError("num_hidden_layers and intermediate_size must be positive")

    realizing_family = _IMPL_BY_SIGNATURE.get(traits.structural_signature())
    needs_custom_code = realizing_family is None
    # A novel design still gets a config + an estimate, but is honestly flagged: no reference
    # implementation builds it, so training it needs the custom-block path. A placeholder model_type
    # marks it (the worker refuses to fabricate an implementation for it).
    model_type = realizing_family if realizing_family is not None else "custom_decoder"
    estimate = estimate_parameters(
        traits,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        intermediate_size=dims["intermediate_size"],
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
    )
    config = _config_dict(
        traits,
        model_type,
        name=name,
        dims=dims,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
    )
    return BuiltArchitecture(
        name=name,
        config=config,
        estimated_parameters=estimate,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        intermediate_size=dims["intermediate_size"],
        vocab_size=vocab_size,
        design_source=design_source,
        realizing_family=realizing_family,
        needs_custom_code=needs_custom_code,
    )


def build_from_family(
    family: str,
    *,
    name: str | None = None,
    preset: str | None = None,
    target_parameters: int | None = None,
    hidden_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    intermediate_size: int | None = None,
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
    tie_word_embeddings: bool | None = None,
) -> BuiltArchitecture:
    """Configure a KNOWN family's architecture (honestly 'based on' that family - NOT your own design)."""
    if family not in _FAMILIES:
        raise ArchitectureBuilderError(
            f"unknown family '{family}'; known: {', '.join(KNOWN_FAMILIES)}"
        )
    traits = _FAMILIES[family]
    if tie_word_embeddings is not None:
        traits = replace(traits, tie_embeddings=tie_word_embeddings)
    return _finalize(
        traits,
        name=name or family,
        design_source=f"family:{family}",
        preset=preset,
        target_parameters=target_parameters,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=traits.tie_embeddings,
    )


def build_composed(
    *,
    name: str,
    positions: str = "rope",
    gated_mlp: bool = True,
    activation: str = "silu",
    norm: str = "rmsnorm",
    attention_bias: bool = False,
    mlp_bias: bool = False,
    tie_embeddings: bool = False,
    intermediate_ratio: float = 8 / 3,
    preset: str | None = None,
    target_parameters: int | None = None,
    hidden_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    intermediate_size: int | None = None,
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
) -> BuiltArchitecture:
    """Compose YOUR OWN architecture DESIGN from building blocks. It is realized on the matching reference
    block implementation; a combination no implementation builds sets ``needs_custom_code`` (a truly-novel
    design needs the custom-block path, not a config)."""
    if positions not in {"rope", "learned"}:
        raise ArchitectureBuilderError("positions must be 'rope' or 'learned'")
    if norm not in {"rmsnorm", "layernorm"}:
        raise ArchitectureBuilderError("norm must be 'rmsnorm' or 'layernorm'")
    traits = ArchitectureTraits(
        positions=positions,  # type: ignore[arg-type]
        gated_mlp=gated_mlp,
        activation=activation,
        norm=norm,  # type: ignore[arg-type]
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        tie_embeddings=tie_embeddings,
        intermediate_ratio=intermediate_ratio,
    )
    return _finalize(
        traits,
        name=name,
        design_source="composed",
        preset=preset,
        target_parameters=target_parameters,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_embeddings,
    )
