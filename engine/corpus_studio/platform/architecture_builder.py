"""Model architecture builder (S3b front-end) - the torch-free producer of the ``architecture_ref`` a
from-scratch pretraining run's ``from_config`` init consumes. Two HONEST modes:

* **base on a family** (``build_from_family``) - configure a KNOWN architecture (its real transformers
  ``model_type``). The result is *based on* that family - NOT "your own model".
* **compose** (``build_composed``) - YOU pick the building blocks (positions, MLP shape, norm, attention
  bias, grouped-query attention) into a novel configuration + your own name. That design is YOURS; it is
  realized on the reference block IMPLEMENTATION whose STRUCTURAL SIGNATURE matches (the same blocks
  families share). A combination NO reference implementation builds sets ``needs_custom_code``: a
  truly-novel design needs the custom-block path (real model code), not a config.

No model is instantiated here (no torch / transformers) - only a config dict + an honest STATIC parameter
ESTIMATE (validated against reference models; a measured count is the worker's job). Estimates are close
for the llama/GPT-2 families and approximate for quirky ones (partial rotary, extra norms) - always
labelled an estimate, never a measured count.
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
    ``intermediate_ratio`` is only a default sizing (explicit dims override it)."""

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


@dataclass(frozen=True)
class _Impl:
    """A known reference implementation: its exact transformers ``model_type`` + architectures class +
    config field style, and the structural traits it realizes. ``supports_gqa`` is False for the classic
    multi-head-only blocks (GPT-2, GPT-NeoX) whose config cannot express distinct KV heads."""

    model_type: str
    architectures: str
    config_style: Literal["hf_standard", "gpt2"]
    traits: ArchitectureTraits
    supports_gqa: bool = True


# transformers ACT2FN activations we allow a composed design to request (anything else would only fail
# far away at worker instantiation, so we fail closed here instead).
_KNOWN_ACTIVATIONS: frozenset[str] = frozenset(
    {
        "silu", "swish", "gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast",
        "quick_gelu", "relu", "relu2", "mish", "tanh", "sigmoid",
    }
)


def _t(
    positions: str,
    gated: bool,
    activation: str,
    norm: str,
    attn_bias: bool,
    mlp_bias: bool,
    tied: bool,
    ratio: float,
) -> ArchitectureTraits:
    return ArchitectureTraits(positions, gated, activation, norm, attn_bias, mlp_bias, tied, ratio)  # type: ignore[arg-type]


# The known families, each a REAL transformers implementation (its own model_type). "Based on <family>".
_FAMILIES: dict[str, _Impl] = {
    "llama": _Impl("llama", "LlamaForCausalLM", "hf_standard", _t("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3)),
    "mistral": _Impl("mistral", "MistralForCausalLM", "hf_standard", _t("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3)),
    "qwen2": _Impl("qwen2", "Qwen2ForCausalLM", "hf_standard", _t("rope", True, "silu", "rmsnorm", True, False, False, 8 / 3)),
    "qwen3": _Impl("qwen3", "Qwen3ForCausalLM", "hf_standard", _t("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3)),
    "gemma": _Impl("gemma", "GemmaForCausalLM", "hf_standard", _t("rope", True, "gelu_pytorch_tanh", "rmsnorm", False, False, True, 8.0)),
    "gemma2": _Impl("gemma2", "Gemma2ForCausalLM", "hf_standard", _t("rope", True, "gelu_pytorch_tanh", "rmsnorm", False, False, True, 4.0)),
    "phi": _Impl("phi", "PhiForCausalLM", "hf_standard", _t("rope", False, "gelu_new", "layernorm", True, True, False, 4.0)),
    "phi3": _Impl("phi3", "Phi3ForCausalLM", "hf_standard", _t("rope", True, "silu", "rmsnorm", False, False, False, 8 / 3)),
    "starcoder2": _Impl("starcoder2", "Starcoder2ForCausalLM", "hf_standard", _t("rope", False, "gelu_pytorch_tanh", "layernorm", True, True, False, 4.0)),
    "stablelm": _Impl("stablelm", "StableLmForCausalLM", "hf_standard", _t("rope", True, "silu", "layernorm", False, False, False, 8 / 3)),
    "gpt2": _Impl("gpt2", "GPT2LMHeadModel", "gpt2", _t("learned", False, "gelu_new", "layernorm", True, True, True, 4.0), supports_gqa=False),
    "gpt_neox": _Impl("gpt_neox", "GPTNeoXForCausalLM", "hf_standard", _t("rope", False, "gelu", "layernorm", True, True, False, 4.0), supports_gqa=False),
}
KNOWN_FAMILIES: tuple[str, ...] = tuple(_FAMILIES)


def _build_signature_map() -> dict[tuple, _Impl]:
    """The representative implementation per STRUCTURAL signature (for compose realizability). The first
    family declared with a given signature is the representative."""
    result: dict[tuple, _Impl] = {}
    for impl in _FAMILIES.values():
        result.setdefault(impl.traits.structural_signature(), impl)
    return result


_IMPL_BY_SIGNATURE = _build_signature_map()


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
    if num_key_value_heads is not None and num_key_value_heads < 1:
        raise ArchitectureBuilderError("num_key_value_heads must be positive")
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
    if best is None:
        raise ArchitectureBuilderError(
            "no architecture in the search range satisfies the constraints (check num_key_value_heads "
            "- it must divide a valid head count)"
        )
    return best[0]


def _config_dict(
    traits: ArchitectureTraits,
    *,
    model_type: str,
    architectures: str,
    config_style: str,
    name: str,
    dims: dict[str, int],
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """The HF-style architecture config dict (a plain dict the worker's ``from_config`` materializes)."""
    if config_style == "gpt2":
        return {
            "model_type": model_type,
            "architectures": [architectures],
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
    config: dict[str, Any] = {
        "model_type": model_type,
        "architectures": [architectures],
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
    if model_type == "custom_decoder":
        # An explicit, durable marker: this design has no reference implementation and needs the
        # custom-block path - planning must refuse it, not seal it as a runnable config.
        config["corpus_studio_needs_custom_code"] = True
    return config


def _finalize(
    traits: ArchitectureTraits,
    *,
    model_type: str,
    architectures: str,
    config_style: str,
    supports_gqa: bool,
    realizing_family: str | None,
    needs_custom_code: bool,
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
    if num_key_value_heads is not None and num_key_value_heads < 1:
        raise ArchitectureBuilderError("num_key_value_heads must be positive")
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

    # A block that cannot express distinct KV heads (GPT-2 / GPT-NeoX) must not silently drop a requested
    # GQA: the estimate would show reduced parameters the built model would not have. Fail closed.
    if not supports_gqa and dims["num_key_value_heads"] < dims["num_attention_heads"]:
        raise ArchitectureBuilderError(
            f"the {realizing_family or model_type} block does not support grouped-query attention "
            "(distinct num_key_value_heads); set it equal to num_attention_heads or choose a GQA-capable "
            "design"
        )

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
        model_type=model_type,
        architectures=architectures,
        config_style=config_style,
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
    """Configure a KNOWN family's architecture, using that family's OWN transformers ``model_type``
    (honestly 'based on' that family - NOT your own design)."""
    if family not in _FAMILIES:
        raise ArchitectureBuilderError(f"unknown family '{family}'; known: {', '.join(KNOWN_FAMILIES)}")
    impl = _FAMILIES[family]
    traits = impl.traits
    if tie_word_embeddings is not None:
        traits = replace(traits, tie_embeddings=tie_word_embeddings)
    return _finalize(
        traits,
        model_type=impl.model_type,
        architectures=impl.architectures,
        config_style=impl.config_style,
        supports_gqa=impl.supports_gqa,
        realizing_family=family,
        needs_custom_code=False,
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
    intermediate_ratio: float | None = None,
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
    """Compose YOUR OWN architecture DESIGN from building blocks. It is realized on the reference
    implementation whose STRUCTURAL SIGNATURE matches; a combination none matches sets
    ``needs_custom_code`` (a truly-novel design needs the custom-block path, not a config)."""
    if positions not in {"rope", "learned"}:
        raise ArchitectureBuilderError("positions must be 'rope' or 'learned'")
    if norm not in {"rmsnorm", "layernorm"}:
        raise ArchitectureBuilderError("norm must be 'rmsnorm' or 'layernorm'")
    if activation not in _KNOWN_ACTIVATIONS:
        raise ArchitectureBuilderError(
            f"unknown activation '{activation}'; choose from: {', '.join(sorted(_KNOWN_ACTIVATIONS))}"
        )
    # Default the MLP ratio by shape when unset (gated ~ 8/3, standard = 4x) so a composed standard MLP is
    # not silently undersized.
    ratio = intermediate_ratio if intermediate_ratio is not None else (8 / 3 if gated_mlp else 4.0)
    traits = ArchitectureTraits(
        positions=positions,  # type: ignore[arg-type]
        gated_mlp=gated_mlp,
        activation=activation,
        norm=norm,  # type: ignore[arg-type]
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        tie_embeddings=tie_embeddings,
        intermediate_ratio=ratio,
    )
    impl = _IMPL_BY_SIGNATURE.get(traits.structural_signature())
    if impl is not None:
        model_type, architectures, config_style = impl.model_type, impl.architectures, impl.config_style
        realizing_family: str | None = impl.model_type
        needs_custom_code = False
        supports_gqa = impl.supports_gqa
    else:
        # A novel design still gets a config + an estimate, but is honestly flagged: no reference
        # implementation builds it, so training it needs the custom-block path (real model code). A
        # custom block is free to implement GQA, so we do not pre-refuse it here.
        model_type, architectures, config_style = "custom_decoder", "CustomDecoderForCausalLM", "hf_standard"
        realizing_family = None
        needs_custom_code = True
        supports_gqa = True
    return _finalize(
        traits,
        model_type=model_type,
        architectures=architectures,
        config_style=config_style,
        supports_gqa=supports_gqa,
        realizing_family=realizing_family,
        needs_custom_code=needs_custom_code,
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
