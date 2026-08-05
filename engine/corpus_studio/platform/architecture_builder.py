"""Model architecture builder (S3b front-end): generate a validated, hash-pinnable architecture config
for a FROM-SCRATCH pretraining run - the torch-free producer of the ``architecture_ref`` the pretraining
worker's ``from_config`` random init consumes (``ModelInitializationSpec.architecture_ref``).

Three ways to say "what to start at" (no model is instantiated here - no torch / transformers; only a
config dict + an HONEST parameter ESTIMATE, always labelled approximate, never a measured count):

* a named ``preset`` (tiny / small / base / large),
* a ``target_parameters`` count - solved to the nearest valid config, OR
* ``explicit`` dimensions.

The parameter formula is per-family and validated in tests against known reference models (GPT-2 small
~124M). It is a STATIC estimate of trainable parameters; the worker's measured accounting is authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Family = Literal["llama", "gpt2"]
_FAMILIES: tuple[Family, ...] = ("llama", "gpt2")
_HEAD_DIM = 64  # every family here sizes heads at 64-d (heads = hidden // 64).


class ArchitectureBuilderError(ValueError):
    """A from-scratch architecture request the builder cannot honor (fail-closed, a clean typed error)."""


@dataclass(frozen=True)
class BuiltArchitecture:
    """The result of a build: the HF-style config dict + the STATIC (approximate) parameter estimate and
    the dimensions chosen. ``config`` is what gets written to the ``--architecture-config`` file and
    hash-pinned into the sealed ``ModelInitializationSpec.architecture_ref``."""

    family: Family
    config: dict[str, Any]
    estimated_parameters: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int


def _intermediate_for(family: Family, hidden_size: int) -> int:
    if family == "gpt2":
        return 4 * hidden_size
    # Llama/SwiGLU: ~8/3 * hidden, rounded to a multiple of 256 (the standard convention).
    raw = int(8 * hidden_size / 3)
    return max(256, ((raw + 255) // 256) * 256)


def estimate_parameters(
    family: Family,
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
    """Static estimate of the trainable parameter count for one of the supported families. Exact enough
    to size a model (validated against reference models); NOT a measured count."""
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_key_value_heads * head_dim
    if family == "gpt2":
        # Learned token + position embeddings, biased Linear, tied output, GELU MLP, LayerNorm.
        embeddings = vocab_size * hidden_size + max_position_embeddings * hidden_size
        attention = (3 * hidden_size * hidden_size + 3 * hidden_size) + (
            hidden_size * hidden_size + hidden_size
        )
        mlp = (hidden_size * intermediate_size + intermediate_size) + (
            intermediate_size * hidden_size + hidden_size
        )
        norms = 4 * hidden_size  # two LayerNorms (weight + bias) per block
        per_layer = attention + mlp + norms
        total = embeddings + num_hidden_layers * per_layer + 2 * hidden_size  # + final LayerNorm
    else:  # llama: RoPE (no learned position embeddings), no biases, SwiGLU, RMSNorm
        embeddings = vocab_size * hidden_size
        attention = (
            hidden_size * hidden_size  # Q
            + 2 * hidden_size * kv_dim  # K, V (GQA-aware)
            + hidden_size * hidden_size  # O
        )
        mlp = 3 * hidden_size * intermediate_size  # gate + up + down
        norms = 2 * hidden_size  # two RMSNorms (weight only) per block
        per_layer = attention + mlp + norms
        total = embeddings + num_hidden_layers * per_layer + hidden_size  # + final RMSNorm
    if not tie_word_embeddings:
        total += vocab_size * hidden_size
    return total


# Named starting points - family-agnostic dimension sets (the "what to start at" menu).
_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4},
    "small": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "base": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "large": {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16},
}


def _default_tie(family: Family) -> bool:
    return family == "gpt2"  # GPT-2 ties input/output embeddings; Llama unties by default.


def solve_for_target(
    family: Family,
    target_parameters: int,
    *,
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> dict[str, int]:
    """Search valid ``{hidden_size, num_hidden_layers}`` for the config whose ESTIMATE lands closest to
    ``target_parameters``. Width sizes heads at 64-d; depth follows a roughly-square heuristic
    (layers ~ hidden/64). Returns the winning dimensions (never claims the target is hit exactly)."""
    if target_parameters <= 0:
        raise ArchitectureBuilderError("target_parameters must be positive")
    best: tuple[dict[str, int], int] | None = None
    for hidden_size in range(128, 8192 + 1, 64):
        num_attention_heads = hidden_size // _HEAD_DIM
        if num_attention_heads < 1 or hidden_size % num_attention_heads != 0:
            continue
        num_hidden_layers = max(2, round(hidden_size / _HEAD_DIM))
        intermediate_size = _intermediate_for(family, hidden_size)
        params = estimate_parameters(
            family,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
        )
        dims = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
        }
        if best is None or abs(params - target_parameters) < abs(best[1] - target_parameters):
            best = (dims, params)
    assert best is not None  # the range always yields at least one candidate
    return best[0]


def _config_dict(
    family: Family,
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    intermediate_size: int,
    vocab_size: int,
    max_position_embeddings: int,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """The HF-style architecture config dict for the family (no transformers import - a plain dict the
    worker's ``AutoConfig``/``from_config`` materializes at train time)."""
    common = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "tie_word_embeddings": tie_word_embeddings,
    }
    if family == "gpt2":
        return {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "n_embd": hidden_size,
            "n_layer": num_hidden_layers,
            "n_head": num_attention_heads,
            "n_inner": intermediate_size,
            "n_positions": max_position_embeddings,
            "vocab_size": vocab_size,
            "tie_word_embeddings": tie_word_embeddings,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
        }
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        **common,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }


def build_architecture(
    family: Family,
    *,
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
    """Build a fresh architecture config by EXACTLY ONE of: ``preset``, ``target_parameters``, or the
    explicit ``hidden_size`` (+ optional dims). Fail-closed on an ambiguous or under-specified request."""
    if family not in _FAMILIES:
        raise ArchitectureBuilderError(
            f"unsupported family '{family}'; supported: {', '.join(_FAMILIES)}"
        )
    if vocab_size < 1 or max_position_embeddings < 1:
        raise ArchitectureBuilderError("vocab_size and max_position_embeddings must be positive")
    tie = _default_tie(family) if tie_word_embeddings is None else tie_word_embeddings
    modes = [preset is not None, target_parameters is not None, hidden_size is not None]
    if sum(modes) != 1:
        raise ArchitectureBuilderError(
            "specify exactly one of: --preset, --params (target), or explicit --hidden-size"
        )

    if preset is not None:
        if preset not in _PRESETS:
            raise ArchitectureBuilderError(
                f"unknown preset '{preset}'; choose from: {', '.join(_PRESETS)}"
            )
        dims = dict(_PRESETS[preset])
        dims["intermediate_size"] = _intermediate_for(family, dims["hidden_size"])
    elif target_parameters is not None:
        dims = solve_for_target(
            family,
            target_parameters,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=tie,
        )
    else:
        assert hidden_size is not None
        heads = num_attention_heads if num_attention_heads is not None else max(1, hidden_size // _HEAD_DIM)
        if heads < 1 or hidden_size % heads != 0:
            raise ArchitectureBuilderError("hidden_size must be divisible by num_attention_heads")
        dims = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers if num_hidden_layers is not None else max(2, round(hidden_size / _HEAD_DIM)),
            "num_attention_heads": heads,
            "intermediate_size": intermediate_size if intermediate_size is not None else _intermediate_for(family, hidden_size),
        }
        if dims["num_hidden_layers"] < 1 or dims["intermediate_size"] < 1:
            raise ArchitectureBuilderError("num_hidden_layers and intermediate_size must be positive")

    kv_heads = num_key_value_heads if num_key_value_heads is not None else dims["num_attention_heads"]
    if dims["num_attention_heads"] % kv_heads != 0:
        raise ArchitectureBuilderError("num_attention_heads must be a multiple of num_key_value_heads")
    estimate = estimate_parameters(
        family,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=kv_heads,
        intermediate_size=dims["intermediate_size"],
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie,
    )
    config = _config_dict(
        family,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=kv_heads,
        intermediate_size=dims["intermediate_size"],
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie,
    )
    return BuiltArchitecture(
        family=family,
        config=config,
        estimated_parameters=estimate,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        intermediate_size=dims["intermediate_size"],
        vocab_size=vocab_size,
    )
