"""In-process adapter evaluation backend (eval workflow S3).

Evaluate a trained LoRA adapter IN-PROCESS by loading the base model in 4-bit nf4 + the PEFT adapter -
the exact QLoRA regime it trained under, and the faithful path the WBG eval close-out used. This is the
only in-repo route to evaluate an adapter: ``train-merge`` produces fp16 HF weights only, and the ollama
safetensors -> GGUF import produced a broken model.

All heavy imports (torch / transformers / peft / bitsandbytes) are LAZY, so importing this module stays
torch-free; the model loads on the first ``generate()``. It satisfies the :class:`ModelBackend` Protocol,
so the existing ``run_evaluation`` + scorer + report path is reused unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from corpus_studio.model_backends.base import (
    BackendGenerateRequest,
    BackendGenerateResponse,
    ModelBackendConfig,
)

_TRAIN_DEPENDENCIES = ("torch", "transformers", "peft", "bitsandbytes")


class InProcessAdapterError(RuntimeError):
    """The in-process adapter backend could not resolve, load, or generate."""


def _generation_kwargs(request: BackendGenerateRequest, default_max_tokens: int) -> dict[str, Any]:
    """The decode kwargs actually PASSED to ``model.generate`` - so a recorded temperature/top_p is
    honoured, not merely logged.

    ``temperature`` None/0 is greedy (deterministic; ``top_p`` is irrelevant). Otherwise the REQUESTED
    ``temperature`` and ``top_p`` are applied. Previously they were omitted from the generate call, so a
    run that recorded temperature 0.7 / top_p 0.9 silently decoded at the model's defaults (1.0/1.0) -
    a decode configuration the saved run claimed but did not honour.
    """
    greedy = not request.temperature
    kwargs: dict[str, Any] = {
        "max_new_tokens": request.max_tokens or default_max_tokens,
        "do_sample": not greedy,
        "use_cache": True,
    }
    if not greedy:
        kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
    return kwargs


def base_model_for_adapter(adapter_dir: str | Path, fallback: str | None = None) -> str:
    """Resolve the base model to load UNDER the adapter. Prefer the adapter's own recorded base
    (``base_model_name_or_path`` in ``adapter_config.json``) - the adapter knows what it trained on -
    and use ``fallback`` (an explicit override) only when the config omits it. Torch-free (a JSON read)."""
    config_path = Path(adapter_dir) / "adapter_config.json"
    if config_path.exists():
        try:
            recorded = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError) as exc:
            raise InProcessAdapterError(f"could not read {config_path}: {exc}") from exc
        if recorded:
            return str(recorded)
    if fallback:
        return fallback
    raise InProcessAdapterError(
        f"no base_model_name_or_path in {adapter_dir}/adapter_config.json; pass the base via --model"
    )


class InProcessAdapterBackend:
    """A :class:`ModelBackend` that loads base (4-bit nf4) + LoRA adapter in-process and decodes greedily
    (deterministic when the request carries ``seed`` + ``temperature`` 0)."""

    def __init__(
        self,
        adapter_dir: str | Path,
        *,
        base_model: str | None = None,
        config: ModelBackendConfig | None = None,
    ) -> None:
        self._adapter_dir = str(adapter_dir)
        self._base_fallback = base_model
        self.config = config or ModelBackendConfig(
            provider_name="in-process", base_url="", model_name=self._adapter_dir, max_tokens=2048,
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        base = base_model_for_adapter(self._adapter_dir, self._base_fallback)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quant,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="sdpa",  # math-safe on sm_120 Blackwell
        )
        self._model = PeftModel.from_pretrained(model, self._adapter_dir).eval()

    def generate(self, request: BackendGenerateRequest) -> BackendGenerateResponse:
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        messages = request.messages or [{"role": "user", "content": request.prompt or ""}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        if request.seed is not None:
            torch.manual_seed(request.seed)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.eos_token_id,
                **_generation_kwargs(request, self.config.max_tokens),
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return BackendGenerateResponse(text=str(text), model_name=self._adapter_dir)

    def list_models(self) -> Sequence[str]:
        return [self._adapter_dir]

    def stream_generate(self, request: BackendGenerateRequest) -> Iterator[str]:
        raise NotImplementedError("the in-process adapter backend does not stream")

    def health_check(self) -> bool:
        """Torch-free readiness: the adapter dir exists AND every [train] dependency is importable
        (checked via find_spec, WITHOUT importing torch here). The real load happens on generate()."""
        import importlib.util  # noqa: PLC0415

        if not Path(self._adapter_dir).exists():
            return False
        return all(importlib.util.find_spec(name) is not None for name in _TRAIN_DEPENDENCIES)
