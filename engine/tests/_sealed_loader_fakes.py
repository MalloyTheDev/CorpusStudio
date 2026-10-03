"""Deterministic, torch-free fakes for the sealed-loader tests (#863).

A fake ``torch`` that exposes exactly what the sealed-loader helpers touch: dtype identities, the three
observable SDPA toggles, ``cuda.is_available``, ``randn`` plus ``nn.functional.scaled_dot_product_attention``
for the isolated kernel probe, and ``nn.attention.sdpa_kernel`` as an exclusive-kernel context. Recording
Transformers/PEFT/bitsandbytes stand-ins return models that report precisely the attention API, device
map, quantization and dtypes they were LOADED with, so a test observes what the worker asked the loader
for rather than what a fake chose to claim. Every call lands on one ``events`` timeline.

Not a test module: imported by ``test_sealed_loader*.py`` and the #862 consumption tests.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from collections.abc import Iterator
from typing import Any

FLOAT_DTYPES = ("bfloat16", "float16", "float32")


class FakeDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    __str__ = __repr__


class FakeTensor:
    def __init__(self, torch: FakeTorch, *, device: str, dtype: FakeDtype) -> None:
        self._torch = torch
        self.device = device
        self.dtype = dtype

    def sum(self) -> FakeTensor:
        return self

    def backward(self) -> None:
        self._torch.events.append(("probe_backward", self.device, self.dtype.name))


class _SdpToggles:
    """``torch.backends.cuda``: three settable, observable global SDPA toggles (torch's defaults)."""

    def __init__(self, torch: FakeTorch) -> None:
        self._torch = torch
        self.state = {"flash": True, "mem_efficient": True, "math": True}

    def _set(self, key: str, value: bool) -> None:
        self.state[key] = bool(value)
        self._torch.events.append(("sdp_toggle", key, bool(value)))

    def enable_flash_sdp(self, value: bool) -> None:
        self._set("flash", value)

    def enable_mem_efficient_sdp(self, value: bool) -> None:
        self._set("mem_efficient", value)

    def enable_math_sdp(self, value: bool) -> None:
        self._set("math", value)

    def flash_sdp_enabled(self) -> bool:
        return self.state["flash"]

    def mem_efficient_sdp_enabled(self) -> bool:
        return self.state["mem_efficient"]

    def math_sdp_enabled(self) -> bool:
        return self.state["math"]


class FakeTorch(types.ModuleType):
    def __init__(self, *, cuda_available: bool = True) -> None:
        super().__init__("torch")
        self.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        self.events: list[tuple[Any, ...]] = []
        self.active_kernels: list[tuple[str, ...]] = []
        for name in (*FLOAT_DTYPES, "long", "int8", "uint8"):
            setattr(self, name, FakeDtype(name))
        self.backends = types.SimpleNamespace(cuda=_SdpToggles(self))
        self.cuda = types.SimpleNamespace(
            is_available=lambda: cuda_available, manual_seed_all=lambda _seed: None
        )
        self.manual_seed = lambda _seed: None

        backend = types.SimpleNamespace(
            MATH="MATH", FLASH_ATTENTION="FLASH_ATTENTION", EFFICIENT_ATTENTION="EFFICIENT_ATTENTION"
        )

        @contextlib.contextmanager
        def sdpa_kernel(backends: list[str]) -> Iterator[None]:
            self.active_kernels.append(tuple(backends))
            self.events.append(("sdpa_kernel_enter", tuple(backends)))
            try:
                yield
            finally:
                self.active_kernels.pop()
                self.events.append(("sdpa_kernel_exit", tuple(backends)))

        def scaled_dot_product_attention(q: FakeTensor, _k: Any, _v: Any) -> FakeTensor:
            self.events.append(
                ("probe_sdpa", self.active_kernels[-1] if self.active_kernels else None)
            )
            return FakeTensor(self, device=q.device, dtype=q.dtype)

        self.nn = _module(
            "torch.nn",
            functional=_module(
                "torch.nn.functional", scaled_dot_product_attention=scaled_dot_product_attention
            ),
            attention=_module("torch.nn.attention", SDPBackend=backend, sdpa_kernel=sdpa_kernel),
        )

    def randn(self, *_shape: int, device: str, dtype: FakeDtype, requires_grad: bool) -> FakeTensor:
        self.events.append(("probe_randn", device, dtype.name, requires_grad))
        return FakeTensor(self, device=device, dtype=dtype)

    def sdp_state(self) -> tuple[bool, bool, bool]:
        state = self.backends.cuda.state
        return state["flash"], state["mem_efficient"], state["math"]


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class FakeParameter:
    """A parameter whose ``.data`` assignment carries a dtype change, as a torch parameter's does."""

    def __init__(
        self, *, device: str, dtype: FakeDtype, requires_grad: bool = False
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.hooks: list[Any] = []

    @property
    def data(self) -> FakeParameter:
        return self

    @data.setter
    def data(self, value: FakeParameter) -> None:
        self.dtype = value.dtype
        self.device = value.device

    def to(self, *, dtype: FakeDtype) -> FakeParameter:
        return FakeParameter(device=self.device, dtype=dtype, requires_grad=self.requires_grad)

    def is_floating_point(self) -> bool:
        return self.dtype.name in FLOAT_DTYPES

    def register_post_accumulate_grad_hook(self, hook: Any) -> None:
        self.hooks.append(hook)


class FakeLinear4bit:
    """Stand-in for ``bitsandbytes.nn.Linear4bit`` reporting the quant type and compute dtype it was
    constructed from (the loader's BitsAndBytesConfig)."""

    def __init__(self, *, quant_type: str, compute_dtype: FakeDtype) -> None:
        self.weight = types.SimpleNamespace(
            quant_state=types.SimpleNamespace(quant_type=quant_type)
        )
        self.compute_dtype = compute_dtype


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeBitsAndBytesConfig) and other.kwargs == self.kwargs

    def __repr__(self) -> str:
        return f"BitsAndBytesConfig({self.kwargs})"


class FakeLoadedModel:
    """A model that reports exactly what ``from_pretrained`` was asked for: its config's attention API is
    the requested ``attn_implementation``, every tensor sits on the requested root device, an unquantized
    load stores its weights in ``torch_dtype``, and a 4-bit load exposes Linear4bit modules built from the
    requested BitsAndBytesConfig."""

    def __init__(self, torch: FakeTorch, location: str, kwargs: dict[str, Any]) -> None:
        self.location = location
        self.load_kwargs = kwargs
        device_map = kwargs.get("device_map") or {"": "cpu"}
        self.device = device_map[""]
        self.hf_device_map = dict(device_map)
        self.config = types.SimpleNamespace(
            _attn_implementation=kwargs.get("attn_implementation"),
            use_cache=True,
            pad_token_id=None,
            to_dict=lambda: {"model_type": "fake"},
        )
        self._linear4bit: list[FakeLinear4bit] = []
        bnb = kwargs.get("quantization_config")
        if bnb is not None:
            bnb_kwargs = bnb.kwargs
            self._linear4bit.append(
                FakeLinear4bit(
                    quant_type=bnb_kwargs["bnb_4bit_quant_type"],
                    compute_dtype=bnb_kwargs["bnb_4bit_compute_dtype"],
                )
            )
            weight_dtype = torch.uint8
        else:
            weight_dtype = kwargs.get("torch_dtype", torch.float32)
        self._parameters: list[tuple[str, FakeParameter]] = [
            (
                "model.layers.0.weight",
                FakeParameter(
                    device=self.device, dtype=weight_dtype, requires_grad=bnb is None
                ),
            )
        ]
        self.gradient_checkpointing = False

    def named_parameters(self, remove_duplicate: bool = True) -> list[tuple[str, FakeParameter]]:
        return list(self._parameters)

    def parameters(self) -> Iterator[FakeParameter]:
        return iter([parameter for _, parameter in self._parameters])

    def named_buffers(self, remove_duplicate: bool = True) -> list[tuple[str, Any]]:
        return []

    def named_modules(self, remove_duplicate: bool = True) -> list[tuple[str, Any]]:
        return [("", self)]

    def modules(self) -> list[Any]:
        return [self, *self._linear4bit]

    def attach_lora(self, dtype: FakeDtype) -> None:
        self._parameters.append(
            (
                "model.layers.0.lora_A.weight",
                FakeParameter(device=self.device, dtype=dtype, requires_grad=True),
            )
        )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def save_pretrained(self, _path: str, **_kwargs: Any) -> None:
        return None


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def __init__(self, location: str, *, chat_template: str | None) -> None:
        self.location = location
        self.chat_template = chat_template

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> str:
        return "|".join(str(message.get("content", "")) for message in messages)

    def __call__(self, _text: str, **_kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [1, 2, 3]}

    def save_pretrained(self, _path: str) -> None:
        return None


class StopAtKbit(Exception):
    """Raised by the fake ``prepare_model_for_kbit_training`` when a test stops right after loading."""


class FakeStack:
    """Installs a coherent fake torch/transformers/peft/bitsandbytes/datasets set into ``sys.modules``."""

    def __init__(
        self,
        monkeypatch: Any,
        *,
        chat_template: str | None = "{{ messages }}",
        stop_at_kbit: bool = False,
        lora_dtype: str = "bfloat16",
        cuda_available: bool = True,
    ) -> None:
        self.torch = FakeTorch(cuda_available=cuda_available)
        self.events = self.torch.events
        self.chat_template = chat_template
        self.models: list[FakeLoadedModel] = []
        stack = self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(location: str, **kwargs: Any) -> FakeTokenizer:
                stack.events.append(("load", "AutoTokenizer", location, kwargs))
                return FakeTokenizer(location, chat_template=stack.chat_template)

        class _AutoModel:
            @classmethod
            def from_pretrained(cls, location: str, **kwargs: Any) -> FakeLoadedModel:
                stack.events.append(("load", cls.__name__, location, kwargs))
                model = FakeLoadedModel(stack.torch, location, kwargs)
                stack.models.append(model)
                return model

        class AutoModelForCausalLM(_AutoModel):
            pass

        class AutoModelForSequenceClassification(_AutoModel):
            pass

        def prepare_model_for_kbit_training(model: Any, **kwargs: Any) -> Any:
            stack.events.append(("prepare_kbit", kwargs))
            if stop_at_kbit:
                raise StopAtKbit("stopped right after the sealed model load")
            return model

        def get_peft_model(model: FakeLoadedModel, config: Any) -> FakeLoadedModel:
            stack.events.append(("get_peft_model", config))
            model.attach_lora(getattr(stack.torch, lora_dtype))
            return model

        class LoraConfig:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class PeftModel:
            @staticmethod
            def from_pretrained(base: Any, location: str) -> Any:
                stack.events.append(("load", "PeftModel", location, {}))
                return base

        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
        self.transformers = _module(
            "transformers",
            AutoTokenizer=AutoTokenizer,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoModelForSequenceClassification=AutoModelForSequenceClassification,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
            TrainerCallback=object,
            set_seed=lambda _seed: None,
        )
        self.peft = _module(
            "peft",
            LoraConfig=LoraConfig,
            PeftModel=PeftModel,
            get_peft_model=get_peft_model,
            get_peft_model_state_dict=lambda _model: {},
            prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        )
        class Dataset:
            @staticmethod
            def from_list(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                stack.events.append(("dataset_from_list", len(rows)))
                return rows

        bnb_nn = _module("bitsandbytes.nn", Linear4bit=FakeLinear4bit)
        modules = {
            "torch": self.torch,
            "torch.nn": self.torch.nn,
            "torch.nn.functional": self.torch.nn.functional,
            "torch.nn.attention": self.torch.nn.attention,
            "transformers": self.transformers,
            "peft": self.peft,
            "bitsandbytes": _module("bitsandbytes", nn=bnb_nn),
            "bitsandbytes.nn": bnb_nn,
            "datasets": _module("datasets", Dataset=Dataset),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)

    def loads(self) -> list[tuple[str, str, dict[str, Any]]]:
        return [(event[1], event[2], event[3]) for event in self.events if event[0] == "load"]
