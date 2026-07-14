"""Dependency-light observations for optional quantization runtimes."""

from __future__ import annotations

from typing import Any


def find_linear4bit_modules(model: Any, linear4bit_type: type[Any]) -> list[Any]:
    """Return true bitsandbytes ``Linear4bit`` instances from ``model``.

    PEFT also exposes an unrelated wrapper named ``Linear4bit``. Runtime type identity, rather than
    the shared class name, keeps wrapper metadata from contaminating quantization observations.
    """

    return [module for module in model.modules() if isinstance(module, linear4bit_type)]
