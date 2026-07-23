"""In-process adapter-eval backend (eval workflow S3): torch-free seams. The real 4-bit load + generate
is a gated integration test (needs the [train] extra + a GPU) - not runnable in the torch-free CI venv."""
import json

import pytest

from corpus_studio.model_backends.base import BackendGenerateRequest
from corpus_studio.model_backends.in_process import (
    InProcessAdapterBackend,
    InProcessAdapterError,
    _generation_kwargs,
    base_model_for_adapter,
)


def _adapter_dir(tmp_path, base="org/base-7b"):
    directory = tmp_path / "adapter"
    directory.mkdir()
    config = {"peft_type": "LORA", "r": 8}
    if base is not None:
        config["base_model_name_or_path"] = base
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_base_model_prefers_the_adapters_own_recorded_base(tmp_path):
    directory = _adapter_dir(tmp_path, base="org/base-7b")
    assert base_model_for_adapter(directory) == "org/base-7b"
    # the adapter knows what it trained on: its recorded base wins over the fallback
    assert base_model_for_adapter(directory, fallback="org/other") == "org/base-7b"


def test_base_model_falls_back_when_config_omits_it_else_raises(tmp_path):
    directory = _adapter_dir(tmp_path, base=None)
    assert base_model_for_adapter(directory, fallback="org/fallback") == "org/fallback"
    with pytest.raises(InProcessAdapterError, match="pass the base via --model"):
        base_model_for_adapter(directory)  # no recorded base, no fallback


def test_missing_adapter_config_raises_when_no_fallback(tmp_path):
    with pytest.raises(InProcessAdapterError, match="pass the base via --model"):
        base_model_for_adapter(tmp_path / "does-not-exist")


def test_backend_construction_is_cheap_and_lists_the_adapter(tmp_path):
    directory = _adapter_dir(tmp_path)
    backend = InProcessAdapterBackend(str(directory))  # no model load here (lazy)
    assert list(backend.list_models()) == [str(directory)]
    assert backend.config.provider_name == "in-process"
    with pytest.raises(NotImplementedError):
        backend.stream_generate(BackendGenerateRequest(prompt="x"))


def test_health_check_is_false_when_the_adapter_dir_is_missing(tmp_path):
    # Honest readiness (torch-free): a missing dir is never ready, regardless of installed deps.
    assert InProcessAdapterBackend(str(tmp_path / "missing")).health_check() is False


def test_build_backend_routes_in_process_and_requires_an_adapter():
    from corpus_studio.cli import _build_backend

    common = dict(model="org/base", base_url=None, api_key=None, timeout_seconds=120)
    with pytest.raises(ValueError, match="requires --adapter"):
        _build_backend(backend="in-process", adapter=None, **common)
    routed = _build_backend(backend="in-process", adapter="/some/adapter", **common)
    assert isinstance(routed, InProcessAdapterBackend)


@pytest.mark.skip(
    reason="gated integration: loads a real base in 4-bit + adapter and generates - needs the "
    "[train] extra and a GPU, not runnable in the torch-free CI venv."
)
def test_in_process_generate_end_to_end():  # pragma: no cover
    raise AssertionError("integration only")


def test_generation_kwargs_applies_requested_temperature_and_top_p():
    # The requested decode is now actually PASSED to model.generate, not merely recorded. Previously
    # temperature/top_p were omitted, so a run claiming 0.7/0.9 silently decoded at the model defaults.
    sampled = _generation_kwargs(
        BackendGenerateRequest(temperature=0.7, top_p=0.9, max_tokens=50), 2048
    )
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.7 and sampled["top_p"] == 0.9
    assert sampled["max_new_tokens"] == 50


def test_generation_kwargs_greedy_forwards_no_sampler_params():
    # temperature 0 / None -> greedy deterministic decode; top_p is irrelevant and not forwarded.
    greedy = _generation_kwargs(BackendGenerateRequest(temperature=0.0, top_p=0.9), 2048)
    assert greedy["do_sample"] is False
    assert "temperature" not in greedy and "top_p" not in greedy
    assert greedy["max_new_tokens"] == 2048  # falls back to the backend default when request omits it
