"""Shared pytest fixtures.

The suite is written against the dependency-light baseline that CI runs: the optional
tokenizer extras (``tokenizer`` -> tiktoken, ``model-tokenizer`` -> tokenizers) are absent,
so token counts come from the Unicode-aware heuristic. A developer machine or a training
host may have those extras installed, which would otherwise flip token counts and estimator
names under the tests' feet (e.g. ``'tiktoken'`` where a test asserts ``'heuristic'``). This
does not change what CI validates -- it only makes local runs match CI regardless of which
optional packages happen to be installed.
"""

import pytest

from corpus_studio.tokenization import estimate as _estimate


@pytest.fixture(autouse=True)
def _deterministic_token_estimator(monkeypatch):
    """Neutralize the optional tiktoken tier and clear the estimator caches so token
    estimation is deterministic (the Unicode-aware heuristic) on every host, matching CI.

    The model-tokenizer (Hub) tier is already opt-in per call, and each model-tier test
    installs its own fake loader via ``monkeypatch`` *after* this autouse fixture runs (so a
    test's explicit tier selection still wins). Clearing ``_hf_tokenizer_cache`` on both ends
    also stops estimator state from leaking between tests.
    """
    _estimate._hf_tokenizer_cache.clear()
    # Optional tiktoken tier off (would download cl100k_base on first use).
    monkeypatch.setattr(_estimate, "_tiktoken_encoder", lambda: None)
    # Model-tokenizer (Hub) tier off at the network boundary, so no test hits the Hub when the
    # `model-tokenizer` extra happens to be installed. `_load_hf_tokenizer`'s offline logic stays
    # real (the offline/online tests mock `_fetch_hf_tokenizer` themselves, which overrides this),
    # and model-tier tests inject a fake `_load_hf_tokenizer` (a level up) that also wins.
    monkeypatch.setattr(_estimate, "_fetch_hf_tokenizer", lambda model_id: None)
    yield
    _estimate._hf_tokenizer_cache.clear()
