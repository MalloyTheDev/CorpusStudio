"""The worker applies the plan's SEALED allocator policy to PYTORCH_CUDA_ALLOC_CONF before torch loads
(7B ladder P4 - allocator wiring). torch reads that variable once at first CUDA init, so it must be set
before the runner imports torch; and it comes only from the hash-verified plan, never the launcher."""

import os
from types import SimpleNamespace

from corpus_studio.platform.enums import AllocatorPolicy
from corpus_studio.platform.worker import _apply_allocator_policy


def _plan(policy: AllocatorPolicy) -> SimpleNamespace:
    return SimpleNamespace(allocator_policy=policy)


def test_expandable_segments_sets_the_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.expandable_segments))
    assert conf == "expandable_segments:True"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_default_is_a_noop(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.default))
    assert conf == "default"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


def test_merges_into_an_existing_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.expandable_segments))
    # keeps the operator's setting, never a silent override
    assert "garbage_collection_threshold:0.8" in conf
    assert "expandable_segments:True" in conf


def test_does_not_duplicate_an_existing_expandable_segments(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.expandable_segments))
    assert conf == "expandable_segments:True"  # replaced, not appended


def test_missing_field_defaults_safely(monkeypatch):
    # a plan without the field (older shape) is treated as default - never crashes the worker
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert _apply_allocator_policy(SimpleNamespace()) == "default"
