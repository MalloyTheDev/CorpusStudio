"""The worker applies the plan's SEALED allocator policy to PYTORCH_CUDA_ALLOC_CONF before torch loads
(7B ladder P4 - allocator wiring). torch reads that variable once at first CUDA init, so it must be set
before the runner imports torch; and it comes only from the hash-verified plan, never the launcher."""

import os
from types import SimpleNamespace

import pytest

from corpus_studio.platform.enums import AllocatorPolicy
from corpus_studio.platform.worker import _apply_allocator_policy
from corpus_studio.platform.worker_protocol import WorkerProtocolError


def _plan(policy: AllocatorPolicy, *, max_split_size_mb=None, gc_threshold=None) -> SimpleNamespace:
    return SimpleNamespace(
        allocator_policy=policy,
        allocator_max_split_size_mb=max_split_size_mb,
        allocator_gc_threshold=gc_threshold,
    )


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


def test_max_split_size_applies_the_sealed_mb(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.max_split_size, max_split_size_mb=128))
    assert conf == "max_split_size_mb:128"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_garbage_collection_applies_the_sealed_threshold(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.garbage_collection, gc_threshold=0.8))
    assert conf == "garbage_collection_threshold:0.8"


def test_max_split_size_merges_and_replaces_same_key(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
    conf = _apply_allocator_policy(_plan(AllocatorPolicy.max_split_size, max_split_size_mb=128))
    assert "expandable_segments:True" in conf  # keeps the operator's other setting
    assert "max_split_size_mb:128" in conf and "max_split_size_mb:64" not in conf  # replaced


def test_max_split_size_without_its_parameter_fails_closed(monkeypatch):
    # A sealed parameterized policy with no parameter must NOT silently downgrade to default.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with pytest.raises(WorkerProtocolError, match="allocator_max_split_size_mb"):
        _apply_allocator_policy(_plan(AllocatorPolicy.max_split_size))
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ  # nothing applied on refusal


def test_garbage_collection_without_its_parameter_fails_closed(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    with pytest.raises(WorkerProtocolError, match="allocator_gc_threshold"):
        _apply_allocator_policy(_plan(AllocatorPolicy.garbage_collection))
