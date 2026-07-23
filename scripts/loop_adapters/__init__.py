"""Concrete runtime adapters that fill the autonomous loop's injected-effect seams.

The loop controller (``scripts/loop/``) is a deterministic, stdlib-only state machine whose every real
effect - executor / reviewer / critic / gh / cs_assure / verify_paths - is an INJECTED callback. An adapter
here is a ``build_context(repo_root, base) -> loop.orchestrate.LoopContext`` module that binds those seams
to real effects, so ``cs_loop run --adapters <file>`` can drive the loop against a live repository.

Adapters are intentionally SEPARATE from ``scripts/loop/`` (which must stay effect-free) and are ordered
by how much they are allowed to DO: the dry-run adapter is read-only (it proposes, never writes); a
write-capable adapter is a later, review-gated step.
"""
