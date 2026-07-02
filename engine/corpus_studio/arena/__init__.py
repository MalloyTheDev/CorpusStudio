"""Model Chat Lab / Arena: run a prompt suite across several models.

Unlike the evaluation benchmark (which scores models against a labeled dataset),
the arena runs ad-hoc prompt suites and captures each model's response side by
side for comparison. Responses are comparison artifacts, not trainable dataset
rows. Judging (an evaluator-only model ranking responses) and saved comparison
reports build on this foundation in later slices.
"""

from corpus_studio.arena.models import (
    ArenaModelSummary,
    ArenaPrompt,
    ArenaReport,
    ArenaResponse,
    build_arena_report,
)
from corpus_studio.arena.runner import load_prompt_suite, run_arena

__all__ = [
    "ArenaModelSummary",
    "ArenaPrompt",
    "ArenaReport",
    "ArenaResponse",
    "build_arena_report",
    "load_prompt_suite",
    "run_arena",
]
