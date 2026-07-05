"""Large-dataset smoke/perf test.

The engine must handle a realistic 100k-row dataset without a pathological slowdown or
memory blowup. The time bounds are deliberately GENEROUS — they exist to catch an accidental
O(n^2) regression (e.g. a per-row full-dataset rescan), not to police micro-performance, so a
slow CI runner still passes with wide margin.
"""

import json
import time
from pathlib import Path

from corpus_studio.quality.basic_quality import build_basic_quality_report
from corpus_studio.versions.version_registry import fingerprint_dataset

_N = 100_000


def _write_large_dataset(path: Path) -> list[dict]:
    rows = [
        {
            "instruction": f"Question number {i} about topic {i % 50}",
            "output": f"Answer number {i} with detail {i % 37}.",
        }
        for i in range(_N)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_engine_handles_a_100k_row_dataset(tmp_path: Path):
    path = tmp_path / "examples.jsonl"
    rows = _write_large_dataset(path)

    # Fingerprint streams the file in O(1) memory — must be fast and correct.
    start = time.perf_counter()
    fingerprint, count = fingerprint_dataset(path)
    fingerprint_elapsed = time.perf_counter() - start
    assert count == _N
    assert fingerprint is not None and len(fingerprint) == 64  # sha256 hex
    assert fingerprint_elapsed < 20.0, (
        f"fingerprinting {_N} rows took {fingerprint_elapsed:.1f}s (expected well under 20s — "
        "a regression here means it stopped streaming)"
    )

    # The full quality report (dedup, PII, synthetic patterns, outliers) is O(n) — a big dataset
    # must complete, not hang, with correct counts.
    start = time.perf_counter()
    report = build_basic_quality_report(rows)
    quality_elapsed = time.perf_counter() - start
    assert report.example_count == _N
    assert report.duplicate_exact_count == 0  # every generated row is unique
    assert quality_elapsed < 60.0, (
        f"quality report over {_N} rows took {quality_elapsed:.1f}s (expected well under 60s — "
        "a regression here likely means an accidental O(n^2) scan)"
    )
