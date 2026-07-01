from corpus_studio.quality.basic_quality import (
    SyntheticPatternIssue,
    build_basic_quality_report,
    cluster_synthetic_pattern_issues,
)


def _issue(kind: str, pattern: str, severity: str, rows: list[int]) -> SyntheticPatternIssue:
    return SyntheticPatternIssue(
        kind=kind,
        pattern=pattern,
        severity=severity,
        message=pattern,
        row_numbers=rows,
        suggestion="s",
    )


def test_clusters_merge_near_duplicate_openings():
    issues = [
        _issue("repeated_opening", "sure here is a", "high", [1, 2, 3]),
        _issue("repeated_opening", "sure here is the", "medium", [4, 5, 6]),
        _issue("repeated_opening", "in a distant galaxy far", "low", [7, 8]),
    ]
    clusters = cluster_synthetic_pattern_issues(issues)

    # The two "sure here is ..." openings merge; the unrelated one stays separate.
    assert len(clusters) == 2
    biggest = clusters[0]
    assert biggest.member_count == 2
    assert biggest.row_numbers == [1, 2, 3, 4, 5, 6]
    assert biggest.severity == "high"  # max of high/medium


def test_clusters_do_not_merge_different_kinds():
    issues = [
        _issue("repeated_opening", "sure here is", "low", [1]),
        _issue("repeated_closing", "sure here is", "low", [2]),
    ]
    assert len(cluster_synthetic_pattern_issues(issues)) == 2


def test_clusters_keep_dissimilar_patterns_separate():
    issues = [
        _issue("generic_phrase", "as an ai language model", "medium", [1, 2]),
        _issue("generic_phrase", "in conclusion", "medium", [3, 4]),
    ]
    assert len(cluster_synthetic_pattern_issues(issues)) == 2


def test_quality_report_includes_clusters():
    rows = [
        {"instruction": "Explain a concept", "output": f"Sure here is the answer number {index} you should use"}
        for index in range(6)
    ]
    report = build_basic_quality_report(rows)
    assert report.synthetic_pattern_clusters
    covered = {row for cluster in report.synthetic_pattern_clusters for row in cluster.row_numbers}
    assert covered == {1, 2, 3, 4, 5, 6}
