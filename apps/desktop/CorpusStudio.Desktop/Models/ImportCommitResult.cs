namespace CorpusStudio.Desktop.Models;

public sealed record ImportCommitResult(
    int ImportedCount,
    int QuarantinedCount,
    string? QuarantinePath,
    int SkippedDuplicateCount = 0
)
{
    /// <summary>An import that actually appended rows changed the dataset, so it should be
    /// snapshotted as a dataset version. An all-duplicates import (nothing added) is not.</summary>
    public bool ShouldAutoCapture => ImportedCount > 0;

    /// <summary>Label for the auto-captured dataset version.</summary>
    public string AutoCaptureLabel => $"After import (+{ImportedCount} rows)";
}
