namespace CorpusStudio.Desktop.Models;

public sealed record ImportCommitResult(
    int ImportedCount,
    int QuarantinedCount,
    string? QuarantinePath,
    int SkippedDuplicateCount = 0
);
