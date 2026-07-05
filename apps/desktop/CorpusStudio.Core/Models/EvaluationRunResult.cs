namespace CorpusStudio.Desktop.Models;

public sealed record EvaluationRunResult(
    EvaluationReport Report,
    string ReportPath,
    string ReportJson
);
