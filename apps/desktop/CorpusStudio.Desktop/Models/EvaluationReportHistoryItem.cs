namespace CorpusStudio.Desktop.Models;

public sealed class EvaluationReportHistoryItem
{
    public EvaluationReportHistoryItem(
        EvaluationReport report,
        string reportPath,
        string reportJson,
        DateTime lastModified
    )
    {
        Report = report;
        ReportPath = reportPath;
        ReportJson = reportJson;
        LastModified = lastModified;
    }

    public EvaluationReport Report { get; }

    public string ReportPath { get; }

    public string ReportJson { get; }

    public DateTime LastModified { get; }

    public string DisplayName =>
        $"{LastModified:yyyy-MM-dd HH:mm} | {Report.Model} | {Report.ExamplesTested} ex | avg {Report.AverageScore:0.##}";
}
