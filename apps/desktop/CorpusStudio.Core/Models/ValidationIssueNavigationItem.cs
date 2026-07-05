namespace CorpusStudio.Desktop.Models;

public sealed class ValidationIssueNavigationItem
{
    public ValidationIssueNavigationItem(ValidationIssue issue)
    {
        Level = string.IsNullOrWhiteSpace(issue.Level) ? "error" : issue.Level;
        Message = issue.Message;
        RowNumber = issue.RowNumber;
        Field = issue.Field;

        var location = RowNumber is null ? "" : $"Row {RowNumber}: ";
        var field = string.IsNullOrWhiteSpace(Field) ? "" : $" [{Field}]";
        DisplayName = $"{Level}: {location}{Message}{field}";
    }

    public string DisplayName { get; }

    public string Level { get; }

    public string Message { get; }

    public int? RowNumber { get; }

    public string? Field { get; }
}
