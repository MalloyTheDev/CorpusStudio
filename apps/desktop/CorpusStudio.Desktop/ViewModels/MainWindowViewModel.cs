using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels;

public sealed class MainWindowViewModel : INotifyPropertyChanged
{
    private string _activeProjectTitle = "New Dataset Project";
    private string? _activeProjectPath;
    private string _activeSchemaId = "instruction";
    private string _activeSchemaDescription =
        "Choose a schema, write examples, validate rows, and export model-ready JSONL.";
    private string _validationSummary = "Create a project to start validation.";
    private string _qualitySummary =
        "Quality dashboard placeholder: duplicates, token length, missing fields, and split leakage.";
    private string _settingsSummary = "Settings load when the app starts.";

    public event PropertyChangedEventHandler? PropertyChanged;

    public ObservableCollection<string> Projects { get; } = [];

    public ObservableCollection<string> Examples { get; } = [];

    public string ActiveProjectTitle
    {
        get => _activeProjectTitle;
        private set => SetField(ref _activeProjectTitle, value);
    }

    public string ActiveSchemaDescription
    {
        get => _activeSchemaDescription;
        private set => SetField(ref _activeSchemaDescription, value);
    }

    public string ActiveSchemaId
    {
        get => _activeSchemaId;
        private set => SetField(ref _activeSchemaId, value);
    }

    public string? ActiveProjectPath
    {
        get => _activeProjectPath;
        private set => SetField(ref _activeProjectPath, value);
    }

    public bool HasActiveProject => !string.IsNullOrWhiteSpace(ActiveProjectPath);

    private string _draftText =
        "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}";

    public string DraftText
    {
        get => _draftText;
        set => SetField(ref _draftText, value);
    }

    public string ValidationSummary
    {
        get => _validationSummary;
        private set => SetField(ref _validationSummary, value);
    }

    public string QualitySummary
    {
        get => _qualitySummary;
        private set => SetField(ref _qualitySummary, value);
    }

    public string SettingsSummary
    {
        get => _settingsSummary;
        private set => SetField(ref _settingsSummary, value);
    }

    public void AddProject(string projectId, string name, string schemaName)
    {
        AddProject(projectId, name, "instruction", schemaName, null);
    }

    public void SetProjects(IEnumerable<DatasetProject> projects)
    {
        Projects.Clear();
        foreach (var project in projects)
        {
            Projects.Add($"{project.Name} ({project.Id})");
        }
    }

    public void SetSettings(DesktopSettings settings)
    {
        SettingsSummary = string.Join(
            Environment.NewLine,
            [
                $"Repository: {settings.RepositoryRoot}",
                $"Engine: {settings.EngineDirectory}",
                $"Python: {settings.PythonExecutable}",
                $"Projects: {settings.ProjectDirectory}",
                $"Exports: {settings.ExportDirectory}",
            ]
        );
    }

    public void AddProject(
        string projectId,
        string name,
        string schemaId,
        string schemaName,
        string? projectPath
    )
    {
        Projects.Add($"{name} ({projectId})");
        ActiveProjectTitle = name;
        ActiveProjectPath = projectPath;
        ActiveSchemaId = schemaId;
        ActiveSchemaDescription = $"{schemaName} project. Ready for examples.";
        ValidationSummary = "No validation has run yet.";
        QualitySummary = "Quality checks will appear after examples are added.";
        DraftText = BuildDraftTemplate(schemaId);
        Examples.Clear();
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(HasActiveProject)));
    }

    public void AddSavedExamples(int count)
    {
        for (var index = 0; index < count; index++)
        {
            Examples.Add($"Example {Examples.Count + 1}");
        }

        QualitySummary = $"{Examples.Count} saved example(s). Quality checks will appear here next.";
    }

    public void SetValidationInProgress()
    {
        ValidationSummary = "Running validation...";
    }

    public void ApplyValidationReport(ValidationReport report)
    {
        var status = report.Valid ? "Valid" : "Invalid";
        var lines = new List<string>
        {
            $"{status}: {report.CheckedRows} row(s) checked against `{report.SchemaId}`.",
        };

        if (report.Errors.Count > 0)
        {
            lines.Add("");
            lines.Add("Errors:");
            lines.AddRange(report.Errors.Select(FormatIssue));
        }

        if (report.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(report.Warnings.Select(FormatIssue));
        }

        ValidationSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetValidationError(string message)
    {
        ValidationSummary = $"Validation could not run.{Environment.NewLine}{message}";
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        return true;
    }

    private static string FormatIssue(ValidationIssue issue)
    {
        var location = issue.RowNumber is null ? "" : $"Row {issue.RowNumber}: ";
        var field = string.IsNullOrWhiteSpace(issue.Field) ? "" : $" [{issue.Field}]";
        return $"- {location}{issue.Message}{field}";
    }

    private static string BuildDraftTemplate(string schemaId)
    {
        return schemaId switch
        {
            "raw_text" => "{\n  \"text\": \"A compiler translates source code into machine instructions.\"\n}",
            "chat" => "{\n  \"messages\": [\n    {\"role\": \"user\", \"content\": \"What is recursion?\"},\n    {\"role\": \"assistant\", \"content\": \"Recursion is when a function calls itself.\"}\n  ]\n}",
            "preference" => "{\n  \"prompt\": \"Explain recursion simply.\",\n  \"chosen\": \"Recursion is when a function calls itself.\",\n  \"rejected\": \"Recursion is when code does things again.\"\n}",
            _ => "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}",
        };
    }
}
