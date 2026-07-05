using System;
using System.Collections.Generic;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Splits tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>); honesty invariants unchanged — the report names rows shared across
/// splits and flags them as train/test leakage, and surfaces the engine's warnings.</summary>
public sealed class SplitsViewModel : ViewModelBase, ISplitsViewModel
{
    private const string DefaultSummary =
        "Create or select a project to generate train, validation, and test splits.";
    private const string PendingSummary = "Generate splits after examples are saved.";

    private string _splitSummary = DefaultSummary;
    private string _splitTrainPercent = "90";
    private string _splitValidationPercent = "5";
    private string _splitSeed = "42";

    /// <summary>Raised when a split operation fails, so the shell can surface it in the shared
    /// error banner (the tab holds no reference to the shell).</summary>
    public event Action<string>? ErrorReported;

    public string SplitSummary
    {
        get => _splitSummary;
        private set => SetField(ref _splitSummary, value);
    }

    public string SplitTrainPercent
    {
        get => _splitTrainPercent;
        set => SetField(ref _splitTrainPercent, value);
    }

    public string SplitValidationPercent
    {
        get => _splitValidationPercent;
        set => SetField(ref _splitValidationPercent, value);
    }

    public string SplitSeed
    {
        get => _splitSeed;
        set => SetField(ref _splitSeed, value);
    }

    public void SetSplitInProgress(double trainRatio, double validationRatio, int seed)
    {
        var testRatio = 1 - trainRatio - validationRatio;
        SplitSummary = string.Join(
            Environment.NewLine,
            [
                "Generating train, validation, and test splits...",
                $"Train: {FormatPercent(trainRatio)}",
                $"Validation: {FormatPercent(validationRatio)}",
                $"Test: {FormatPercent(testRatio)}",
                $"Seed: {seed}",
            ]
        );
    }

    public void ApplySplitSettings(SplitSettings settings)
    {
        SplitTrainPercent = settings.TrainPercentText;
        SplitValidationPercent = settings.ValidationPercentText;
        SplitSeed = settings.Seed.ToString();
    }

    public void ApplySplitReport(SplitReport report)
    {
        var lines = new List<string>
        {
            $"Train: {report.Train}",
            $"Validation: {report.Validation}",
            $"Test: {report.Test}",
            $"Ratios: train {FormatPercent(report.TrainRatio)}, validation {FormatPercent(report.ValidationRatio)}, test {FormatPercent(report.TestRatio)}",
            $"Seed: {report.Seed}",
            $"Rows shared across splits: {report.RowsSharedAcrossSplits}"
                + (report.RowsSharedAcrossSplits > 0 ? " (train/test leakage)" : ""),
            $"Output: {report.OutputDirectory}",
        };

        if (report.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(report.Warnings.Select(warning => $"- {warning}"));
        }

        SplitSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetSplitError(string message)
    {
        SplitSummary = $"Splits could not be generated.{Environment.NewLine}{message}";
        ErrorReported?.Invoke(message);
    }

    /// <summary>Reset the summary to the project-open pending state on a project switch. The shell
    /// pushes the new project's saved ratios separately via <see cref="ApplySplitSettings"/>.</summary>
    public void Reset()
    {
        SplitSummary = PendingSummary;
    }

    private static string FormatPercent(double ratio)
    {
        return $"{ratio * 100:0.##}%";
    }
}
