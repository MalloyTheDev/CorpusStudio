using System;
using System.Collections.Generic;
using System.Globalization;
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

    private bool _hasReport;
    private int _trainCount;
    private int _validationCount;
    private int _testCount;
    private double _trainRatio;
    private double _validationRatio;
    private double _testRatio;
    private int _rowsSharedAcrossSplits;
    private string _splitWarnings = string.Empty;

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
        set
        {
            if (SetField(ref _splitTrainPercent, value))
            {
                OnPropertyChanged(nameof(SplitTestPercent));
            }
        }
    }

    public string SplitValidationPercent
    {
        get => _splitValidationPercent;
        set
        {
            if (SetField(ref _splitValidationPercent, value))
            {
                OnPropertyChanged(nameof(SplitTestPercent));
            }
        }
    }

    public string SplitSeed
    {
        get => _splitSeed;
        set => SetField(ref _splitSeed, value);
    }

    /// <summary>Test share (percent text), derived from the train/validation inputs as
    /// <c>100 − train − val</c>. Returns "—" when either input can't be parsed, and clamps a
    /// negative remainder to 0 so an over-100 configuration never shows a fabricated negative.</summary>
    public string SplitTestPercent
    {
        get
        {
            if (!double.TryParse(_splitTrainPercent, NumberStyles.Any, CultureInfo.CurrentCulture, out var train)
                || !double.TryParse(_splitValidationPercent, NumberStyles.Any, CultureInfo.CurrentCulture, out var validation))
            {
                return "—";
            }

            var test = 100 - train - validation;
            if (test < 0)
            {
                test = 0;
            }

            return test.ToString("0.##", CultureInfo.CurrentCulture);
        }
    }

    public bool HasReport
    {
        get => _hasReport;
        private set
        {
            if (SetField(ref _hasReport, value))
            {
                OnPropertyChanged(nameof(ShowNoLeakageChip));
                OnPropertyChanged(nameof(ShowLeakageChip));
            }
        }
    }

    public int TrainCount
    {
        get => _trainCount;
        private set => SetField(ref _trainCount, value);
    }

    public int ValidationCount
    {
        get => _validationCount;
        private set => SetField(ref _validationCount, value);
    }

    public int TestCount
    {
        get => _testCount;
        private set => SetField(ref _testCount, value);
    }

    public double TrainRatio
    {
        get => _trainRatio;
        private set => SetField(ref _trainRatio, value);
    }

    public double ValidationRatio
    {
        get => _validationRatio;
        private set => SetField(ref _validationRatio, value);
    }

    public double TestRatio
    {
        get => _testRatio;
        private set => SetField(ref _testRatio, value);
    }

    public int RowsSharedAcrossSplits
    {
        get => _rowsSharedAcrossSplits;
        private set
        {
            if (SetField(ref _rowsSharedAcrossSplits, value))
            {
                OnPropertyChanged(nameof(HasLeakage));
                OnPropertyChanged(nameof(ShowNoLeakageChip));
                OnPropertyChanged(nameof(ShowLeakageChip));
                OnPropertyChanged(nameof(SharedRowsDetail));
            }
        }
    }

    public bool HasLeakage => _rowsSharedAcrossSplits > 0;

    public bool ShowNoLeakageChip => _hasReport && !HasLeakage;

    public bool ShowLeakageChip => _hasReport && HasLeakage;

    public string SharedRowsDetail =>
        $"checked all pairs across splits — {_rowsSharedAcrossSplits} shared rows";

    public string SplitWarnings
    {
        get => _splitWarnings;
        private set
        {
            if (SetField(ref _splitWarnings, value))
            {
                OnPropertyChanged(nameof(HasWarnings));
            }
        }
    }

    public bool HasWarnings => _splitWarnings.Length > 0;

    public void SetSplitInProgress(double trainRatio, double validationRatio, int seed)
    {
        // Generation started — drop back to the neutral card (no bar/chip/footer) until the real
        // report lands, so nothing stale from a prior run is shown as if current.
        HasReport = false;
        SplitWarnings = string.Empty;
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

        // Discrete signals for the result card's proportion bar + counts footer (bound directly,
        // rather than re-parsed from the flattened SplitSummary text above).
        TrainCount = report.Train;
        ValidationCount = report.Validation;
        TestCount = report.Test;
        TrainRatio = report.TrainRatio;
        ValidationRatio = report.ValidationRatio;
        TestRatio = report.TestRatio;
        RowsSharedAcrossSplits = report.RowsSharedAcrossSplits;
        SplitWarnings = report.Warnings.Count > 0
            ? string.Join(Environment.NewLine, report.Warnings.Select(warning => $"- {warning}"))
            : string.Empty;
        HasReport = true;
    }

    public void SetSplitError(string message)
    {
        HasReport = false;
        SplitWarnings = string.Empty;
        SplitSummary = $"Splits could not be generated.{Environment.NewLine}{message}";
        ErrorReported?.Invoke(message);
    }

    /// <summary>Reset the summary to the project-open pending state on a project switch. The shell
    /// pushes the new project's saved ratios separately via <see cref="ApplySplitSettings"/>.</summary>
    public void Reset()
    {
        HasReport = false;
        SplitWarnings = string.Empty;
        SplitSummary = PendingSummary;
    }

    private static string FormatPercent(double ratio)
    {
        return $"{ratio * 100:0.##}%";
    }
}
