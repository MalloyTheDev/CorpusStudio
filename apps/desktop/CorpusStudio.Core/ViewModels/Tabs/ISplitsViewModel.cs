using System;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Splits tab's own view-model (Phase-2 decomposition). Self-contained: the
/// train/validation/seed inputs (persisted per project) and the split summary/report pane. The
/// engine owns split generation; this holds the tab's display + input state.
///
/// <para>A split failure must still surface in the shell's shared error banner, but the tab holds no
/// reference to the shell — so it raises <see cref="ErrorReported"/>, which the shell wires to its
/// <c>ReportError</c>. The shell also pushes the loaded settings (<see cref="ApplySplitSettings"/>)
/// and forwards <see cref="Reset"/> on project switch. Behind an interface so the shell/tests/DI
/// depend on the contract.</para></summary>
public interface ISplitsViewModel : INotifyPropertyChanged
{
    string SplitSummary { get; }
    string SplitTrainPercent { get; set; }
    string SplitValidationPercent { get; set; }
    string SplitSeed { get; set; }

    /// <summary>Test share (percent text), derived from the train/validation inputs as
    /// <c>100 − train − val</c>; read-only so the fourth Configure-split box mirrors the design
    /// without a fourth editable field. "—" when the inputs don't parse.</summary>
    string SplitTestPercent { get; }

    // --- discrete result signals (populate the result card's proportion bar + counts footer;
    //     bound directly instead of parsed out of the flattened SplitSummary text). ---

    /// <summary>True once a real report has been applied; gates the proportion bar / chip / footer
    /// so the pre-run card shows a neutral empty state (never fabricated ratios/counts).</summary>
    bool HasReport { get; }

    int TrainCount { get; }
    int ValidationCount { get; }
    int TestCount { get; }

    double TrainRatio { get; }
    double ValidationRatio { get; }
    double TestRatio { get; }

    int RowsSharedAcrossSplits { get; }

    /// <summary><c>RowsSharedAcrossSplits &gt; 0</c> — real train/test leakage.</summary>
    bool HasLeakage { get; }

    /// <summary>Show the Ok-tinted "No leakage" chip (a report exists and nothing is shared).</summary>
    bool ShowNoLeakageChip { get; }

    /// <summary>Show the Bad-tinted "Leakage" chip (a report exists and rows are shared).</summary>
    bool ShowLeakageChip { get; }

    /// <summary>Right-aligned footer detail, e.g. "checked all pairs across splits — 0 shared rows".</summary>
    string SharedRowsDetail { get; }

    /// <summary>Joined engine warnings (one per line), or empty when the report is clean.</summary>
    string SplitWarnings { get; }

    bool HasWarnings { get; }

    /// <summary>Raised when a split operation fails; the shell forwards it to its shared error
    /// banner (the tab keeps no shell reference).</summary>
    event Action<string>? ErrorReported;

    void SetSplitInProgress(double trainRatio, double validationRatio, int seed);
    void ApplySplitSettings(SplitSettings settings);
    void ApplySplitReport(SplitReport report);
    void SetSplitError(string message);

    /// <summary>Reset the summary to the project-open pending state on a project switch. The shell
    /// pushes the new project's saved ratios separately via <see cref="ApplySplitSettings"/>.</summary>
    void Reset();
}
