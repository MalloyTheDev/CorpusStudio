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
