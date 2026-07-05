using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Evaluation Suites tab's own view-model (Phase-2 decomposition). Self-contained: the
/// registered-suite list + selection, the last run's per-metric roll-up + per-case rows, the overall
/// verdict, and the busy flag. The engine owns list/new/run; this holds the tab's display state and
/// the honest verdict framing (per-metric never folded; a run makes live backend calls).
///
/// <para>The Run gate <see cref="CanRunSuite"/> depends on whether a project is open — a piece of
/// shell state — so the shell pushes <see cref="HasActiveProject"/> down (and forwards
/// <see cref="Reset"/>) on project switch. Behind an interface so the shell/tests/DI depend on the
/// contract.</para></summary>
public interface ISuitesViewModel : INotifyPropertyChanged
{
    ObservableCollection<SuiteSummary> Suites { get; }
    ObservableCollection<SuiteMetricRollup> SuiteMetricRows { get; }
    ObservableCollection<SuiteCaseResult> SuiteCaseRows { get; }
    string SuiteHonestyNote { get; }

    SuiteSummary? SelectedSuite { get; set; }
    bool IsSuitesBusy { get; set; }

    /// <summary>Mirror of the shell's HasActiveProject, pushed on project switch. Feeds
    /// <see cref="CanRunSuite"/> (Run needs an open project).</summary>
    bool HasActiveProject { get; set; }

    /// <summary>Run is enabled for a valid selected suite when a project is open and not busy.</summary>
    bool CanRunSuite { get; }

    bool HasSuiteReport { get; }
    string SuitesStatus { get; }
    string SuiteReportSummary { get; }
    string SuiteOverallStatus { get; }
    string SuiteOverallColor { get; }

    void ApplySuites(IReadOnlyList<SuiteSummary> summaries);
    void ApplySuiteReport(SuiteReport report);
    void SetSuitesError(string message);

    /// <summary>Reset all suite state on a project switch (list, selection, report, panes).</summary>
    void Reset();
}
