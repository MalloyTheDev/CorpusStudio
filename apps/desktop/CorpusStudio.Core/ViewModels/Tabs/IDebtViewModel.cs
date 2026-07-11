using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Dataset Debt tab's own view-model, extracted from the shell as the first step of
/// the god-object decomposition (backlog #4). The shell hosts it as a child and forwards the
/// cross-cutting lifecycle: <see cref="Reset"/> on project switch, <see cref="InvalidateDebt"/>
/// when the dataset changes. Kept behind an interface so the shell/tests/DI depend on the
/// contract, not the concrete class.</summary>
public interface IDebtViewModel : INotifyPropertyChanged
{
    ObservableCollection<DebtDisplayItem> DebtItems { get; }
    string DebtGrade { get; }
    string DebtGradeColor { get; }
    string DebtSummary { get; }
    bool DebtStale { get; }

    /// <summary>Whether a real letter grade has been computed (not the neutral "—" placeholder).</summary>
    bool HasGrade { get; }

    /// <summary>The bold hero verdict headline derived honestly from the real grade (A/B ready to
    /// train, C/D/F not, neutral "—"/N/A prompts a check).</summary>
    string DebtVerdict { get; }

    /// <summary>Show a fresh debt result (grade + ranked ledger + honest summary).</summary>
    void ApplyDebtReport(DebtReport report);

    /// <summary>Collapse to the neutral unknown state after a failed check (never keep a
    /// confident colored grade under a failure).</summary>
    void SetDebtError(string message);

    /// <summary>Invalidate the grade because the dataset changed (stale if a grade was shown,
    /// else the neutral default).</summary>
    void InvalidateDebt();

    /// <summary>Reset to the clean per-project default (project switch).</summary>
    void Reset();
}
