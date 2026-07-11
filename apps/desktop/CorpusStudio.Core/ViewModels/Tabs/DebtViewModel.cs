using System;
using System.Collections.ObjectModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Dataset Debt tab view-model (backlog #4 decomposition proof). Behaviour is
/// moved verbatim from the shell; honesty invariants unchanged (N/A / stale is neutral gray, never
/// green; a failed check collapses to neutral; a fresh project reads "run a debt check").</summary>
public sealed class DebtViewModel : ViewModelBase, IDebtViewModel
{
    private const string NeutralGray = "#64748B";
    private const string DefaultSummary = "Run a debt check to grade the current dataset.";

    private string _debtGrade = "—";
    private string _debtGradeColor = NeutralGray;
    private string _debtSummary = DefaultSummary;
    private bool _debtStale;

    public ObservableCollection<DebtDisplayItem> DebtItems { get; } = [];

    public string DebtGrade
    {
        get => _debtGrade;
        private set
        {
            if (SetField(ref _debtGrade, value))
            {
                OnPropertyChanged(nameof(HasGrade));
            }
        }
    }

    /// <summary>Whether a real letter grade has been computed (not the neutral "—" placeholder).
    /// Drives the sidebar nav-row grade pill so it stays hidden until a debt check has run.</summary>
    public bool HasGrade => _debtGrade != "—";

    public string DebtGradeColor
    {
        get => _debtGradeColor;
        private set => SetField(ref _debtGradeColor, value);
    }

    public string DebtSummary
    {
        get => _debtSummary;
        private set => SetField(ref _debtSummary, value);
    }

    public bool DebtStale
    {
        get => _debtStale;
        private set => SetField(ref _debtStale, value);
    }

    public void ApplyDebtReport(DebtReport report)
    {
        DebtItems.Clear();
        foreach (var item in report.Items)
        {
            DebtItems.Add(new DebtDisplayItem(item));
        }
        DebtGrade = report.Grade;
        DebtGradeColor = DebtReport.GradeColor(report.Grade);
        DebtStale = false;
        DebtSummary = !report.HasData
            ? "No rows to assess (grade N/A). Add examples, then run a debt check."
            : report.Items.Count == 0
                ? $"Grade {report.Grade} — no debt detected. This dataset is clean by the current checks."
                : $"Grade {report.Grade}: {report.Items.Count} item(s), highest-severity first. "
                  + "Fix the top items before training.";
    }

    public void SetDebtError(string message)
    {
        DebtItems.Clear();
        DebtGrade = "—";
        DebtGradeColor = NeutralGray;
        DebtStale = false;
        DebtSummary = $"Debt check failed.{Environment.NewLine}{message}";
    }

    public void InvalidateDebt()
    {
        var hadGrade = DebtItems.Count > 0 || DebtGrade != "—";
        DebtItems.Clear();
        DebtGrade = "—";
        DebtGradeColor = NeutralGray;
        if (hadGrade)
        {
            DebtStale = true;
            DebtSummary = "The dataset changed — run a debt check to re-grade it.";
        }
        else
        {
            DebtStale = false;
            DebtSummary = DefaultSummary;
        }
    }

    public void Reset()
    {
        DebtItems.Clear();
        DebtGrade = "—";
        DebtGradeColor = NeutralGray;
        DebtStale = false;
        DebtSummary = DefaultSummary;
    }
}
