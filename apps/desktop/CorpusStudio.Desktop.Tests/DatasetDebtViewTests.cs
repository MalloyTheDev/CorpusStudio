using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class DatasetDebtViewTests
{
    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new DateTime(2026, 1, 1), new DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    private static DebtItem Item(string severity = "critical", string category = "secrets", int count = 1, double? rate = null)
        => new() { Category = category, Severity = severity, Count = count, Rate = rate, Message = "m", Remediation = "fix it" };

    private static DebtReport Report(string grade, bool hasData, params DebtItem[] items)
        => new() { Grade = grade, HasData = hasData, ExampleCount = hasData ? 10 : 0, Items = [.. items] };

    // --- parse + mapping ---------------------------------------------------

    [Fact]
    public void ParseDebtReport_ReadsGradeAndItems()
    {
        const string json = """
            {"example_count": 10, "has_data": true, "grade": "F",
             "items": [{"category": "secrets", "severity": "critical", "count": 1, "rate": null, "message": "leaked key", "remediation": "redact"}]}
            """;
        var report = PythonEngineService.ParseDebtReport(json);
        Assert.Equal("F", report.Grade);
        Assert.True(report.HasData);
        Assert.Single(report.Items);
        Assert.Equal("secrets", report.Items[0].Category);
        Assert.Null(report.Items[0].Rate);
    }

    [Theory]
    [InlineData("A", "#16A34A")]
    [InlineData("F", "#DC2626")]
    [InlineData("N/A", "#64748B")]
    [InlineData("—", "#64748B")]
    public void GradeColor_MapsGrades_NaNeverGreen(string grade, string expected)
    {
        Assert.Equal(expected, DebtReport.GradeColor(grade));
        Assert.NotEqual("#16A34A", DebtReport.GradeColor("N/A"));  // never green
    }

    [Fact]
    public void DebtDisplayItem_BadgeAndMeasure()
    {
        var rated = new DebtDisplayItem(Item("high", "exact_duplicates", 6, 0.6));
        Assert.Contains("HIGH", rated.SeverityBadge);
        Assert.Contains("60.0%", rated.Measure);
        Assert.Contains("(6)", rated.Measure);

        var presence = new DebtDisplayItem(Item("critical", "secrets", 1, null));
        Assert.Contains("CRITICAL", presence.SeverityBadge);
        Assert.Equal("count 1", presence.Measure);

        // slice-5 fidelity: the severity chip color (Nocturne semantics).
        Assert.Equal("#d76d6d", rated.SeverityColor);      // high  -> bad red
        Assert.Equal("#d76d6d", presence.SeverityColor);   // critical -> bad red
        Assert.Equal("#d9a35f", new DebtDisplayItem(Item("moderate", "near_dupes", 2, 0.1)).SeverityColor);
        Assert.Equal("#9397ab", new DebtDisplayItem(Item("low", "imbalance", 3, null)).SeverityColor);
    }

    // --- ApplyDebtReport honest summaries ----------------------------------

    [Fact]
    public void ApplyDebtReport_CriticalIsGradeFRedAndListed()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("F", true, Item("critical", "secrets", 1, null)));
        Assert.Equal("F", vm.Debt.DebtGrade);
        Assert.Equal("#DC2626", vm.Debt.DebtGradeColor);
        Assert.Single(vm.Debt.DebtItems);
        Assert.False(vm.Debt.DebtStale);
        Assert.Contains("Grade F", vm.Debt.DebtSummary);
    }

    [Fact]
    public void ApplyDebtReport_CleanIsGradeANoDebt()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("A", true));
        Assert.Equal("A", vm.Debt.DebtGrade);
        Assert.Equal("#16A34A", vm.Debt.DebtGradeColor);
        Assert.Empty(vm.Debt.DebtItems);
        Assert.Contains("no debt", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ApplyDebtReport_EmptyIsNaNotClean()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("N/A", false));
        Assert.Equal("N/A", vm.Debt.DebtGrade);
        Assert.NotEqual("#16A34A", vm.Debt.DebtGradeColor);  // never green
        Assert.Contains("No rows to assess", vm.Debt.DebtSummary);
        Assert.DoesNotContain("no debt", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    // --- the non-negotiable honesty rule: invalidate on dataset change -----

    [Fact]
    public void SetExamples_InvalidatesAPriorDebtGrade()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("F", true, Item()));
        Assert.Equal("F", vm.Debt.DebtGrade);

        vm.SetExamples(new List<SavedExampleItem>());  // the dataset changed

        Assert.Equal("—", vm.Debt.DebtGrade);
        Assert.Empty(vm.Debt.DebtItems);
        Assert.True(vm.Debt.DebtStale);
        Assert.Contains("changed", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void InvalidateDebt_WithoutPriorGrade_IsNeutralNotScary()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.InvalidateDebt();  // no grade was ever shown
        Assert.False(vm.Debt.DebtStale);
        Assert.DoesNotContain("changed", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SelectProject_ClearsDebtToNeutralDefault()
    {
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("F", true, Item()));
        vm.SelectProject(Project("other"));
        Assert.Equal("—", vm.Debt.DebtGrade);
        Assert.Empty(vm.Debt.DebtItems);
        Assert.False(vm.Debt.DebtStale);
        Assert.DoesNotContain("changed", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    // --- audit fixes -------------------------------------------------------

    [Fact]
    public void SetDebtError_CollapsesGradeToNeutral()
    {
        // A failed check must not leave a confident colored grade on screen.
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("A", true));  // big green A shown
        vm.Debt.SetDebtError("python not found");
        Assert.Equal("—", vm.Debt.DebtGrade);
        Assert.Equal("#64748B", vm.Debt.DebtGradeColor);  // gray, not green
        Assert.Empty(vm.Debt.DebtItems);
        Assert.Contains("failed", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Measure_UsesInvariantCulture()
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");  // comma decimal
            var item = new DebtDisplayItem(Item("high", "exact_duplicates", 6, 0.6));
            Assert.Equal("60.0% (6)", item.Measure);  // never "60,0%"
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    [Fact]
    public async Task GetDatasetDebtAsync_MissingExamples_ReturnsNa()
    {
        // A project with no examples.jsonl degrades to N/A (no engine call, no raw error).
        using var dir = new TempProjectDirectory();
        var report = await new PythonEngineService().GetDatasetDebtAsync(dir.Path);
        Assert.False(report.HasData);
        Assert.Equal("N/A", report.Grade);
        Assert.Empty(report.Items);
    }

    [Fact]
    public void ProjectLoadOrdering_LeavesNeutralNotChanged()
    {
        // LoadProjectAsync order: SelectProject (resets debt) then SetExamples
        // (invalidates, but no grade is shown post-reset, so it stays neutral).
        var vm = new MainWindowViewModel();
        vm.Debt.ApplyDebtReport(Report("F", true, Item()));   // a grade from the prior project
        vm.SelectProject(Project("other"));
        vm.SetExamples(new List<SavedExampleItem>());
        Assert.Equal("—", vm.Debt.DebtGrade);
        Assert.False(vm.Debt.DebtStale);
        Assert.DoesNotContain("changed", vm.Debt.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }
}
