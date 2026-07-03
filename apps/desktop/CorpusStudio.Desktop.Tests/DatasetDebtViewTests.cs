using System;
using System.Collections.Generic;
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
    }

    // --- ApplyDebtReport honest summaries ----------------------------------

    [Fact]
    public void ApplyDebtReport_CriticalIsGradeFRedAndListed()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("F", true, Item("critical", "secrets", 1, null)));
        Assert.Equal("F", vm.DebtGrade);
        Assert.Equal("#DC2626", vm.DebtGradeColor);
        Assert.Single(vm.DebtItems);
        Assert.False(vm.DebtStale);
        Assert.Contains("Grade F", vm.DebtSummary);
    }

    [Fact]
    public void ApplyDebtReport_CleanIsGradeANoDebt()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("A", true));
        Assert.Equal("A", vm.DebtGrade);
        Assert.Equal("#16A34A", vm.DebtGradeColor);
        Assert.Empty(vm.DebtItems);
        Assert.Contains("no debt", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ApplyDebtReport_EmptyIsNaNotClean()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("N/A", false));
        Assert.Equal("N/A", vm.DebtGrade);
        Assert.NotEqual("#16A34A", vm.DebtGradeColor);  // never green
        Assert.Contains("No rows to assess", vm.DebtSummary);
        Assert.DoesNotContain("no debt", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    // --- the non-negotiable honesty rule: invalidate on dataset change -----

    [Fact]
    public void SetExamples_InvalidatesAPriorDebtGrade()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("F", true, Item()));
        Assert.Equal("F", vm.DebtGrade);

        vm.SetExamples(new List<SavedExampleItem>());  // the dataset changed

        Assert.Equal("—", vm.DebtGrade);
        Assert.Empty(vm.DebtItems);
        Assert.True(vm.DebtStale);
        Assert.Contains("changed", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void InvalidateDebt_WithoutPriorGrade_IsNeutralNotScary()
    {
        var vm = new MainWindowViewModel();
        vm.InvalidateDebt();  // no grade was ever shown
        Assert.False(vm.DebtStale);
        Assert.DoesNotContain("changed", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SelectProject_ClearsDebtToNeutralDefault()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("F", true, Item()));
        vm.SelectProject(Project("other"));
        Assert.Equal("—", vm.DebtGrade);
        Assert.Empty(vm.DebtItems);
        Assert.False(vm.DebtStale);
        Assert.DoesNotContain("changed", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ProjectLoadOrdering_LeavesNeutralNotChanged()
    {
        // LoadProjectAsync order: SelectProject (resets debt) then SetExamples
        // (invalidates, but no grade is shown post-reset, so it stays neutral).
        var vm = new MainWindowViewModel();
        vm.ApplyDebtReport(Report("F", true, Item()));   // a grade from the prior project
        vm.SelectProject(Project("other"));
        vm.SetExamples(new List<SavedExampleItem>());
        Assert.Equal("—", vm.DebtGrade);
        Assert.False(vm.DebtStale);
        Assert.DoesNotContain("changed", vm.DebtSummary, StringComparison.OrdinalIgnoreCase);
    }
}
