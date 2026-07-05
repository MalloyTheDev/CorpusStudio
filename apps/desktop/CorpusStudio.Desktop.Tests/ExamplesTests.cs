using System.Collections.Generic;
using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Saved Examples tab view-model (Phase-2 extraction, the dataset spine). List/selection/JSON
/// pane in isolation, plus the shell's SetExamples orchestrator that rebuilds the tab and fans the
/// dataset change out to Preference/Quality/Debt, and the per-project reset.</summary>
public sealed class ExamplesTests
{
    private static SavedExampleItem Example(int row, string output = "out")
        => new(row, $"preview {row}", $"{{\"instruction\":\"i{row}\",\"input\":\"\",\"output\":\"{output}\"}}");

    private static DatasetProjectListItem Project(string id = "p1")
        => new(
            new DatasetProject(id, id, "instruction", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            $"C:/projects/{id}");

    // --- isolated display core --------------------------------------------------------

    [Fact]
    public void SetItems_PopulatesSelectsFirstAndShowsJson()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems([Example(1), Example(2)]);
        Assert.Equal(2, vm.Items.Count);
        Assert.Same(vm.Items[0], vm.SelectedExample);
        Assert.Equal(vm.Items[0].Json, vm.SelectedExampleJson);
    }

    [Fact]
    public void SetItems_EmptyShowsHonestNoExamplesMessage()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems([]);
        Assert.Empty(vm.Items);
        Assert.Null(vm.SelectedExample);
        Assert.Contains("No saved examples yet", vm.SelectedExampleJson);
        Assert.Contains("Writing Studio", vm.SelectedExampleJson);
    }

    [Fact]
    public void SelectingRow_ShowsItsJson_NullClearsToPrompt()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems([Example(1), Example(2)]);
        vm.SelectedExample = vm.Items[1];
        Assert.Equal(vm.Items[1].Json, vm.SelectedExampleJson);

        vm.SelectedExample = null;
        Assert.Contains("Select a saved example to inspect its JSON", vm.SelectedExampleJson);
    }

    [Fact]
    public void Reset_ClearsListAndSelection()
    {
        var vm = new ExamplesViewModel();
        vm.SetItems([Example(1)]);
        vm.Reset();
        Assert.Empty(vm.Items);
        Assert.Null(vm.SelectedExample);
    }

    // --- shell orchestrator: SetExamples fans the dataset change out ------------------

    [Fact]
    public void SetExamples_RebuildsTab_UpdatesQuality_AndInvalidatesDebt()
    {
        var vm = new MainWindowViewModel();
        // A prior debt grade that must be invalidated when the dataset changes.
        vm.Debt.ApplyDebtReport(new DebtReport { Grade = "F", HasData = true, ExampleCount = 10 });
        Assert.False(vm.Debt.DebtStale);

        vm.SetExamples([Example(1), Example(2)]);

        Assert.Equal(2, vm.Examples.Items.Count);                  // Examples tab rebuilt
        Assert.Contains("2 saved example(s)", vm.Quality.QualitySummary);  // Quality summary updated
        Assert.True(vm.Debt.DebtStale);                            // Debt grade invalidated
    }

    [Fact]
    public void SetExamples_FeedsPreferencePairs_ForPreferenceProjects()
    {
        var vm = new MainWindowViewModel();
        vm.SelectProject(Project()); // pushes a non-preference schema; preference feed stays empty
        vm.SetExamples([Example(1)]);
        Assert.Empty(vm.PreferenceReview.PreferenceReviewItems); // non-preference -> no pairs built

        // Confirm the feed IS wired: a preference project builds pairs from the same examples.
        var pref = new MainWindowViewModel();
        pref.SelectProject(new DatasetProjectListItem(
            new DatasetProject("pp", "pp", "preference", new System.DateTime(2026, 1, 1), new System.DateTime(2026, 1, 1)),
            "C:/projects/pp"));
        pref.SetExamples(
        [
            new SavedExampleItem(1, "p", """{"prompt":"P","chosen":"alpha beta gamma","rejected":"xxxx yyyy zzzz"}"""),
        ]);
        Assert.Single(pref.PreferenceReview.PreferenceReviewItems);
    }

    [Fact]
    public void SelectProject_ResetsExamplesState()
    {
        var vm = new MainWindowViewModel();
        vm.SetExamples([Example(1), Example(2)]);
        Assert.Equal(2, vm.Examples.Items.Count);

        vm.SelectProject(Project("other"));

        Assert.Empty(vm.Examples.Items);
        Assert.Null(vm.Examples.SelectedExample);
    }
}
