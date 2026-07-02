using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class ProjectSearchViewModelTests
{
    private static DatasetProjectListItem Item(string id, string name, string schema = "instruction")
    {
        var project = new DatasetProject(id, name, schema, new DateTime(2026, 1, 1), new DateTime(2026, 1, 1));
        return new DatasetProjectListItem(project, $"C:/projects/{id}");
    }

    [Fact]
    public void SetProjects_WithNoSearch_ShowsAll()
    {
        var vm = new MainWindowViewModel();
        vm.SetProjects(new[] { Item("a", "Alpha"), Item("b", "Beta") });
        Assert.Equal(2, vm.Projects.Count);
    }

    [Fact]
    public void Search_FiltersByName_CaseInsensitive()
    {
        var vm = new MainWindowViewModel();
        vm.SetProjects(new[] { Item("a", "Alpha"), Item("b", "Beta") });
        vm.ProjectSearch = "alp";
        Assert.Single(vm.Projects);
        Assert.Equal("Alpha", vm.Projects[0].Name);
    }

    [Fact]
    public void Search_FiltersById()
    {
        var vm = new MainWindowViewModel();
        vm.SetProjects(new[] { Item("coding_tutor", "Alpha"), Item("math_drills", "Beta") });
        vm.ProjectSearch = "math";
        Assert.Single(vm.Projects);
        Assert.Equal("math_drills", vm.Projects[0].Id);
    }

    [Fact]
    public void Search_FiltersBySchema()
    {
        var vm = new MainWindowViewModel();
        vm.SetProjects(new[] { Item("a", "Alpha", "instruction"), Item("b", "Beta", "chat") });
        vm.ProjectSearch = "chat";
        Assert.Single(vm.Projects);
        Assert.Equal("b", vm.Projects[0].Id);
    }

    [Fact]
    public void ClearingSearch_RestoresAll()
    {
        var vm = new MainWindowViewModel();
        vm.SetProjects(new[] { Item("a", "Alpha"), Item("b", "Beta") });
        vm.ProjectSearch = "alpha";
        Assert.Single(vm.Projects);
        vm.ProjectSearch = "";
        Assert.Equal(2, vm.Projects.Count);
    }

    [Fact]
    public void Search_AppliesToProjectsSetAfterSearch()
    {
        var vm = new MainWindowViewModel();
        vm.ProjectSearch = "beta";
        vm.SetProjects(new[] { Item("a", "Alpha"), Item("b", "Beta") });
        Assert.Single(vm.Projects);
        Assert.Equal("Beta", vm.Projects[0].Name);
    }
}
