using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>New Project wizard view-model (v1.2.4, slice 3c-2): auto-slug, the live
/// folder-structure preview per template, target-folder derivation, and validation. Pure —
/// no dialogs or engine.</summary>
public sealed class WorkspaceWizardViewModelTests
{
    private static DatasetSchema Schema(string id, string name, string? exampleJson = null)
    {
        JsonElement? example = exampleJson is null ? null : JsonSerializer.Deserialize<JsonElement>(exampleJson);
        return new DatasetSchema(id, name, "1", Array.Empty<DatasetField>(), null, example);
    }

    private static WorkspaceWizardViewModel Vm()
    {
        var schemas = new[]
        {
            Schema("instruction", "Instruction", "{\"instruction\":\"Explain a variable.\",\"input\":\"\",\"output\":\"A named value.\"}"),
            Schema("chat", "Chat"),
            Schema("image_caption", "Image Caption", "{\"image\":\"assets/images/x.jpg\",\"caption\":\"...\"}"),
            Schema("preference", "Preference"),
        };
        return new WorkspaceWizardViewModel(schemas);
    }

    private static void SelectTemplate(WorkspaceWizardViewModel vm, string id) =>
        vm.SelectedTemplate = vm.Templates.First(t => t.Id == id);

    [Fact]
    public void ProjectName_AutoSlugsProjectId_UntilEdited()
    {
        var vm = Vm();
        vm.ProjectName = "My Cool Dataset";
        Assert.Equal("my_cool_dataset", vm.ProjectId);

        vm.ProjectId = "custom_id";        // user takes over
        vm.ProjectName = "Another Name";   // no longer overwrites
        Assert.Equal("custom_id", vm.ProjectId);
    }

    [Fact]
    public void Preview_Empty_OnlyManifestDir_NoExamples()
    {
        var vm = Vm();
        SelectTemplate(vm, "empty");
        Assert.Contains(vm.PreviewLines, l => l.Text.StartsWith(".corpus"));
        Assert.DoesNotContain(vm.PreviewLines, l => l.Text.StartsWith("examples.jsonl"));
    }

    [Fact]
    public void Preview_Standard_HasWorkingDirsAndCard()
    {
        var vm = Vm();
        SelectTemplate(vm, "standard");
        Assert.Contains(vm.PreviewLines, l => l.Text == "splits/");
        Assert.Contains(vm.PreviewLines, l => l.Text == "reports/");
        Assert.Contains(vm.PreviewLines, l => l.Text == "dataset_card.json");
        Assert.Contains(vm.PreviewLines, l => l.Text == "examples.jsonl");
    }

    [Fact]
    public void Preview_Full_HasAssetKindsAndGeneratedDirs()
    {
        var vm = Vm();
        SelectTemplate(vm, "full");
        Assert.Contains(vm.PreviewLines, l => l.Text == "images/");          // assets/images
        Assert.Contains(vm.PreviewLines, l => l.Text == "model_artifacts/");
        Assert.Contains(vm.PreviewLines, l => l.Text == "dataset_versions/");
    }

    [Fact]
    public void SchemaStarter_Instruction_SeedsExamplesRow()
    {
        var vm = Vm();
        SelectTemplate(vm, "schema");
        vm.SelectedSchema = vm.Schemas.First(s => s.Id == "instruction");

        var examples = vm.BuildPlan().Files.Single(f => f.RelativePath == "examples.jsonl");
        Assert.Contains("instruction", examples.Content);
    }

    [Fact]
    public void SchemaStarter_ImageCaption_LeavesExamplesEmpty_SetsNote()
    {
        var vm = Vm();
        SelectTemplate(vm, "schema");
        vm.SelectedSchema = vm.Schemas.First(s => s.Id == "image_caption");

        Assert.Equal(string.Empty, vm.BuildPlan().Files.Single(f => f.RelativePath == "examples.jsonl").Content);
        Assert.True(vm.HasPreviewNote);
        Assert.NotNull(vm.PreviewNote);
    }

    [Fact]
    public void TargetFolder_IsLocationPlusSafeId()
    {
        var vm = Vm();
        vm.ProjectName = "Demo Set";
        vm.Location = Path.Combine("C:", "workspaces");
        Assert.Equal(Path.Combine("C:", "workspaces", "demo_set"), vm.TargetFolder);
    }

    [Fact]
    public void CanCreate_RequiresNameSchemaTemplateLocation()
    {
        var vm = Vm();
        Assert.False(vm.CanCreate);           // no name / location yet
        Assert.False(string.IsNullOrEmpty(vm.ValidationMessage));

        vm.ProjectName = "Demo";
        Assert.False(vm.CanCreate);           // still no location
        vm.Location = Path.Combine("C:", "ws");
        Assert.True(vm.CanCreate);            // schema+template default-selected
    }

    [Fact]
    public void BuildManifest_CarriesSelections()
    {
        var vm = Vm();
        vm.ProjectName = "Demo";
        vm.SelectedSchema = vm.Schemas.First(s => s.Id == "instruction");
        SelectTemplate(vm, "standard");

        var manifest = vm.BuildManifest();
        Assert.Equal("demo", manifest.ProjectId);
        Assert.Equal("Demo", manifest.Name);
        Assert.Equal("instruction", manifest.SchemaId);
        Assert.Equal("standard", manifest.TemplateId);
        Assert.True(manifest.IsRecognized);
    }
}
