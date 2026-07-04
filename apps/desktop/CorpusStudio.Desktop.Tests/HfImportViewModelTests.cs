using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>HfImportViewModel (#7 slice 2): inspection → split/mapping population, auto-detect,
/// gating, and the field→column mapping the dialog sends to `hf-import`. All pure, no network.</summary>
public sealed class HfImportViewModelTests
{
    private static readonly DatasetField[] InstructionFields =
    [
        new("instruction", "text", true),
        new("input", "text", false),
        new("output", "markdown", true),
        new("tags", "list", false),
    ];

    private static HfImportViewModel NewVm() =>
        new("instruction", "Instruction Dataset", InstructionFields);

    private static HfDatasetInspection Inspection(bool gated = false) => new()
    {
        DatasetId = "acme/set",
        Viewable = true,
        Gated = gated,
        License = "apache-2.0",
        LicenseNote = "License: apache-2.0. Imported rows are NOT assumed to be licensed for training.",
        ConfigsSplits =
        [
            new HfConfigSplit { Config = "default", Split = "train" },
            new HfConfigSplit { Config = "default", Split = "test" },
        ],
        SampleColumns = ["instruction", "input", "output", "text"],
    };

    [Fact]
    public void ApplyInspection_PopulatesSplitsAndAutoMapsMatchingColumns()
    {
        var vm = NewVm();
        vm.ApplyInspection(Inspection());

        Assert.True(vm.HasInspection);
        Assert.Equal(2, vm.ConfigsSplits.Count);
        Assert.Equal("default / train", vm.SelectedConfigSplit!.Display);

        // One mapping row per schema field, exact-name columns auto-selected.
        var byField = vm.FieldMappings.ToDictionary(m => m.FieldName, m => m.SelectedColumn);
        Assert.Equal("instruction", byField["instruction"]);
        Assert.Equal("input", byField["input"]);
        Assert.Equal("output", byField["output"]);
        Assert.Equal(string.Empty, byField["tags"]);   // no "tags" column -> unmapped

        // The column dropdown offers "(none)" plus every dataset column.
        var options = vm.FieldMappings.First().AvailableColumns;
        Assert.Equal(string.Empty, options[0]);
        Assert.Contains("text", options);
    }

    [Fact]
    public void BuildMapping_ReturnsOnlyMappedFields()
    {
        var vm = NewVm();
        vm.ApplyInspection(Inspection());

        var mapping = vm.BuildMapping();
        Assert.Equal(new[] { "input", "instruction", "output" }, mapping.Keys.OrderBy(k => k).ToArray());
        Assert.False(mapping.ContainsKey("tags"));  // unmapped field omitted
        Assert.Equal("output", mapping["output"]);
    }

    [Fact]
    public void CanImport_RequiresInspectionMappingAndConfig()
    {
        var vm = NewVm();
        Assert.False(vm.CanImport);   // nothing inspected yet

        vm.ApplyInspection(Inspection());
        Assert.True(vm.CanImport);    // has a config + auto-mapped fields
    }

    [Fact]
    public void CanImport_IsFalseForGatedDataset()
    {
        var vm = NewVm();
        vm.ApplyInspection(Inspection(gated: true));

        Assert.True(vm.Gated);
        Assert.False(vm.CanImport);
        Assert.Contains("gated", vm.StatusMessage.ToLowerInvariant());
    }

    [Fact]
    public void CanImport_IsFalseWhenEveryFieldUnmapped()
    {
        var vm = NewVm();
        vm.ApplyInspection(Inspection());
        foreach (var mapping in vm.FieldMappings)
        {
            mapping.SelectedColumn = string.Empty;
        }

        Assert.False(vm.CanImport);   // no columns mapped -> nothing to import
    }

    [Fact]
    public void EditingAMapping_UpdatesCanImport()
    {
        var vm = NewVm();
        vm.ApplyInspection(Inspection());
        foreach (var mapping in vm.FieldMappings)
        {
            mapping.SelectedColumn = string.Empty;
        }
        Assert.False(vm.CanImport);

        // Re-mapping one field must re-enable import (live grid editing).
        vm.FieldMappings.First(m => m.FieldName == "output").SelectedColumn = "text";
        Assert.True(vm.CanImport);
    }

    [Fact]
    public void RowLimit_ClampsToAtLeastOne()
    {
        var vm = NewVm();
        vm.RowLimit = 0;
        Assert.Equal(1, vm.RowLimit);
        vm.RowLimit = 500;
        Assert.Equal(500, vm.RowLimit);
    }
}
