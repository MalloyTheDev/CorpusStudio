using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Desktop.Views;

/// <summary>"Import from Hugging Face" dialog (#7 slice 2). Inspects a public dataset,
/// surfaces its license, maps its columns to the active project's schema, and stages the
/// mapped rows to a temp JSONL file. The caller runs that file through the normal
/// import-preview / quarantine flow — the engine and this dialog never write examples.jsonl.</summary>
public partial class HfImportWindow : Window
{
    private readonly PythonEngineService _engine;

    public HfImportWindow(
        PythonEngineService engine,
        string schemaId,
        string schemaName,
        IReadOnlyList<DatasetField> schemaFields)
    {
        InitializeComponent();
        _engine = engine;
        ViewModel = new HfImportViewModel(schemaId, schemaName, schemaFields);
        DataContext = ViewModel;
    }

    public HfImportViewModel ViewModel { get; }

    /// <summary>Set on a successful import: the staging file to hand to the normal import flow,
    /// plus the engine's import summary (mapping, license, row count).</summary>
    public HfImportDialogResult? Result { get; private set; }

    private async void InspectButton_Click(object sender, RoutedEventArgs e)
    {
        var datasetId = ViewModel.DatasetId.Trim();
        if (string.IsNullOrEmpty(datasetId))
        {
            return;
        }

        ViewModel.SetBusy($"Inspecting {datasetId}…");
        try
        {
            var inspection = await _engine.HfInspectAsync(datasetId);
            ViewModel.ApplyInspection(inspection);
        }
        catch (Exception ex)
        {
            ViewModel.SetError($"Could not inspect '{datasetId}': {ex.Message}");
        }
    }

    private async void ImportButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.CanImport || ViewModel.SelectedConfigSplit is null)
        {
            return;
        }

        var stagingPath = Path.Combine(
            Path.GetTempPath(),
            "corpus-hf-" + Guid.NewGuid().ToString("N") + ".jsonl");

        ViewModel.SetBusy("Fetching and mapping rows…");
        try
        {
            var summary = await _engine.HfImportAsync(
                ViewModel.DatasetId.Trim(),
                ViewModel.SchemaId,
                stagingPath,
                ViewModel.SelectedConfigSplit.Config,
                ViewModel.SelectedConfigSplit.Split,
                ViewModel.RowLimit,
                ViewModel.BuildMapping());

            Result = new HfImportDialogResult(stagingPath, summary);
            DialogResult = true;
        }
        catch (Exception ex)
        {
            ViewModel.SetError($"Import failed: {ex.Message}");
        }
    }
}

/// <summary>A successful Hugging Face import: the staging file to preview/commit, and the
/// engine's summary of what was fetched and mapped.</summary>
public sealed record HfImportDialogResult(string StagingPath, HfImportResult Summary);
