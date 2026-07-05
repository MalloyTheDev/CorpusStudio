using System;
using System.Collections.Generic;
using System.Windows;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Desktop.Views;

/// <summary>New Project wizard (v1.2.4 Workspace System, slice 3c-2): schema + template
/// pickers with a live folder-structure preview. On Create it scaffolds the chosen template
/// (refusing a non-empty target) and exposes the folder in <see cref="Result"/>; the caller
/// opens it as a workspace.</summary>
public partial class WorkspaceWizardWindow : Window
{
    private readonly ProjectTemplateService _templates = new();

    public WorkspaceWizardWindow(IReadOnlyList<DatasetSchema> schemas)
    {
        InitializeComponent();
        ViewModel = new WorkspaceWizardViewModel(schemas);
        DataContext = ViewModel;
    }

    public WorkspaceWizardViewModel ViewModel { get; }

    public WorkspaceWizardResult? Result { get; private set; }

    /// <summary>Head-agnostic folder picker seam (defaults to the WPF adapter). See
    /// docs/AVALONIA_MIGRATION_PLAN.md.</summary>
    public IFilePickerService FilePicker { get; set; } = new Win32FilePickerService();

    private async void BrowseButton_Click(object sender, RoutedEventArgs e)
    {
        var folder = await FilePicker.PickFolderAsync("Choose where to create the project");
        if (folder is not null)
        {
            ViewModel.Location = folder;
        }
    }

    private void CreateButton_Click(object sender, RoutedEventArgs e)
    {
        ErrorText.Text = string.Empty;

        if (!ViewModel.CanCreate)
        {
            ErrorText.Text = ViewModel.ValidationMessage;
            return;
        }

        var target = ViewModel.TargetFolder;
        try
        {
            var result = _templates.Scaffold(target, ViewModel.BuildPlan(), ViewModel.BuildManifest(), allowNonEmpty: false);
            if (!result.Ok)
            {
                ErrorText.Text = result.Error;
                return;
            }

            Result = new WorkspaceWizardResult(target);
            DialogResult = true;
        }
        catch (Exception ex)
        {
            ErrorText.Text = ex.Message;
        }
    }
}

/// <summary>The outcome of a successful wizard run — the scaffolded workspace folder.</summary>
public sealed record WorkspaceWizardResult(string Folder);
