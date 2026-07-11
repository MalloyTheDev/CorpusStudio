using System;
using System.Collections.Generic;
using Avalonia.Controls;
using Avalonia.Interactivity;
using CorpusStudio.Avalonia.Services;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Avalonia.Views;

/// <summary>New Project wizard for the Avalonia head — the counterpart of the WPF
/// <c>WorkspaceWizardWindow</c>. Same shared <see cref="WorkspaceWizardViewModel"/> (schema +
/// template pickers, live folder preview, validation) and the same one creation path: on Create it
/// scaffolds the chosen template via <see cref="ProjectTemplateService"/> (refusing a non-empty
/// target) and exposes the folder in <see cref="Result"/>; the caller opens it as a workspace.</summary>
public partial class WorkspaceWizardWindow : Window
{
    private readonly ProjectTemplateService _templates = new();

    // Parameterless ctor for the XAML previewer / tooling only.
    public WorkspaceWizardWindow() : this(Array.Empty<DatasetSchema>())
    {
    }

    public WorkspaceWizardWindow(IReadOnlyList<DatasetSchema> schemas)
    {
        InitializeComponent();
        ViewModel = new WorkspaceWizardViewModel(schemas);
        DataContext = ViewModel;
    }

    public WorkspaceWizardViewModel ViewModel { get; }

    public WorkspaceWizardResult? Result { get; private set; }

    /// <summary>Head-agnostic folder picker seam (defaults to the Avalonia adapter), mirroring the
    /// WPF window so the shell injects the same service.</summary>
    public IFilePickerService FilePicker { get; set; } = new AvaloniaFilePickerService();

    private void ContinueButton_Click(object? sender, RoutedEventArgs e)
    {
        SetError(string.Empty);
        ViewModel.GoNext();
    }

    private void BackButton_Click(object? sender, RoutedEventArgs e)
    {
        SetError(string.Empty);
        ViewModel.GoBack();
    }

    private async void BrowseButton_Click(object? sender, RoutedEventArgs e)
    {
        var folder = await FilePicker.PickFolderAsync("Choose where to create the project");
        if (folder is not null)
        {
            ViewModel.Location = folder;
        }
    }

    private void CreateButton_Click(object? sender, RoutedEventArgs e)
    {
        SetError(string.Empty);

        if (!ViewModel.CanCreate)
        {
            SetError(ViewModel.ValidationMessage);
            return;
        }

        var target = ViewModel.TargetFolder;
        try
        {
            var result = _templates.Scaffold(
                target, ViewModel.BuildPlan(), ViewModel.BuildManifest(), allowNonEmpty: false);
            if (!result.Ok)
            {
                SetError(result.Error ?? "Could not create the project.");
                return;
            }

            Result = new WorkspaceWizardResult(target);
            Close(true);
        }
        catch (Exception ex)
        {
            SetError(ex.Message);
        }
    }

    private void CancelButton_Click(object? sender, RoutedEventArgs e) => Close(false);

    private void SetError(string message)
    {
        if (this.FindControl<TextBlock>("ErrorText") is { } error)
        {
            error.Text = message;
            error.IsVisible = !string.IsNullOrEmpty(message);
        }
    }
}

/// <summary>The outcome of a successful wizard run — the scaffolded workspace folder.</summary>
public sealed record WorkspaceWizardResult(string Folder);
