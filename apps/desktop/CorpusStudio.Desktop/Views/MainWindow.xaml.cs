using System.Windows;
using System.Windows.Input;

using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Desktop.Views;

public partial class MainWindow : Window
{
    private readonly PythonEngineService _engineService = new();

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
    }

    private MainWindowViewModel ViewModel => (MainWindowViewModel)DataContext;

    private void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        ViewModel.SetProjects(_engineService.LoadProjects());
        ViewModel.SetSettings(_engineService.GetSettings());
    }

    private async void NewDatasetProjectButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            var schemas = await _engineService.GetSchemasAsync();
            Mouse.OverrideCursor = null;

            var dialog = new NewProjectWindow(schemas)
            {
                Owner = this
            };

            if (dialog.ShowDialog() != true || dialog.ProjectRequest is null)
            {
                return;
            }

            Mouse.OverrideCursor = Cursors.Wait;
            var createdPath = await CreateProjectAsync(dialog.ProjectRequest);
            Mouse.OverrideCursor = null;

            ViewModel.AddProject(
                dialog.ProjectRequest.ProjectId,
                dialog.ProjectRequest.Name,
                dialog.ProjectRequest.SchemaId,
                dialog.ProjectRequest.SchemaName,
                createdPath
            );

            MessageBox.Show(
                this,
                $"Created project at:\n{createdPath}",
                "Project Created",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
        }
        catch (Exception ex)
        {
            Mouse.OverrideCursor = null;
            MessageBox.Show(this, ex.Message, "Corpus Studio", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async Task<string> CreateProjectAsync(NewProjectRequest request)
    {
        return (await _engineService.CreateProjectAsync(
            request.ProjectId,
            request.Name,
            request.SchemaId
        )).Trim();
    }

    private async void ValidateButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetValidationInProgress();
            var report = await _engineService.ValidateDraftAsync(
                ViewModel.DraftText,
                ViewModel.ActiveSchemaId
            );

            ViewModel.ApplyValidationReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetValidationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void SaveExampleButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            MessageBox.Show(
                this,
                "Create a dataset project before saving examples.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetValidationInProgress();

            var report = await _engineService.ValidateDraftAsync(
                ViewModel.DraftText,
                ViewModel.ActiveSchemaId
            );

            ViewModel.ApplyValidationReport(report);
            if (!report.Valid)
            {
                return;
            }

            var savedCount = _engineService.AppendDraftToProjectExamples(
                ViewModel.ActiveProjectPath,
                ViewModel.DraftText
            );

            ViewModel.AddSavedExamples(savedCount);
            MessageBox.Show(
                this,
                $"Saved {savedCount} example(s).",
                "Example Saved",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetValidationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void ExportJsonlButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            MessageBox.Show(
                this,
                "Create a dataset project before exporting.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            var outputPath = await _engineService.ExportProjectExamplesAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId
            );

            MessageBox.Show(
                this,
                $"Exported JSONL to:\n{outputPath}",
                "Export Complete",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetValidationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }
}
