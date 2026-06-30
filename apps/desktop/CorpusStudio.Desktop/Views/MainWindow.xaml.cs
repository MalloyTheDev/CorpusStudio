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

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        var projects = _engineService.LoadProjects();
        ViewModel.SetProjects(projects);
        ViewModel.SetSettings(_engineService.GetSettings());

        var firstProject = projects.FirstOrDefault();
        if (firstProject is not null)
        {
            await LoadProjectAsync(firstProject);
        }
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
            ViewModel.SetExamples(_engineService.LoadExamples(createdPath));
            await RefreshQualityAsync();

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

            ViewModel.SetExamples(_engineService.LoadExamples(ViewModel.ActiveProjectPath));
            await RefreshQualityAsync();
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

    private async void RunQualityButton_Click(object sender, RoutedEventArgs e)
    {
        await RefreshQualityAsync();
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

    private async void GenerateSplitsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetSplitError("Create a dataset project before generating splits.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetSplitInProgress();
            var report = await _engineService.GenerateProjectSplitsAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId
            );

            ViewModel.ApplySplitReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetSplitError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void ProjectsListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (ViewModel.SelectedProject is not null)
        {
            await LoadProjectAsync(ViewModel.SelectedProject);
        }
    }

    private async Task LoadProjectAsync(DatasetProjectListItem project)
    {
        ViewModel.SelectProject(project);
        ViewModel.SetExamples(_engineService.LoadExamples(project.ProjectPath));
        await RefreshQualityAsync();
    }

    private async Task RefreshQualityAsync()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetQualityError("Create or select a dataset project before running quality checks.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetQualityInProgress();
            var report = await _engineService.BuildQualityReportAsync(ViewModel.ActiveProjectPath);
            ViewModel.ApplyQualityReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetQualityError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }
}
