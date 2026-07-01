using System.Windows;
using System.Windows.Controls;
using System.Globalization;
using System.Windows.Input;
using Microsoft.Win32;

using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Desktop.Views;

public partial class MainWindow : Window
{
    private const int MaxAiAssistBulkUndoSteps = 20;

    private readonly PythonEngineService _engineService = new();
    private readonly List<IReadOnlyDictionary<string, string>> _aiAssistBulkUndoStack = [];

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
            ViewModel.SetImportQuarantineItems(_engineService.LoadImportQuarantineItems(createdPath));
            await RefreshQualityAsync(recordHistory: false);

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

    private async void ImportDatasetButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            MessageBox.Show(
                this,
                "Create or select a dataset project before importing.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        var dialog = new OpenFileDialog
        {
            Title = "Import JSONL Dataset",
            Filter = "JSONL files (*.jsonl)|*.jsonl|All files (*.*)|*.*",
            CheckFileExists = true,
            Multiselect = false
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        await PreviewAndImportJsonlAsync(dialog.FileName);
    }

    private async Task PreviewAndImportJsonlAsync(string importPath)
    {
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetImportInProgress(importPath);
            var report = await _engineService.PreviewImportAsync(importPath, ViewModel.ActiveSchemaId);
            ViewModel.ApplyImportPreview(report);
            Mouse.OverrideCursor = null;

            if (report.AcceptedRows == 0 && report.RejectedRows == 0)
            {
                MessageBox.Show(
                    this,
                    "No importable rows were found.",
                    "Import Preview",
                    MessageBoxButton.OK,
                    MessageBoxImage.Information
                );
                return;
            }

            var importChoice = report.RejectedRows > 0
                ? MessageBox.Show(
                    this,
                    BuildPartialImportPrompt(report),
                    "Import Preview",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Warning
                )
                : MessageBox.Show(
                    this,
                    $"Import {report.AcceptedRows} row(s) into {ViewModel.ActiveProjectTitle}?",
                    "Import Preview",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Question
                );

            if (importChoice != MessageBoxResult.Yes)
            {
                return;
            }

            Mouse.OverrideCursor = Cursors.Wait;
            var importResult = _engineService.CommitJsonlImportToProjectExamples(
                ViewModel.ActiveProjectPath!,
                importPath,
                report
            );
            ViewModel.SetExamples(_engineService.LoadExamples(ViewModel.ActiveProjectPath!));
            ViewModel.SetImportQuarantineItems(
                _engineService.LoadImportQuarantineItems(ViewModel.ActiveProjectPath!)
            );
            await RefreshQualityAsync();
            Mouse.OverrideCursor = null;

            MessageBox.Show(
                this,
                BuildImportCompleteMessage(importResult),
                "Import Complete",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetImportError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private string BuildPartialImportPrompt(ImportPreviewReport report)
    {
        if (report.AcceptedRows == 0)
        {
            return $"No rows can be imported. Save {report.RejectedRows} rejected row(s) to quarantine?";
        }

        return string.Join(
            Environment.NewLine,
            [
                $"Import {report.AcceptedRows} valid row(s) into {ViewModel.ActiveProjectTitle}?",
                $"The {report.RejectedRows} rejected row(s) will be saved to quarantine for repair.",
            ]
        );
    }

    private static string BuildImportCompleteMessage(ImportCommitResult result)
    {
        var lines = new List<string>
        {
            $"Imported {result.ImportedCount} row(s).",
        };

        if (result.QuarantinedCount > 0)
        {
            lines.Add($"Quarantined {result.QuarantinedCount} rejected row(s).");
            if (!string.IsNullOrWhiteSpace(result.QuarantinePath))
            {
                lines.Add(result.QuarantinePath);
            }
        }

        return string.Join(Environment.NewLine, lines);
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

    private void PrepareSyntheticRewriteButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PrepareSyntheticIssueRewrite())
        {
            return;
        }

        AiAssistTab.IsSelected = true;
    }

    private void PrepareSyntheticBatchRewriteButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PrepareSyntheticBatchRewrite())
        {
            return;
        }

        AiAssistTab.IsSelected = true;
    }

    private void PrepareEvaluationFailureButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PrepareEvaluationFailureReview())
        {
            return;
        }

        AiAssistTab.IsSelected = true;
    }

    private void EditEvaluationFailureButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PrepareEvaluationFailureEdit())
        {
            return;
        }

        WritingStudioTab.IsSelected = true;
        Dispatcher.BeginInvoke(new Action(() =>
        {
            DraftTextBox.Focus();
            DraftTextBox.Select(0, DraftTextBox.Text.Length);
        }));
    }

    private void PreparePreferenceJudgeButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PreparePreferenceJudgeReview())
        {
            return;
        }

        AiAssistTab.IsSelected = true;
    }

    private void PreparePreferenceBatchJudgeButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.PreparePreferenceBatchJudgeReview())
        {
            return;
        }

        AiAssistTab.IsSelected = true;
    }

    private void ExportPreferenceRankingButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetPreferenceRankingExportError("Create or select a preference project before exporting rankings.");
            return;
        }

        try
        {
            var items = ViewModel.GetVisiblePreferenceReviewItems();
            var outputPath = _engineService.ExportPreferenceRanking(
                ViewModel.ActiveProjectPath,
                items
            );
            ViewModel.ApplyPreferenceRankingExport(outputPath, items.Count);
        }
        catch (Exception ex)
        {
            ViewModel.SetPreferenceRankingExportError(ex.Message);
        }
    }

    private async void CheckEvaluationBackendButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadBackendOptions(
            ViewModel.EvaluationBackend,
            ViewModel.EvaluationModel,
            ViewModel.EvaluationBaseUrl,
            ViewModel.EvaluationTimeoutSeconds,
            "Evaluation",
            out var backend,
            out var model,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            ViewModel.SetEvaluationError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetEvaluationHealthCheckInProgress();
            var report = await _engineService.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            ViewModel.ApplyEvaluationBackendHealthReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void RefreshEvaluationModelsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadModelListOptions(
            ViewModel.EvaluationBackend,
            ViewModel.EvaluationBaseUrl,
            ViewModel.EvaluationTimeoutSeconds,
            "Evaluation",
            out var backend,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            ViewModel.SetEvaluationModelListError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetEvaluationModelListInProgress();
            var report = await _engineService.ListBackendModelsAsync(
                backend,
                baseUrl,
                timeoutSeconds
            );
            ViewModel.ApplyEvaluationModelListReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationModelListError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void RunEvaluationButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetEvaluationError("Create or select a dataset project before running evaluation.");
            return;
        }

        if (ViewModel.ActiveSchemaId is not ("instruction" or "chat"))
        {
            ViewModel.SetEvaluationError("Evaluation Lab MVP supports instruction and chat projects.");
            return;
        }

        if (!TryReadEvaluationOptions(
            out var backend,
            out var model,
            out var baseUrl,
            out var limit,
            out var scoreThreshold,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            ViewModel.SetEvaluationError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetEvaluationPreflightInProgress();
            var healthReport = await _engineService.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            if (!IsEvaluationBackendReady(healthReport))
            {
                ViewModel.SetEvaluationError(FormatEvaluationPreflightError(healthReport));
                return;
            }

            ViewModel.SetEvaluationInProgress();
            var result = await _engineService.RunEvaluationAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId,
                backend,
                model,
                baseUrl,
                limit,
                scoreThreshold,
                timeoutSeconds
            );
            ViewModel.ApplyEvaluationRunResult(result);
            ViewModel.SetEvaluationReportHistory(
                _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void RerunEvaluationReportButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetEvaluationError("Create or select a dataset project before rerunning evaluation.");
            return;
        }

        if (!ViewModel.TryGetSelectedEvaluationRunSettings(
            out var settings,
            out var errorMessage
        ))
        {
            ViewModel.SetEvaluationError(errorMessage);
            return;
        }

        if (settings.SchemaId is not ("instruction" or "chat"))
        {
            ViewModel.SetEvaluationError("Evaluation regression reruns support instruction and chat reports.");
            return;
        }

        if (!string.Equals(settings.SchemaId, ViewModel.ActiveSchemaId, StringComparison.OrdinalIgnoreCase))
        {
            ViewModel.SetEvaluationError(
                $"The selected report uses schema '{settings.SchemaId}', but the active project uses '{ViewModel.ActiveSchemaId}'."
            );
            return;
        }

        var baselineReportPath = ViewModel.SelectedEvaluationReportHistoryItem?.ReportPath;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetEvaluationRegressionRerunPreflightInProgress(settings);
            var healthReport = await _engineService.CheckBackendHealthAsync(
                settings.Backend,
                settings.Model,
                settings.BaseUrl,
                settings.TimeoutSeconds
            );
            if (!IsEvaluationBackendReady(healthReport))
            {
                ViewModel.SetEvaluationError(FormatEvaluationPreflightError(healthReport));
                return;
            }

            ViewModel.SetEvaluationRegressionRerunInProgress(settings);
            var result = await _engineService.RunEvaluationAsync(
                ViewModel.ActiveProjectPath,
                settings.SchemaId,
                settings.Backend,
                settings.Model,
                settings.BaseUrl,
                settings.Limit,
                settings.ScoreThreshold,
                settings.TimeoutSeconds
            );
            ViewModel.ApplyEvaluationRunResult(result);
            ViewModel.SetEvaluationReportHistory(
                _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
            );

            var newItem = ViewModel.EvaluationReportHistory
                .FirstOrDefault(item => item.ReportPath == result.ReportPath);
            var baselineItem = string.IsNullOrWhiteSpace(baselineReportPath)
                ? null
                : ViewModel.EvaluationReportHistory
                    .FirstOrDefault(item => item.ReportPath == baselineReportPath);

            if (newItem is not null)
            {
                ViewModel.SelectedEvaluationReportHistoryItem = newItem;
            }

            if (baselineItem is not null)
            {
                ViewModel.SecondaryEvaluationReportHistoryItem = baselineItem;
                ViewModel.CompareSelectedEvaluationReports();
            }
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void CheckAiAssistBackendButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadBackendOptions(
            ViewModel.AiAssistBackend,
            ViewModel.AiAssistModel,
            ViewModel.AiAssistBaseUrl,
            ViewModel.AiAssistTimeoutSeconds,
            "AI Assist",
            out var backend,
            out var model,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            ViewModel.SetAiAssistError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetAiAssistHealthCheckInProgress();
            var report = await _engineService.CheckBackendHealthAsync(
                backend,
                model,
                baseUrl,
                timeoutSeconds
            );
            ViewModel.ApplyAiAssistBackendHealthReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void RefreshAiAssistModelsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadModelListOptions(
            ViewModel.AiAssistBackend,
            ViewModel.AiAssistBaseUrl,
            ViewModel.AiAssistTimeoutSeconds,
            "AI Assist",
            out var backend,
            out var baseUrl,
            out var timeoutSeconds,
            out var errorMessage
        ))
        {
            ViewModel.SetAiAssistModelListError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetAiAssistModelListInProgress();
            var report = await _engineService.ListBackendModelsAsync(
                backend,
                baseUrl,
                timeoutSeconds
            );
            ViewModel.ApplyAiAssistModelListReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistModelListError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void RunAiAssistButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistError("Create or select a dataset project before running AI Assist.");
            return;
        }

        if (string.IsNullOrWhiteSpace(ViewModel.DraftText))
        {
            ViewModel.SetAiAssistError("Add a draft example before running AI Assist.");
            return;
        }

        if (!TryReadAiAssistOptions(
            out var backend,
            out var model,
            out var baseUrl,
            out var action,
            out var timeoutSeconds,
            out var instruction,
            out var errorMessage
        ))
        {
            ViewModel.SetAiAssistError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetAiAssistInProgress();
            var result = await _engineService.RunAiAssistAsync(
                ViewModel.DraftText,
                ViewModel.ActiveSchemaId,
                action,
                backend,
                model,
                baseUrl,
                timeoutSeconds,
                instruction
            );
            ViewModel.ApplyAiAssistRunResult(result);
            var queuedItem = _engineService.SaveAiAssistReviewQueueItem(
                ViewModel.ActiveProjectPath,
                ViewModel.DraftText,
                result
            );
            ViewModel.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            ViewModel.SelectedAiAssistReviewQueueItem = ViewModel.AiAssistReviewQueue
                .FirstOrDefault(item => item.ReviewId == queuedItem.ReviewId);
            ClearAiAssistBulkUndoStack();
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private void UseAiAssistSuggestionButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedAiAssistReviewQueueItem is null
            || string.IsNullOrWhiteSpace(ViewModel.SelectedAiAssistReviewQueueItem.SuggestedJsonl))
        {
            ViewModel.MoveAiAssistSuggestionToDraft();
            return;
        }

        if (!MarkSelectedAiAssistReview("accepted"))
        {
            return;
        }

        if (!ViewModel.MoveAiAssistSuggestionToDraft())
        {
            return;
        }

        WritingStudioTab.IsSelected = true;
        Dispatcher.BeginInvoke(new Action(() =>
        {
            DraftTextBox.Focus();
            DraftTextBox.Select(0, DraftTextBox.Text.Length);
        }));
    }

    private void AcceptAiAssistReviewButton_Click(object sender, RoutedEventArgs e)
    {
        MarkSelectedAiAssistReview("accepted");
    }

    private void RejectAiAssistReviewButton_Click(object sender, RoutedEventArgs e)
    {
        MarkSelectedAiAssistReview("rejected");
    }

    private void BulkAcceptAiAssistReviewsButton_Click(object sender, RoutedEventArgs e)
    {
        BulkMarkVisibleAiAssistReviews("accepted");
    }

    private void BulkRejectAiAssistReviewsButton_Click(object sender, RoutedEventArgs e)
    {
        BulkMarkVisibleAiAssistReviews("rejected");
    }

    private void UndoBulkAiAssistReviewsButton_Click(object sender, RoutedEventArgs e)
    {
        UndoBulkAiAssistReviews();
    }

    private void SaveAiAssistQueueViewButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistQueueError("Create or select a dataset project before saving an AI Assist queue view.");
            return;
        }

        var view = ViewModel.BuildCurrentAiAssistQueueView();
        if (string.IsNullOrWhiteSpace(view.Name))
        {
            ViewModel.SetAiAssistQueueError("Name the AI Assist queue view before saving.");
            return;
        }

        try
        {
            var savedView = _engineService.SaveAiAssistQueueView(ViewModel.ActiveProjectPath, view);
            ViewModel.SetAiAssistQueueViews(
                _engineService.LoadAiAssistQueueViews(ViewModel.ActiveProjectPath)
            );
            ViewModel.SelectedAiAssistQueueView = ViewModel.AiAssistQueueViews
                .FirstOrDefault(item => string.Equals(
                    item.Name,
                    savedView.Name,
                    StringComparison.OrdinalIgnoreCase
                ));
            ViewModel.ApplyAiAssistQueueViewSaved(savedView);
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistQueueError(ex.Message);
        }
    }

    private void LoadAiAssistQueueViewButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedAiAssistQueueView is null)
        {
            ViewModel.SetAiAssistQueueError("Select a saved AI Assist queue view before loading.");
            return;
        }

        var view = ViewModel.SelectedAiAssistQueueView;
        ViewModel.ApplyAiAssistQueueView(view);
        ViewModel.ApplyAiAssistQueueViewLoaded(view);
    }

    private bool MarkSelectedAiAssistReview(string reviewState)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistQueueError("Create or select a dataset project before updating AI Assist review state.");
            return false;
        }

        if (ViewModel.SelectedAiAssistReviewQueueItem is null)
        {
            ViewModel.SetAiAssistQueueError("Select an AI Assist review before updating its state.");
            return false;
        }

        try
        {
            var reviewId = ViewModel.SelectedAiAssistReviewQueueItem.ReviewId;
            var updatedItem = _engineService.UpdateAiAssistReviewState(
                ViewModel.ActiveProjectPath,
                reviewId,
                reviewState
            );
            ViewModel.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            ViewModel.SelectedAiAssistReviewQueueItem = ViewModel.AiAssistReviewQueue
                .FirstOrDefault(item => item.ReviewId == reviewId);
            ViewModel.ApplyAiAssistReviewState(updatedItem);
            ClearAiAssistBulkUndoStack();
            return true;
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistQueueError(ex.Message);
            return false;
        }
    }

    private void BulkMarkVisibleAiAssistReviews(string reviewState)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistQueueError("Create or select a dataset project before updating AI Assist review state.");
            return;
        }

        var reviewIds = ViewModel.GetVisibleAiAssistReviewIds();
        var previousStates = ViewModel.GetVisibleAiAssistReviewStates();
        if (reviewIds.Count == 0)
        {
            ViewModel.SetAiAssistQueueError("No AI Assist reviews match the current filter.");
            return;
        }

        try
        {
            var updatedCount = _engineService.UpdateAiAssistReviewStates(
                ViewModel.ActiveProjectPath,
                reviewIds,
                reviewState
            );
            ViewModel.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            PushAiAssistBulkUndoStep(previousStates);
            ViewModel.ApplyAiAssistBulkReviewState(
                updatedCount,
                reviewState,
                _aiAssistBulkUndoStack.Count
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistQueueError(ex.Message);
        }
    }

    private void UndoBulkAiAssistReviews()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistQueueError("Create or select a dataset project before undoing AI Assist bulk triage.");
            return;
        }

        if (_aiAssistBulkUndoStack.Count == 0)
        {
            ViewModel.SetAiAssistQueueError("No AI Assist bulk triage action is available to undo.");
            return;
        }

        try
        {
            var previousStates = _aiAssistBulkUndoStack[^1];
            var restoredCount = _engineService.UpdateAiAssistReviewStates(
                ViewModel.ActiveProjectPath,
                previousStates
            );
            ViewModel.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            _aiAssistBulkUndoStack.RemoveAt(_aiAssistBulkUndoStack.Count - 1);
            ViewModel.ApplyAiAssistBulkUndoReviewState(
                restoredCount,
                _aiAssistBulkUndoStack.Count
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetAiAssistQueueError(ex.Message);
        }
    }

    private void PushAiAssistBulkUndoStep(IReadOnlyDictionary<string, string> previousStates)
    {
        if (previousStates.Count == 0)
        {
            return;
        }

        if (_aiAssistBulkUndoStack.Count >= MaxAiAssistBulkUndoSteps)
        {
            _aiAssistBulkUndoStack.RemoveAt(0);
        }

        _aiAssistBulkUndoStack.Add(previousStates.ToDictionary(
            pair => pair.Key,
            pair => pair.Value,
            StringComparer.Ordinal
        ));
    }

    private void ClearAiAssistBulkUndoStack()
    {
        _aiAssistBulkUndoStack.Clear();
    }

    private void SaveEvaluationReviewButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedEvaluationReportHistoryItem is null)
        {
            ViewModel.SetEvaluationReviewError("Select an evaluation report before saving review notes.");
            return;
        }

        if (ViewModel.SelectedEvaluationExampleResult is null)
        {
            ViewModel.SetEvaluationReviewError("Select an evaluation example before saving review notes.");
            return;
        }

        if (!TryReadEvaluationManualReview(
            out var manualScore,
            out var manualNotes,
            out var errorMessage
        ))
        {
            ViewModel.SetEvaluationReviewError(errorMessage);
            return;
        }

        try
        {
            var exampleId = ViewModel.SelectedEvaluationExampleResult.ExampleId;
            var updatedItem = _engineService.SaveEvaluationManualReview(
                ViewModel.SelectedEvaluationReportHistoryItem,
                exampleId,
                manualScore,
                manualNotes
            );

            if (!string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
            {
                ViewModel.SetEvaluationReportHistory(
                    _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
                );
                ViewModel.SelectedEvaluationReportHistoryItem = ViewModel.EvaluationReportHistory
                    .FirstOrDefault(item => item.ReportPath == updatedItem.ReportPath);
            }

            ViewModel.SelectedEvaluationExampleResult = ViewModel.EvaluationResults
                .FirstOrDefault(result => result.ExampleId == exampleId);
            ViewModel.ApplySavedEvaluationManualReview(updatedItem);
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationReviewError(ex.Message);
        }
    }

    private void CompareEvaluationReportsButton_Click(object sender, RoutedEventArgs e)
    {
        ViewModel.CompareSelectedEvaluationReports();
    }

    private async void GenerateTrainingConfigButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetTrainingConfigError("Create or select a dataset project before generating a training config.");
            return;
        }

        if (!TryReadTrainingConfigOptions(
            out var target,
            out var baseModel,
            out var datasetFormat,
            out var sequenceLen,
            out var loraR,
            out var loraAlpha,
            out var microBatchSize,
            out var gradientAccumulationSteps,
            out var learningRate,
            out var errorMessage
        ))
        {
            ViewModel.SetTrainingConfigError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetTrainingConfigInProgress();
            var result = await _engineService.GenerateTrainingConfigAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId,
                target,
                baseModel,
                datasetFormat,
                sequenceLen,
                loraR,
                loraAlpha,
                microBatchSize,
                gradientAccumulationSteps,
                learningRate
            );
            ViewModel.ApplyTrainingConfigExportResult(result);
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingConfigError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void GenerateDatasetCardButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetDatasetCardError("Create or select a dataset project before generating a dataset card.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetDatasetCardInProgress();
            var result = await _engineService.GenerateDatasetCardAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId
            );
            ViewModel.ApplyDatasetCardResult(result);
        }
        catch (Exception ex)
        {
            ViewModel.SetDatasetCardError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private void SaveLabSettingsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetLabSettingsError("Create or select a dataset project before saving lab backend settings.");
            return;
        }

        try
        {
            var settings = ViewModel.BuildCurrentLabSettings();
            _engineService.SaveProjectLabSettings(ViewModel.ActiveProjectPath, settings);
            ViewModel.ApplyLabSettingsSaved(ViewModel.ActiveProjectPath);
        }
        catch (Exception ex)
        {
            ViewModel.SetLabSettingsError(ex.Message);
        }
    }

    private void RetryQuarantineItemButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedImportQuarantineItem is null)
        {
            MessageBox.Show(
                this,
                "Select a quarantined row before retrying.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        ViewModel.RetrySelectedImportQuarantineItem();
        WritingStudioTab.IsSelected = true;
        Dispatcher.BeginInvoke(new Action(() =>
        {
            DraftTextBox.Focus();
            DraftTextBox.Select(0, DraftTextBox.Text.Length);
        }));
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
            if (!TryReadSplitOptions(
                out var trainRatio,
                out var validationRatio,
                out var seed,
                out var errorMessage
            ))
            {
                ViewModel.SetSplitError(errorMessage);
                return;
            }

            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetSplitInProgress(trainRatio, validationRatio, seed);
            var report = await _engineService.GenerateProjectSplitsAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId,
                trainRatio,
                validationRatio,
                seed
            );
            _engineService.SaveProjectSplitSettings(
                ViewModel.ActiveProjectPath,
                new SplitSettings
                {
                    TrainRatio = trainRatio,
                    ValidationRatio = validationRatio,
                    Seed = seed,
                }
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

    private void ValidationIssuesListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (ViewModel.SelectedValidationIssue is null)
        {
            return;
        }

        FocusDraftForIssue(ViewModel.SelectedValidationIssue);
    }

    private void FocusDraftForIssue(ValidationIssueNavigationItem issue)
    {
        WritingStudioTab.IsSelected = true;
        Dispatcher.BeginInvoke(new Action(() =>
        {
            var selection = FindDraftIssueSelection(DraftTextBox.Text, issue);
            DraftTextBox.Focus();
            DraftTextBox.Select(selection.Start, selection.Length);

            var lineIndex = DraftTextBox.GetLineIndexFromCharacterIndex(selection.Start);
            if (lineIndex >= 0)
            {
                DraftTextBox.ScrollToLine(lineIndex);
            }
        }));
    }

    private static (int Start, int Length) FindDraftIssueSelection(
        string draftText,
        ValidationIssueNavigationItem issue
    )
    {
        if (!string.IsNullOrWhiteSpace(issue.Field))
        {
            var quotedField = $"\"{issue.Field}\"";
            var fieldIndex = draftText.IndexOf(quotedField, StringComparison.OrdinalIgnoreCase);
            if (fieldIndex >= 0)
            {
                return (fieldIndex, quotedField.Length);
            }

            fieldIndex = draftText.IndexOf(issue.Field, StringComparison.OrdinalIgnoreCase);
            if (fieldIndex >= 0)
            {
                return (fieldIndex, issue.Field.Length);
            }
        }

        if (issue.RowNumber is not null)
        {
            var lineStart = FindLineStart(draftText, issue.RowNumber.Value);
            var lineEnd = draftText.IndexOf('\n', lineStart);
            var length = lineEnd < 0 ? draftText.Length - lineStart : lineEnd - lineStart;
            return (lineStart, Math.Max(0, length));
        }

        return (0, 0);
    }

    private static int FindLineStart(string value, int rowNumber)
    {
        if (rowNumber <= 1)
        {
            return 0;
        }

        var currentRow = 1;
        for (var index = 0; index < value.Length; index++)
        {
            if (value[index] != '\n')
            {
                continue;
            }

            currentRow++;
            if (currentRow == rowNumber)
            {
                return Math.Min(index + 1, value.Length);
            }
        }

        return 0;
    }

    private bool TryReadSplitOptions(
        out double trainRatio,
        out double validationRatio,
        out int seed,
        out string errorMessage
    )
    {
        trainRatio = 0;
        validationRatio = 0;
        seed = 0;
        errorMessage = string.Empty;

        if (!double.TryParse(
            ViewModel.SplitTrainPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var trainPercent
        ))
        {
            errorMessage = "Train split must be a number from 1 to 98.";
            return false;
        }

        if (!double.TryParse(
            ViewModel.SplitValidationPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var validationPercent
        ))
        {
            errorMessage = "Validation split must be a number from 0 to 98.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.SplitSeed,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out seed
        ))
        {
            errorMessage = "Seed must be a whole number.";
            return false;
        }

        if (!double.IsFinite(trainPercent) || !double.IsFinite(validationPercent))
        {
            errorMessage = "Split percentages must be finite numbers.";
            return false;
        }

        trainRatio = trainPercent / 100;
        validationRatio = validationPercent / 100;
        var testRatio = 1 - trainRatio - validationRatio;

        if (trainRatio <= 0 || validationRatio < 0 || testRatio <= 0)
        {
            errorMessage = "Split percentages must leave at least some room for train and test rows.";
            return false;
        }

        return true;
    }

    private bool TryReadEvaluationOptions(
        out string backend,
        out string model,
        out string? baseUrl,
        out int? limit,
        out double scoreThreshold,
        out int timeoutSeconds,
        out string errorMessage
    )
    {
        backend = ViewModel.EvaluationBackend.Trim();
        model = ViewModel.EvaluationModel.Trim();
        baseUrl = string.IsNullOrWhiteSpace(ViewModel.EvaluationBaseUrl)
            ? null
            : ViewModel.EvaluationBaseUrl.Trim();
        limit = null;
        scoreThreshold = 0;
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = "Evaluation backend is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            errorMessage = "Evaluation model is required.";
            return false;
        }

        if (!string.IsNullOrWhiteSpace(ViewModel.EvaluationLimit))
        {
            if (!int.TryParse(
                ViewModel.EvaluationLimit,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out var parsedLimit
            ) || parsedLimit <= 0)
            {
                errorMessage = "Evaluation limit must be a positive whole number or blank.";
                return false;
            }

            limit = parsedLimit;
        }

        if (!double.TryParse(
            ViewModel.EvaluationScoreThreshold,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out scoreThreshold
        ) || !double.IsFinite(scoreThreshold) || scoreThreshold < 0 || scoreThreshold > 100)
        {
            errorMessage = "Evaluation score threshold must be a number from 0 to 100.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.EvaluationTimeoutSeconds,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = "Evaluation timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private static bool IsEvaluationBackendReady(BackendHealthReport report)
    {
        return report.Reachable && report.ModelAvailable;
    }

    private static string FormatEvaluationPreflightError(BackendHealthReport report)
    {
        var lines = new List<string>
        {
            "Pre-run backend health check failed.",
            $"Backend: {report.ProviderName}",
            $"Model: {report.ModelName}",
            $"Base URL: {report.BaseUrl}",
        };

        if (!report.Reachable)
        {
            lines.Add("Backend is not reachable.");
        }
        else if (!report.ModelAvailable)
        {
            lines.Add("The configured model was not listed by the backend.");
        }

        if (report.AvailableModels.Count > 0)
        {
            lines.Add($"Available models: {string.Join(", ", report.AvailableModels.Take(5))}");
        }

        if (!string.IsNullOrWhiteSpace(report.Error))
        {
            lines.Add($"Error: {report.Error}");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private bool TryReadEvaluationManualReview(
        out double? manualScore,
        out string? manualNotes,
        out string errorMessage
    )
    {
        manualScore = null;
        manualNotes = string.IsNullOrWhiteSpace(ViewModel.EvaluationManualNotes)
            ? null
            : ViewModel.EvaluationManualNotes.Trim();
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(ViewModel.EvaluationManualScore))
        {
            return true;
        }

        if (!double.TryParse(
            ViewModel.EvaluationManualScore,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var parsedScore
        ) || !double.IsFinite(parsedScore) || parsedScore < 0 || parsedScore > 100)
        {
            errorMessage = "Manual evaluation score must be blank or a number from 0 to 100.";
            return false;
        }

        manualScore = parsedScore;
        return true;
    }

    private static bool TryReadBackendOptions(
        string backendText,
        string modelText,
        string baseUrlText,
        string timeoutText,
        string label,
        out string backend,
        out string model,
        out string? baseUrl,
        out int timeoutSeconds,
        out string errorMessage
    )
    {
        backend = backendText.Trim();
        model = modelText.Trim();
        baseUrl = string.IsNullOrWhiteSpace(baseUrlText) ? null : baseUrlText.Trim();
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = $"{label} backend is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            errorMessage = $"{label} model is required.";
            return false;
        }

        if (!int.TryParse(
            timeoutText,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = $"{label} timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private static bool TryReadModelListOptions(
        string backendText,
        string baseUrlText,
        string timeoutText,
        string label,
        out string backend,
        out string? baseUrl,
        out int timeoutSeconds,
        out string errorMessage
    )
    {
        backend = backendText.Trim();
        baseUrl = string.IsNullOrWhiteSpace(baseUrlText) ? null : baseUrlText.Trim();
        timeoutSeconds = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = $"{label} backend is required.";
            return false;
        }

        if (!int.TryParse(
            timeoutText,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = $"{label} timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private bool TryReadAiAssistOptions(
        out string backend,
        out string model,
        out string? baseUrl,
        out string action,
        out int timeoutSeconds,
        out string? instruction,
        out string errorMessage
    )
    {
        backend = ViewModel.AiAssistBackend.Trim();
        model = ViewModel.AiAssistModel.Trim();
        baseUrl = string.IsNullOrWhiteSpace(ViewModel.AiAssistBaseUrl)
            ? null
            : ViewModel.AiAssistBaseUrl.Trim();
        action = ViewModel.AiAssistAction.Trim();
        timeoutSeconds = 0;
        instruction = string.IsNullOrWhiteSpace(ViewModel.AiAssistInstruction)
            ? null
            : ViewModel.AiAssistInstruction.Trim();
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(backend))
        {
            errorMessage = "AI Assist backend is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(model))
        {
            errorMessage = "AI Assist model is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(action))
        {
            errorMessage = "AI Assist action is required.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.AiAssistTimeoutSeconds,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out timeoutSeconds
        ) || timeoutSeconds <= 0)
        {
            errorMessage = "AI Assist timeout must be a positive whole number.";
            return false;
        }

        return true;
    }

    private bool TryReadTrainingConfigOptions(
        out string target,
        out string baseModel,
        out string datasetFormat,
        out int sequenceLen,
        out int loraR,
        out int loraAlpha,
        out int microBatchSize,
        out int gradientAccumulationSteps,
        out double learningRate,
        out string errorMessage
    )
    {
        target = ViewModel.TrainingTarget.Trim();
        baseModel = ViewModel.TrainingBaseModel.Trim();
        datasetFormat = ViewModel.TrainingFormat.Trim();
        sequenceLen = 0;
        loraR = 0;
        loraAlpha = 0;
        microBatchSize = 0;
        gradientAccumulationSteps = 0;
        learningRate = 0;
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(target))
        {
            errorMessage = "Training target is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(baseModel))
        {
            errorMessage = "Training base model is required.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(datasetFormat))
        {
            errorMessage = "Training format is required.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.TrainingSequenceLen,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out sequenceLen
        ) || sequenceLen <= 0)
        {
            errorMessage = "Training sequence length must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.TrainingLoraR,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out loraR
        ) || loraR <= 0)
        {
            errorMessage = "Training LoRA r must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.TrainingLoraAlpha,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out loraAlpha
        ) || loraAlpha <= 0)
        {
            errorMessage = "Training LoRA alpha must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.TrainingMicroBatchSize,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out microBatchSize
        ) || microBatchSize <= 0)
        {
            errorMessage = "Training micro batch size must be a positive whole number.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.TrainingGradientAccumulationSteps,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out gradientAccumulationSteps
        ) || gradientAccumulationSteps <= 0)
        {
            errorMessage = "Training gradient accumulation steps must be a positive whole number.";
            return false;
        }

        if (!double.TryParse(
            ViewModel.TrainingLearningRate,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out learningRate
        ) || !double.IsFinite(learningRate) || learningRate <= 0)
        {
            errorMessage = "Training learning rate must be a positive number.";
            return false;
        }

        return true;
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
        ViewModel.ApplySplitSettings(_engineService.LoadProjectSplitSettings(project.ProjectPath));
        ViewModel.ApplyLabSettings(_engineService.LoadProjectLabSettings(project.ProjectPath));
        ViewModel.SetExamples(_engineService.LoadExamples(project.ProjectPath));
        ViewModel.SetImportQuarantineItems(_engineService.LoadImportQuarantineItems(project.ProjectPath));
        ViewModel.SetAiAssistReviewQueue(_engineService.LoadAiAssistReviewQueue(project.ProjectPath));
        ViewModel.SetAiAssistQueueViews(_engineService.LoadAiAssistQueueViews(project.ProjectPath));
        ClearAiAssistBulkUndoStack();
        ViewModel.SetEvaluationReportHistory(
            _engineService.LoadEvaluationReportHistory(project.ProjectPath)
        );
        await RefreshQualityAsync(recordHistory: false);
    }

    private async Task RefreshQualityAsync(bool recordHistory = true)
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
            if (recordHistory)
            {
                _engineService.SaveQualityHistoryEntry(ViewModel.ActiveProjectPath, report);
            }

            var history = _engineService.LoadQualityHistory(ViewModel.ActiveProjectPath);
            ViewModel.ApplyQualityReport(report, history);
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
