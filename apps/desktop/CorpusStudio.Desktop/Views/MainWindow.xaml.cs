using System.Windows;
using System.Windows.Controls;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Windows.Input;
using System.Windows.Threading;

using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;

namespace CorpusStudio.Desktop.Views;

public partial class MainWindow : Window
{
    private const int MaxAiAssistBulkUndoSteps = 20;

    private readonly PythonEngineService _engineService = new();
    private readonly List<IReadOnlyDictionary<string, string>> _aiAssistBulkUndoStack = [];
    private readonly TrainingProcessRunner _trainingRunner = new();
    private CancellationTokenSource? _trainingRunCts;
    private readonly ConcurrentQueue<string> _trainingLogQueue = new();
    private bool _trainingCancelRequested;

    /// <summary>Head-agnostic dialog seam (set from DI in App.OnStartup; defaults to the WPF adapter
    /// so a plain `new MainWindow()` still works). Confirm/message prompts route through this so the
    /// logic behind them can move to shared view-models during the cross-platform port. See
    /// docs/AVALONIA_MIGRATION_PLAN.md.</summary>
    public IDialogService Dialogs { get; set; } = new MessageBoxDialogService();

    /// <summary>Head-agnostic file/folder picker seam (set from DI; defaults to the WPF adapter).
    /// See docs/AVALONIA_MIGRATION_PLAN.md.</summary>
    public IFilePickerService FilePicker { get; set; } = new Win32FilePickerService();

    public MainWindow()
    {
        InitializeComponent();
        Loaded += MainWindow_Loaded;
        Closing += MainWindow_Closing;
        // Record every engine CLI invocation in the Output / Logs panel. The event fires on a
        // background thread, so marshal onto the dispatcher before touching the view-model.
        _engineService.CommandCompleted += OnEngineCommandCompleted;
    }

    private void OnEngineCommandCompleted(object? sender, EngineLogEntry entry)
    {
        // The event fires on a background thread; marshal onto the dispatcher. Null-safe on the
        // deferred call because DataContext may be gone during teardown (the EmitCommandLog
        // try/catch only guards the synchronous BeginInvoke, not this lambda's later execution).
        if (Dispatcher.HasShutdownStarted || Dispatcher.HasShutdownFinished)
        {
            return;
        }

        Dispatcher.BeginInvoke(() => (DataContext as MainWindowViewModel)?.AppendEngineLog(entry));
    }

    private MainWindowViewModel ViewModel => (MainWindowViewModel)DataContext;

    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        if (!_engineService.IsEngineAvailable)
        {
            // Don't touch the engine — show the setup screen instead of crashing.
            ViewModel.SetEngineUnavailable(_engineService.EngineUnavailableReason);
            return;
        }

        await InitializeWorkspaceAsync();
    }

    /// <summary>Load projects/settings from the engine. Safe to call again after the engine is
    /// located via the setup screen.</summary>
    private async Task InitializeWorkspaceAsync()
    {
        try
        {
            var projects = _engineService.LoadProjects();
            ViewModel.SetProjects(projects);
            ViewModel.Settings.SetSettings(_engineService.GetSettings());

            var firstProject = projects.FirstOrDefault();
            if (firstProject is not null)
            {
                await LoadProjectAsync(firstProject);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Corpus Studio", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private async void LocateEngineButton_Click(object sender, RoutedEventArgs e)
    {
        var folder = await FilePicker.PickFolderAsync(
            "Select the Corpus Studio engine folder (or the repo root that contains it)");
        if (folder is null)
        {
            return;
        }

        if (_engineService.TryLocateEngine(folder))
        {
            ViewModel.ClearEngineUnavailable();
            await InitializeWorkspaceAsync();
        }
        else
        {
            MessageBox.Show(
                this,
                "That folder does not contain the Corpus Studio engine "
                + "(expected corpus_studio/cli.py, or an engine/ subfolder).",
                "Engine not found",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
        }
    }

    private async void RetryEngineButton_Click(object sender, RoutedEventArgs e)
    {
        if (_engineService.TryReinitialize())
        {
            ViewModel.ClearEngineUnavailable();
            await InitializeWorkspaceAsync();
        }
        else
        {
            ViewModel.SetEngineUnavailable(_engineService.EngineUnavailableReason);
        }
    }

    private async void RebuildProjectIndexButton_Click(object sender, RoutedEventArgs e)
    {
        var selectedId = ViewModel.SelectedProject?.Id;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Rebuilding project index...");
            var result = await _engineService.RebuildProjectIndexAsync();
            ViewModel.SetProjects(await _engineService.LoadProjectsFromIndexAsync());
            ViewModel.SelectedProject = ViewModel.Projects
                .FirstOrDefault(project => project.Id == selectedId);
            ViewModel.ApplyProjectIndexRebuilt(result);
        }
        catch (Exception ex)
        {
            ViewModel.SetProjectIndexError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    // The Studio-sidebar "New Dataset Project" button and the Start Center both open the
    // single workspace wizard (v1.2.5 unification) — one creation UX, one on-disk result.
    private async void NewDatasetProjectButton_Click(object sender, RoutedEventArgs e) =>
        await LaunchNewProjectWizardAsync();

    // ---- Workspace shell + Start Center (v1.2.4 view layer) ----------------------

    private void ActivityHomeButton_Click(object sender, RoutedEventArgs e) => ViewModel.ShowStartCenter();

    private void ActivityFilesButton_Click(object sender, RoutedEventArgs e) => ViewModel.ShowFiles();

    private void ActivityStudioButton_Click(object sender, RoutedEventArgs e) => ViewModel.ShowStudio();

    private void ActivitySettingsButton_Click(object sender, RoutedEventArgs e) => ViewModel.ShowStudio();

    private void ActivitySearchButton_Click(object sender, RoutedEventArgs e)
    {
        ViewModel.ToggleSearchPanel();
        if (ViewModel.SearchPanelVisible)
        {
            // Focus the query box once the panel has laid out so the user can type immediately.
            Dispatcher.BeginInvoke(new Action(() => SearchQueryBox.Focus()), DispatcherPriority.Input);
        }
    }

    private void SearchCloseButton_Click(object sender, RoutedEventArgs e) =>
        ViewModel.ToggleSearchPanel();

    private async void SearchRunButton_Click(object sender, RoutedEventArgs e) =>
        await ViewModel.Search.RunAsync();

    private async void SearchQueryBox_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            e.Handled = true;
            await ViewModel.Search.RunAsync();
        }
    }

    private async void SearchResult_DoubleClick(object sender, MouseButtonEventArgs e) =>
        await OpenSelectedSearchResultAsync();

    private async void SearchResultsList_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            e.Handled = true;
            await OpenSelectedSearchResultAsync();
        }
    }

    /// <summary>Open the selected search result in the Explorer (switching to the Files view so
    /// the document tab is visible). Reuses the Explorer's node-open path. Enter and double-click
    /// both route here.</summary>
    private async Task OpenSelectedSearchResultAsync()
    {
        if (SearchResultsList.SelectedItem is not WorkspaceSearchMatch match)
        {
            return;
        }

        try
        {
            ViewModel.ShowFiles(); // sets the Explorer root + shows the Files view
            await ViewModel.Explorer.OpenNodeAsync(new WorkspaceTreeNode
            {
                RelativePath = match.RelativePath,
                IsDirectory = false,
            });
        }
        catch (Exception ex)
        {
            // A result can go stale between searching and opening (file moved/deleted/locked).
            // These handlers are async void, so an unguarded throw would crash the app — surface
            // it and stay put instead.
            MessageBox.Show(this, ex.Message, "Open Search Result", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void ProblemsButton_Click(object sender, RoutedEventArgs e) =>
        ViewModel.ToggleProblemsPanel();

    private void OutputButton_Click(object sender, RoutedEventArgs e) =>
        ViewModel.ToggleOutputPanel();

    private void ClearOutputButton_Click(object sender, RoutedEventArgs e) =>
        ViewModel.ClearOutputLog();

    private async void StartNewProject_Click(object sender, RoutedEventArgs e) =>
        await LaunchNewProjectWizardAsync();

    /// <summary>Shared New Project flow used by both the Start Center and the Studio sidebar:
    /// run the workspace wizard, and on success open the scaffolded folder as the active
    /// workspace. There is intentionally only one creation path.</summary>
    private async Task LaunchNewProjectWizardAsync()
    {
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            var schemas = await _engineService.GetSchemasAsync();
            Mouse.OverrideCursor = null;

            var wizard = new WorkspaceWizardWindow(schemas) { Owner = this, FilePicker = FilePicker };
            if (wizard.ShowDialog() == true && wizard.Result is not null)
            {
                await OpenWorkspaceFolder(wizard.Result.Folder);
            }
        }
        catch (Exception ex)
        {
            Mouse.OverrideCursor = null;
            MessageBox.Show(this, ex.Message, "New Project", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    private async void StartOpenFolder_Click(object sender, RoutedEventArgs e)
    {
        var folder = await FilePicker.PickFolderAsync("Open dataset workspace folder");
        if (folder is not null)
        {
            await RouteOpenFolder(folder);
        }
    }

    private void StartImport_Click(object sender, RoutedEventArgs e)
    {
        ViewModel.ShowStudio();
        MessageBox.Show(
            this,
            "Import runs from the Studio view (Examples / Quarantine). A Start Center import entry is on the roadmap.",
            "Import",
            MessageBoxButton.OK,
            MessageBoxImage.Information);
    }

    private async void RecentOpen_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is not RecentWorkspaceDisplayItem item)
        {
            return;
        }

        if (!System.IO.Directory.Exists(item.Path))
        {
            var choice = MessageBox.Show(
                this,
                $"'{item.Path}' no longer exists. Remove it from Recent Workspaces?",
                "Missing workspace",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No);
            if (choice == MessageBoxResult.Yes)
            {
                ViewModel.StartCenter.Remove(item.Path);
            }

            return;
        }

        await RouteOpenFolder(item.Path);
    }

    private void RecentPin_Click(object sender, RoutedEventArgs e)
    {
        e.Handled = true; // don't also trigger the card's open click
        if ((sender as FrameworkElement)?.DataContext is RecentWorkspaceDisplayItem item)
        {
            ViewModel.StartCenter.SetPinned(item.Path, !item.IsPinned);
        }
    }

    private void RecentRemove_Click(object sender, RoutedEventArgs e)
    {
        e.Handled = true;
        if ((sender as FrameworkElement)?.DataContext is RecentWorkspaceDisplayItem item)
        {
            ViewModel.StartCenter.Remove(item.Path);
        }
    }

    private void RecordRecentWorkspace(string path, string name, string? schemaId) =>
        ViewModel.StartCenter.RecordOpened(
            path, name, schemaId, DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture));

    // ---- Open / Initialize a folder as the active workspace (slice 3c) ------------

    /// <summary>Route an Open-Folder request through the four cases (see the prototype):
    /// open a manifest workspace; offer to initialize a dataset folder; offer to create in
    /// an empty folder; or refuse a random folder without mutating anything.</summary>
    private async Task RouteOpenFolder(string folder)
    {
        if (string.IsNullOrWhiteSpace(folder) || !System.IO.Directory.Exists(folder))
        {
            MessageBox.Show(this, "That folder no longer exists.", "Open Folder", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        switch (WorkspaceOpenRouting.Inspect(folder, new WorkspaceManifestService()))
        {
            case WorkspaceOpenAction.OpenManifest:
                await OpenWorkspaceFolder(folder);
                break;

            case WorkspaceOpenAction.OfferInitializeDataset:
                if (await ConfirmOpenAsync($"'{FolderDisplayName(folder)}' looks like a dataset but has no Corpus Studio metadata.\n\nInitialize it? This adds a .corpus/project.json manifest; your existing rows are left untouched."))
                {
                    await InitializeAndOpen(folder);
                }

                break;

            case WorkspaceOpenAction.OfferCreateEmpty:
                if (await ConfirmOpenAsync($"'{FolderDisplayName(folder)}' is empty.\n\nCreate a new Corpus Studio workspace here?"))
                {
                    await InitializeAndOpen(folder);
                }

                break;

            default:
                MessageBox.Show(
                    this,
                    "This folder isn't a Corpus Studio workspace and doesn't contain a dataset. Use Import to bring rows in, or pick another folder. Nothing was changed.",
                    "Open Folder",
                    MessageBoxButton.OK,
                    MessageBoxImage.Information);
                break;
        }
    }

    /// <summary>Open <paramref name="folder"/> as the active workspace: reuse or add a
    /// project pointing at it, load its examples/quarantine/quality, record it to Recent,
    /// and land in the Explorer. The engine only ever reads from this path.</summary>
    private async Task OpenWorkspaceFolder(string folder)
    {
        // Guard unsaved work before opening replaces the current draft and clears open
        // documents — this is the single chokepoint for every open path (Open Folder, Recent,
        // Initialize, New Project wizard), mirroring the project-switch and app-close prompts.
        // On decline, open nothing and leave the current workspace untouched.
        if (!await ConfirmDiscardUnsavedWorkAsync("Open this workspace"))
        {
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;

            var manifest = new WorkspaceManifestService().Read(folder);
            var (projectId, name, schemaId) =
                WorkspaceOpenRouting.DeriveOpenArgs(manifest.Manifest, FolderDisplayName(folder));

            var existing = ViewModel.Projects.FirstOrDefault(p => SamePath(p.ProjectPath, folder));
            if (existing is not null)
            {
                ViewModel.SelectProject(existing, schemaId);
            }
            else
            {
                ViewModel.AddProject(projectId, name, schemaId, schemaId, folder);
            }

            ViewModel.SetExamples(_engineService.LoadExamples(folder));
            ViewModel.Quarantine.SetItems(_engineService.LoadImportQuarantineItems(folder));
            Mouse.OverrideCursor = null;
            await RefreshQualityAsync(recordHistory: false);

            RecordRecentWorkspace(folder, name, schemaId);
            ViewModel.ShowFiles();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Open Workspace", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }
    }

    /// <summary>Write the workspace manifest into a folder (empty template — nothing but the
    /// manifest, existing files untouched) and open it.</summary>
    private async Task InitializeAndOpen(string folder)
    {
        var name = FolderDisplayName(folder);
        const string schema = WorkspaceOpenRouting.DefaultSchemaId;
        try
        {
            var templates = new ProjectTemplateService();
            var plan = templates.BuildPlan("empty", schema, name, name);
            var manifest = new WorkspaceProjectManifest
            {
                ProjectId = name,
                Name = name,
                SchemaId = schema,
                TemplateId = "empty",
                CreatedAt = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
            };

            var result = templates.Scaffold(folder, plan, manifest, allowNonEmpty: true);
            if (!result.Ok)
            {
                MessageBox.Show(this, result.Error, "Initialize Workspace", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            await OpenWorkspaceFolder(folder);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Initialize Workspace", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    // Confirm dialogs route through the head-agnostic IDialogService (Phase 0 of the Avalonia
    // migration): the same call ports to Avalonia by swapping the adapter, and the decision logic
    // (ShouldReplaceWorkspace) is already a pure, tested helper. Default = the safe/negative button.
    private Task<bool> ConfirmOpenAsync(string message) =>
        Dialogs.ConfirmAsync(message, "Open Folder", DialogButtons.YesNo, DialogSeverity.Question);

    /// <summary>Prompt to discard unsaved work (an edited draft or open documents) before an
    /// action that replaces the current workspace. Returns true to proceed. A no-op returning
    /// true when there is nothing unsaved, so a clean workspace opens without a prompt. Mirrors
    /// the project-switch (ProjectsListBox_SelectionChanged) and app-close guards.</summary>
    private Task<bool> ConfirmDiscardUnsavedWorkAsync(string actionDescription) =>
        WorkspaceOpenRouting.ShouldReplaceWorkspaceAsync(
            ViewModel.HasUnsavedWork,
            () => Dialogs.ConfirmAsync(
                "You have unsaved changes (an edited draft or open documents). "
                + $"{actionDescription} and discard them?",
                "Unsaved changes",
                DialogButtons.YesNo,
                DialogSeverity.Warning));

    private static string FolderDisplayName(string folder)
    {
        var name = System.IO.Path.GetFileName(folder.TrimEnd('/', '\\'));
        return string.IsNullOrWhiteSpace(name) ? "workspace" : name;
    }

    private static bool SamePath(string? left, string? right)
    {
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
        {
            return false;
        }

        return string.Equals(
            left.Replace('/', '\\').TrimEnd('\\'),
            right.Replace('/', '\\').TrimEnd('\\'),
            StringComparison.OrdinalIgnoreCase);
    }

    // ---- Universal Workspace Explorer (v1.2.4 view layer, slice 3b) ---------------

    private async void ExplorerTree_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
    {
        if (e.NewValue is WorkspaceTreeNode node)
        {
            await ViewModel.Explorer.OpenNodeAsync(node);
        }
    }

    private void DocTab_Click(object sender, MouseButtonEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is OpenWorkspaceDocument doc)
        {
            ViewModel.Explorer.ActiveDocument = doc;
        }
    }

    private void DocTabClose_Click(object sender, MouseButtonEventArgs e)
    {
        e.Handled = true; // don't also select the tab
        if ((sender as FrameworkElement)?.DataContext is not OpenWorkspaceDocument doc)
        {
            return;
        }

        if (doc.IsDirty)
        {
            var choice = MessageBox.Show(
                this,
                $"{doc.DisplayName} has unsaved changes. Close without saving?",
                "Unsaved changes",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No);
            if (choice != MessageBoxResult.Yes)
            {
                return;
            }
        }

        ViewModel.Explorer.CloseDocument(doc);
    }

    private async void ExplorerNewFile_Click(object sender, RoutedEventArgs e)
    {
        if (!EnsureWorkspace())
        {
            return;
        }

        var relative = PromptForRelativePath("New File", "New file path (relative to the workspace root):");
        if (relative is null)
        {
            return;
        }

        var error = await ViewModel.Explorer.CreateFileAsync(relative);
        if (error is not null)
        {
            MessageBox.Show(this, error, "New File", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void ExplorerNewFolder_Click(object sender, RoutedEventArgs e)
    {
        if (!EnsureWorkspace())
        {
            return;
        }

        var relative = PromptForRelativePath("New Folder", "New folder path (relative to the workspace root):");
        if (relative is null)
        {
            return;
        }

        var error = ViewModel.Explorer.CreateFolder(relative);
        if (error is not null)
        {
            MessageBox.Show(this, error, "New Folder", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void ExplorerRefresh_Click(object sender, RoutedEventArgs e) => ViewModel.Explorer.RefreshTree();

    private void ExplorerCollapseAll_Click(object sender, RoutedEventArgs e) => ViewModel.Explorer.CollapseAll();

    private async void ExplorerSave_Click(object sender, RoutedEventArgs e)
    {
        // Capture the target BEFORE saving (the active doc doesn't change on save, but be safe).
        var wasDatasetFile = ViewModel.HasActiveProject
            && ViewModel.Explorer.ActiveDocumentIsDatasetFile(ViewModel.ActiveProjectPath);

        var error = ViewModel.Explorer.SaveActiveDocument();
        if (error is not null)
        {
            MessageBox.Show(this, error, "Save", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        // Editing examples.jsonl in the editor changes the dataset — reload it (which invalidates
        // the stale debt grade) and re-check version integrity, so those badges stop asserting a
        // verdict the edit just outdated.
        if (wasDatasetFile && !string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetExamples(_engineService.LoadExamples(ViewModel.ActiveProjectPath));
            await RefreshDatasetVersionsAsync();
        }
    }

    private void ExplorerReveal_Click(object sender, RoutedEventArgs e)
    {
        var doc = ViewModel.Explorer.ActiveDocument;
        if (doc is null || string.IsNullOrEmpty(doc.FullPath))
        {
            return;
        }

        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(
                "explorer.exe", $"/select,\"{doc.FullPath}\"")
            {
                UseShellExecute = true,
            });
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Reveal", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void ExplorerCopyPath_Click(object sender, RoutedEventArgs e)
    {
        var relative = ViewModel.Explorer.ActiveRelPath;
        if (string.IsNullOrEmpty(relative))
        {
            return;
        }

        try
        {
            Clipboard.SetText(relative);
        }
        catch (Exception)
        {
            // Clipboard can be transiently locked by another app; ignore.
        }
    }

    private bool EnsureWorkspace()
    {
        if (ViewModel.Explorer.HasWorkspace)
        {
            return true;
        }

        MessageBox.Show(
            this,
            "Open or create a project first (from the Studio Dashboard or the Start Center).",
            "No workspace",
            MessageBoxButton.OK,
            MessageBoxImage.Information);
        return false;
    }

    /// <summary>Minimal single-line text prompt (no dependency on VisualBasic). Returns the
    /// trimmed input, or null if cancelled or empty.</summary>
    private string? PromptForRelativePath(string title, string prompt)
    {
        var input = new TextBox { MinWidth = 340, Margin = new Thickness(0, 8, 0, 0) };
        var ok = new Button { Content = "Create", IsDefault = true, MinWidth = 84, Margin = new Thickness(0, 0, 8, 0) };
        var cancel = new Button { Content = "Cancel", IsCancel = true, MinWidth = 84 };
        var buttons = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(0, 14, 0, 0),
        };
        buttons.Children.Add(ok);
        buttons.Children.Add(cancel);

        var panel = new StackPanel { Margin = new Thickness(18) };
        panel.Children.Add(new TextBlock { Text = prompt, TextWrapping = TextWrapping.Wrap });
        panel.Children.Add(input);
        panel.Children.Add(buttons);

        var dialog = new Window
        {
            Title = title,
            Content = panel,
            Owner = this,
            SizeToContent = SizeToContent.WidthAndHeight,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            ResizeMode = ResizeMode.NoResize,
            ShowInTaskbar = false,
        };
        ok.Click += (_, _) => dialog.DialogResult = true;
        dialog.Loaded += (_, _) => input.Focus();

        return dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(input.Text)
            ? input.Text.Trim()
            : null;
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

        var file = await FilePicker.PickFileAsync(
            "Import JSONL Dataset",
            new FilePickerFilter("JSONL files", "jsonl"),
            new FilePickerFilter("All files", "*"));
        if (file is null)
        {
            return;
        }

        await PreviewAndImportJsonlAsync(file);
    }

    private async void ImportFromHuggingFaceButton_Click(object sender, RoutedEventArgs e)
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

        // Map HF columns to the ACTIVE project's schema so imported rows match the project.
        IReadOnlyList<DatasetSchema> schemas;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            schemas = await _engineService.GetSchemasAsync();
        }
        catch (Exception ex)
        {
            ViewModel.SetImportError(ex.Message);
            return;
        }
        finally
        {
            Mouse.OverrideCursor = null;
        }

        var schema = schemas.FirstOrDefault(s => s.Id == ViewModel.ActiveSchemaId);
        if (schema is null)
        {
            MessageBox.Show(
                this,
                $"The active project's schema ('{ViewModel.ActiveSchemaId}') is not a built-in schema, so Hugging Face import can't map to it.",
                "Import from Hugging Face",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        var dialog = new HfImportWindow(_engineService, schema.Id, schema.Name, schema.Fields) { Owner = this };
        if (dialog.ShowDialog() != true || dialog.Result is null)
        {
            return;
        }

        // Hand the staging file to the SAME preview/confirm/append+quarantine flow as any
        // JSONL import (the desktop is the single writer of examples.jsonl), then clean up.
        var staging = dialog.Result.StagingPath;
        try
        {
            await PreviewAndImportJsonlAsync(staging);
        }
        finally
        {
            try
            {
                if (File.Exists(staging))
                {
                    File.Delete(staging);
                }
            }
            catch (IOException)
            {
                // best-effort temp cleanup
            }
        }
    }

    private void ExportCenterButton_Click(object sender, RoutedEventArgs e)
    {
        ExportJsonlButton.Focus();
    }

    private void DismissErrorButton_Click(object sender, RoutedEventArgs e)
    {
        ViewModel.DismissError();
    }

    private void CancelEngineButton_Click(object sender, RoutedEventArgs e)
    {
        _engineService.CancelRunningEngineCommand();
        ViewModel.SetBusy("Cancelling...");
    }

    private void GoToWritingStudioButton_Click(object sender, RoutedEventArgs e)
    {
        WritingStudioTab.IsSelected = true;
    }

    private void GoToSplitsButton_Click(object sender, RoutedEventArgs e)
    {
        SplitsTab.IsSelected = true;
    }

    private void GoToEvaluationButton_Click(object sender, RoutedEventArgs e)
    {
        EvaluationTab.IsSelected = true;
    }

    private void GoToTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        TrainingTab.IsSelected = true;
    }

    private void GoToDebtTab_Click(object sender, RoutedEventArgs e)
    {
        ViewModel.ShowStudio();
        DebtTab.IsSelected = true;
    }

    private async void CheckTrainingCompatibilityButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveSchemaId))
        {
            ViewModel.SetTrainingConfigError(
                "Create or select a dataset project before checking training compatibility."
            );
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Checking training compatibility...");
            var result = await _engineService.CheckTrainingCompatibilityAsync(
                ViewModel.ActiveSchemaId,
                ViewModel.TrainingFormat,
                ViewModel.TrainingTarget
            );
            ViewModel.ApplyTrainingCompatibility(result);
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingConfigError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async Task PreviewAndImportJsonlAsync(string importPath)
    {
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Importing dataset...");
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
            ViewModel.Quarantine.SetItems(
                _engineService.LoadImportQuarantineItems(ViewModel.ActiveProjectPath!)
            );
            await RefreshQualityAsync();

            // Snapshot the dataset change so an import is never silent. Best-effort: the import
            // already succeeded, so a failed snapshot is a note, not a failure — never claim a
            // snapshot that didn't happen.
            var snapshotNote = await AutoCaptureAfterImportAsync(importResult);

            Mouse.OverrideCursor = null;
            MessageBox.Show(
                this,
                BuildImportCompleteMessage(importResult, snapshotNote),
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
            ViewModel.ClearBusy();
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

    /// <summary>Snapshot the dataset as a version after an import that added rows, so the change
    /// is never silent. Best-effort: the import already succeeded, so a snapshot failure returns
    /// an honest note (never a claim that a snapshot happened) and does not fail the import.
    /// Returns a message line, or null when nothing was captured.</summary>
    private async Task<string?> AutoCaptureAfterImportAsync(ImportCommitResult importResult)
    {
        if (!importResult.ShouldAutoCapture || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return null;
        }

        string note;
        try
        {
            var version = await _engineService.CreateDatasetVersionAsync(
                ViewModel.ActiveProjectPath!,
                importResult.AutoCaptureLabel,
                "import"
            );
            note = $"Snapshotted this import as dataset version {version.VersionId}.";
        }
        catch (Exception ex)
        {
            note = $"Note: could not snapshot this import as a dataset version ({ex.Message}).";
        }

        await RefreshDatasetVersionsAsync();
        return note;
    }

    private static string BuildImportCompleteMessage(ImportCommitResult result, string? snapshotNote = null)
    {
        var lines = new List<string>
        {
            $"Imported {result.ImportedCount} row(s).",
        };

        if (result.SkippedDuplicateCount > 0)
        {
            lines.Add($"Skipped {result.SkippedDuplicateCount} duplicate row(s) already in the dataset.");
        }

        if (result.QuarantinedCount > 0)
        {
            lines.Add($"Quarantined {result.QuarantinedCount} rejected row(s).");
            if (!string.IsNullOrWhiteSpace(result.QuarantinePath))
            {
                lines.Add(result.QuarantinePath);
            }
        }

        if (!string.IsNullOrWhiteSpace(snapshotNote))
        {
            lines.Add(snapshotNote);
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
                ViewModel.WritingStudio.DraftText,
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
            ViewModel.ClearBusy();
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
                ViewModel.WritingStudio.DraftText,
                ViewModel.ActiveSchemaId
            );

            ViewModel.ApplyValidationReport(report);
            if (!report.Valid)
            {
                return;
            }

            var savedCount = _engineService.AppendDraftToProjectExamples(
                ViewModel.ActiveProjectPath,
                ViewModel.WritingStudio.DraftText
            );

            ViewModel.SetExamples(_engineService.LoadExamples(ViewModel.ActiveProjectPath));
            ViewModel.WritingStudio.MarkDraftClean(); // the draft is now persisted — no longer unsaved work

            // If this save repaired a quarantined row, clear that record so it doesn't orphan.
            var retried = ViewModel.TakePendingRetryItem();
            if (retried is not null)
            {
                _engineService.RemoveImportQuarantineItem(retried);
                ViewModel.Quarantine.SetItems(
                    _engineService.LoadImportQuarantineItems(ViewModel.ActiveProjectPath));
            }

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
            ViewModel.ClearBusy();
        }
    }

    private async void RunQualityButton_Click(object sender, RoutedEventArgs e)
    {
        await RefreshQualityAsync();
    }

    private async void RunGatesButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetGateError("Create or select a dataset project before running gates.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Running gates...");
            ViewModel.SetGateInProgress();
            var report = await _engineService.RunDatasetGatesAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId
            );
            ViewModel.ApplyGateReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetGateError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void RunChatGatesButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetGateError("Create or select a dataset project before running gates.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Running chat gates...");
            ViewModel.SetGateInProgress();
            var report = await _engineService.RunChatGatesAsync(ViewModel.ActiveProjectPath);
            ViewModel.ApplyGateReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetGateError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void RefreshProviderPoliciesButton_Click(object sender, RoutedEventArgs e)
    {
        await RefreshProviderPoliciesAsync();
    }

    private async void RunArenaButton_Click(object sender, RoutedEventArgs e)
    {
        var models = ArenaViewModel.ParseModelList(ViewModel.Arena.ArenaModelsInput);
        if (string.IsNullOrWhiteSpace(ViewModel.Arena.ArenaPromptsInput))
        {
            ViewModel.Arena.SetArenaError("Enter at least one prompt (one per line).");
            return;
        }
        if (models.Count == 0)
        {
            ViewModel.Arena.SetArenaError("Enter at least one model (comma or newline separated).");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Running arena...");
            ViewModel.Arena.SetArenaInProgress();
            var judge = string.IsNullOrWhiteSpace(ViewModel.Arena.ArenaJudgeModelInput)
                ? null
                : ViewModel.Arena.ArenaJudgeModelInput.Trim();
            var projectPath = ViewModel.HasActiveProject ? ViewModel.ActiveProjectPath : null;
            var report = await _engineService.RunArenaAsync(
                ViewModel.Arena.ArenaPromptsInput,
                models,
                judge,
                projectPath
            );
            ViewModel.Arena.ApplyArenaReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.Arena.SetArenaError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    // ---- Evaluation Suites tab (v1.3 M2) ---------------------------------------------

    private async void RefreshSuitesButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Suites.SetSuitesError("Create or select a dataset project first.");
            return;
        }
        try
        {
            ViewModel.Suites.IsSuitesBusy = true;
            ViewModel.Suites.ApplySuites(await _engineService.ListSuitesAsync(ViewModel.ActiveProjectPath));
        }
        catch (Exception ex)
        {
            ViewModel.Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            ViewModel.Suites.IsSuitesBusy = false;
        }
    }

    private async void NewSuiteButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Suites.SetSuitesError("Create or select a dataset project first.");
            return;
        }
        var name = NewSuiteNameBox.Text?.Trim() ?? string.Empty;
        if (name.Length == 0)
        {
            ViewModel.Suites.SetSuitesError("Enter a suite name to create.");
            return;
        }
        try
        {
            ViewModel.Suites.IsSuitesBusy = true;
            await _engineService.NewSuiteAsync(ViewModel.ActiveProjectPath, name);
            NewSuiteNameBox.Clear();
            ViewModel.Suites.ApplySuites(await _engineService.ListSuitesAsync(ViewModel.ActiveProjectPath));
            ViewModel.Suites.SetSuitesError($"Created suite '{name}'. Open evaluation_suites/{name}.json in Files to edit its cases.");
        }
        catch (Exception ex)
        {
            ViewModel.Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            ViewModel.Suites.IsSuitesBusy = false;
        }
    }

    private async void RunSuiteButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.Suites.CanRunSuite || ViewModel.Suites.SelectedSuite is not { } suite
            || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.Suites.IsSuitesBusy = true;
            ViewModel.SetBusy($"Running suite '{suite.Name}' (live backend evaluations)...");
            ViewModel.Suites.ApplySuiteReport(await _engineService.RunSuiteAsync(ViewModel.ActiveProjectPath, suite.Name));
        }
        catch (Exception ex)
        {
            ViewModel.Suites.SetSuitesError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.Suites.IsSuitesBusy = false;
            ViewModel.ClearBusy();
        }
    }

    private async Task RefreshProviderPoliciesAsync()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Settings.SetProviderPolicyError("Create or select a dataset project first.");
            return;
        }

        try
        {
            var policies = await _engineService.GetProviderPoliciesAsync(ViewModel.ActiveProjectPath);
            ViewModel.Settings.ApplyProviderPolicies(policies);
        }
        catch (Exception ex)
        {
            ViewModel.Settings.SetProviderPolicyError(ex.Message);
        }
    }

    private async void ApproveProviderGenerationButton_Click(object sender, RoutedEventArgs e)
    {
        await ApplyProviderApprovalAsync(revoke: false);
    }

    private async void RevokeProviderGenerationButton_Click(object sender, RoutedEventArgs e)
    {
        await ApplyProviderApprovalAsync(revoke: true);
    }

    private async Task ApplyProviderApprovalAsync(bool revoke)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Settings.SetProviderPolicyError("Create or select a dataset project first.");
            return;
        }

        var provider = (ProviderApprovalProviderComboBox.SelectedItem as ComboBoxItem)?.Content?.ToString()
            ?? string.Empty;
        var model = ProviderApprovalModelTextBox.Text?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(provider) || string.IsNullOrWhiteSpace(model))
        {
            ViewModel.Settings.SetProviderPolicyError("Choose a provider and enter a model name.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy(revoke ? "Revoking generation approval..." : "Approving generation...");
            await _engineService.ApproveProviderGenerationAsync(
                ViewModel.ActiveProjectPath,
                provider,
                model,
                revoke
            );
            await RefreshProviderPoliciesAsync();
        }
        catch (Exception ex)
        {
            ViewModel.Settings.SetProviderPolicyError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
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

        SaveLastPreparedAiAssistRewriteBatch();
        AiAssistTab.IsSelected = true;
    }

    private void SaveLastPreparedAiAssistRewriteBatch()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.RewriteBatches.SetAiAssistRewriteBatchError(
                "Create or select a dataset project before saving a prepared rewrite batch."
            );
            return;
        }

        if (!ViewModel.RewriteBatches.TryGetLastPreparedAiAssistRewriteBatch(out var batch, out var errorMessage))
        {
            ViewModel.RewriteBatches.SetAiAssistRewriteBatchError(errorMessage);
            return;
        }

        try
        {
            var savedBatch = _engineService.SaveAiAssistRewriteBatch(
                ViewModel.ActiveProjectPath,
                batch
            );
            ViewModel.RewriteBatches.SetAiAssistRewriteBatches(
                _engineService.LoadAiAssistRewriteBatches(ViewModel.ActiveProjectPath)
            );
            ViewModel.RewriteBatches.SelectedAiAssistRewriteBatch = ViewModel.RewriteBatches.AiAssistRewriteBatches
                .FirstOrDefault(item => item.BatchId == savedBatch.BatchId);
            ViewModel.RewriteBatches.ApplyAiAssistRewriteBatchSaved(savedBatch);
        }
        catch (Exception ex)
        {
            ViewModel.RewriteBatches.SetAiAssistRewriteBatchError(ex.Message);
        }
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

        RecordReviewedFixFromLastPrepared();
        WritingStudioTab.IsSelected = true;
        Dispatcher.BeginInvoke(new Action(() =>
        {
            DraftTextBox.Focus();
            DraftTextBox.Select(0, DraftTextBox.Text.Length);
        }));
    }

    private void RecordReviewedFixFromLastPrepared()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetReviewedFixError(
                "Create or select a dataset project before tracking a reviewed fix."
            );
            return;
        }

        if (!ViewModel.TryGetLastPreparedEvaluationFix(out var fix, out var errorMessage))
        {
            ViewModel.SetReviewedFixError(errorMessage);
            return;
        }

        try
        {
            var savedFix = _engineService.RecordReviewedFix(ViewModel.ActiveProjectPath, fix);
            ViewModel.SetReviewedFixes(
                _engineService.LoadReviewedFixes(ViewModel.ActiveProjectPath)
            );
            ViewModel.SelectedReviewedFix = ViewModel.ReviewedFixes
                .FirstOrDefault(item => item.FixId == savedFix.FixId);
            ViewModel.ApplyReviewedFixRecorded(savedFix);
        }
        catch (Exception ex)
        {
            ViewModel.SetReviewedFixError(ex.Message);
        }
    }

    private void SaveEvaluationFailureFilterButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetEvaluationFailureFilterError(
                "Create or select a dataset project before saving a failure filter."
            );
            return;
        }

        try
        {
            var savedFilter = _engineService.SaveEvaluationFailureFilter(
                ViewModel.ActiveProjectPath,
                ViewModel.BuildCurrentEvaluationFailureFilter()
            );
            ViewModel.SetEvaluationFailureFilters(
                _engineService.LoadEvaluationFailureFilters(ViewModel.ActiveProjectPath)
            );
            ViewModel.SelectedEvaluationFailureFilter = ViewModel.EvaluationFailureFilters
                .FirstOrDefault(item => item.Name == savedFilter.Name);
            ViewModel.ApplyEvaluationFailureFilterSaved(savedFilter);
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationFailureFilterError(ex.Message);
        }
    }

    private void ApplyEvaluationFailureFilterButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.SelectedEvaluationFailureFilter is null)
        {
            ViewModel.SetEvaluationFailureFilterError("Select a saved failure filter before applying it.");
            return;
        }

        ViewModel.ApplyEvaluationFailureFilter(ViewModel.SelectedEvaluationFailureFilter);
    }

    private void ResumeReviewedFixButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.ResumeReviewedFix())
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

    private void ReconcileReviewedFixesAfterRun(EvaluationRunResult result)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        try
        {
            ViewModel.SetReviewedFixes(
                _engineService.ReconcileReviewedFixes(ViewModel.ActiveProjectPath, result.Report.Results)
            );
            ViewModel.ApplyReviewedFixesReconciled();
        }
        catch (Exception ex)
        {
            ViewModel.SetReviewedFixError(ex.Message);
        }
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

    private async void ExportPreferenceForTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.PreferenceReview.SetPreferenceRankingExportError("Create or select a preference project before exporting.");
            return;
        }

        if (ViewModel.ActiveSchemaId != "preference")
        {
            ViewModel.PreferenceReview.SetPreferenceRankingExportError("Training export is available for preference projects.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Exporting preference data...");
            var result = await _engineService.ExportPreferenceForTrainingAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.PreferenceReview.PreferenceExportFormat
            );
            ViewModel.PreferenceReview.ApplyPreferenceTrainingExport(result);
        }
        catch (Exception ex)
        {
            ViewModel.PreferenceReview.SetPreferenceRankingExportError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private void ExportPreferenceRankingButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.PreferenceReview.SetPreferenceRankingExportError("Create or select a preference project before exporting rankings.");
            return;
        }

        try
        {
            var items = ViewModel.PreferenceReview.GetVisiblePreferenceReviewItems();
            var outputPath = _engineService.ExportPreferenceRanking(
                ViewModel.ActiveProjectPath,
                items
            );
            ViewModel.PreferenceReview.ApplyPreferenceRankingExport(outputPath, items.Count);
        }
        catch (Exception ex)
        {
            ViewModel.PreferenceReview.SetPreferenceRankingExportError(ex.Message);
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
            ViewModel.SetBusy("Checking evaluation backend...");
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
            ViewModel.ClearBusy();
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
            ViewModel.SetBusy("Loading models...");
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
            ViewModel.ClearBusy();
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
            ViewModel.SetBusy("Running evaluation...");
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
            ReconcileReviewedFixesAfterRun(result);
        }
        catch (Exception ex)
        {
            ViewModel.SetEvaluationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void RunBenchmarkButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetBenchmarkError("Create or select a dataset project before benchmarking.");
            return;
        }

        if (ViewModel.ActiveSchemaId is not ("instruction" or "chat"))
        {
            ViewModel.SetBenchmarkError("Evaluation Lab supports instruction and chat projects.");
            return;
        }

        var models = ViewModel.GetBenchmarkModels();
        if (models.Count == 0)
        {
            ViewModel.SetBenchmarkError("Enter at least one model to benchmark (one per line).");
            return;
        }

        if (!TryReadEvaluationOptions(
            out var backend,
            out _,
            out var baseUrl,
            out var limit,
            out var scoreThreshold,
            out var timeoutSeconds,
            out var errorMessage,
            requireModel: false
        ))
        {
            ViewModel.SetBenchmarkError(errorMessage);
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy($"Benchmarking {models.Count} model(s)...");
            var report = await _engineService.RunBenchmarkAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId,
                backend,
                models,
                baseUrl,
                limit,
                scoreThreshold,
                timeoutSeconds
            );
            ViewModel.ApplyBenchmarkReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.SetBenchmarkError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
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
            ViewModel.SetBusy("Rerunning evaluation...");
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
            ReconcileReviewedFixesAfterRun(result);

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
            ViewModel.ClearBusy();
        }
    }

    private async void CheckAiAssistBackendButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadBackendOptions(
            ViewModel.AiAssistConnection.AiAssistBackend,
            ViewModel.AiAssistConnection.AiAssistModel,
            ViewModel.AiAssistConnection.AiAssistBaseUrl,
            ViewModel.AiAssistConnection.AiAssistTimeoutSeconds,
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
            ViewModel.SetBusy("Checking AI Assist backend...");
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
            ViewModel.ClearBusy();
        }
    }

    private async void RefreshAiAssistModelsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!TryReadModelListOptions(
            ViewModel.AiAssistConnection.AiAssistBackend,
            ViewModel.AiAssistConnection.AiAssistBaseUrl,
            ViewModel.AiAssistConnection.AiAssistTimeoutSeconds,
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
            ViewModel.SetBusy("Loading models...");
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
            ViewModel.ClearBusy();
        }
    }

    private async void RunAiAssistButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetAiAssistError("Create or select a dataset project before running AI Assist.");
            return;
        }

        if (string.IsNullOrWhiteSpace(ViewModel.WritingStudio.DraftText))
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
            ViewModel.SetBusy("Running AI Assist...");
            ViewModel.SetAiAssistInProgress();
            var result = await _engineService.RunAiAssistAsync(
                ViewModel.WritingStudio.DraftText,
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
                ViewModel.WritingStudio.DraftText,
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
            ViewModel.ClearBusy();
        }
    }

    private void UseAiAssistSuggestionButton_Click(object sender, RoutedEventArgs e)
    {
        // Confirm-on-block: the pre-review candidate gate only INFORMS — a block never
        // auto-rejects, but the human should not pull a blocked candidate into the draft
        // by accident. Confirm-then-allow (never refuse); moving to the draft is not
        // acceptance. Covers both the queue-item and fresh-run paths (the VM checks the
        // active gate: selected queue item's gate, else the current run's gate).
        if (ViewModel.SelectedAiAssistCandidateGateBlocks && !ConfirmMoveBlockedCandidate())
        {
            return;
        }

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

    // Confirm-then-allow prompt when moving a gate-BLOCKED candidate into the draft.
    // Default button is No; a "Yes" only moves it to Writing Studio (still not accepted —
    // the human validates and saves). Never refuses outright.
    private bool ConfirmMoveBlockedCandidate()
    {
        return MessageBox.Show(
            this,
            "This generated candidate was BLOCKED by the pre-review gate "
            + "(schema / quality / PII). Moving it to Writing Studio does not accept it — "
            + "you still validate and save. Move anyway?",
            "Candidate gate: BLOCK",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning,
            MessageBoxResult.No) == MessageBoxResult.Yes;
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

    private void ResumeAiAssistRewriteBatchButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.ResumeAiAssistRewriteBatch())
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

    private async void LaunchTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        await RunTrainingAsync(ViewModel.TrainingLaunchArgv, ViewModel.TrainingLaunchCommand);
    }

    private async void ResumeTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        await RunTrainingAsync(ViewModel.TrainingResumeArgv, ViewModel.TrainingResumeCommand);
    }

    /// <summary>Shared launch core for fresh runs and resume-from-checkpoint: the
    /// user confirms the exact command, then the trainer is spawned and streamed.</summary>
    private async Task RunTrainingAsync(IReadOnlyList<string> argv, string command)
    {
        if (argv.Count == 0)
        {
            MessageBox.Show(
                this,
                "Generate a training config first — the launch command is produced with it.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        if (ViewModel.IsTrainingRunning)
        {
            return;
        }

        var confirm = MessageBox.Show(
            this,
            "This runs the trainer on your machine (it can use significant CPU/GPU for a long "
                + "time) with your installed tools. Corpus Studio only launches the command below "
                + "and streams its output.\n\n"
                + command
                + "\n\nRun it now?",
            "Launch training",
            MessageBoxButton.OKCancel,
            MessageBoxImage.Warning
        );
        if (confirm != MessageBoxResult.OK)
        {
            return;
        }

        // Capture the pre-training baseline (newest saved eval report, if any) so
        // the run can be compared before/after once the trained model is evaluated.
        try
        {
            var baseline = ViewModel.HasActiveProject && !string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath)
                ? _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath).FirstOrDefault()
                : null;
            ViewModel.SetTrainingBaseline(baseline);
        }
        catch
        {
            ViewModel.SetTrainingBaseline(null);
        }

        var workingDirectory = ViewModel.TrainingLaunchWorkingDirectory;
        var cts = new CancellationTokenSource();
        _trainingRunCts = cts;
        _trainingCancelRequested = false;
        while (_trainingLogQueue.TryDequeue(out _)) { } // discard any residual lines
        var runId = ViewModel.BeginTrainingRun();

        // Durable run record (v0.8): recorded to the project's training_runs/.
        var runProjectPath = ViewModel.HasActiveProject ? ViewModel.ActiveProjectPath : null;
        var runRecord = CreateAndSaveRunRecord(runProjectPath, argv);
        var terminalStatus = "interrupted";
        int? terminalExitCode = null;
        string? terminalNote = null;

        // Coalesce log lines: background reader threads enqueue, and a timer flushes
        // to the UI at a fixed rate so a chatty trainer can't flood the dispatcher.
        var logTimer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromMilliseconds(150),
        };
        logTimer.Tick += (_, _) => FlushTrainingLogQueue(runId);
        logTimer.Start();

        // Slow poll so checkpoints surface while the run is live (they appear
        // minutes apart; no hot timer).
        var checkpointTimer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromSeconds(15),
        };
        checkpointTimer.Tick += async (_, _) => await RefreshTrainingCheckpointsAsync();
        checkpointTimer.Start();

        try
        {
            var exitCode = await _trainingRunner.RunAsync(
                argv,
                workingDirectory,
                _trainingLogQueue.Enqueue,
                cts.Token,
                onStarted: (pid, startedAt) => RecordRunPid(runProjectPath, runRecord, pid, startedAt)
            );

            FlushTrainingLogQueue(runId);
            terminalExitCode = exitCode;
            if (_trainingCancelRequested)
            {
                terminalStatus = "cancelled";
                ViewModel.SetTrainingRunCancelled();
            }
            else
            {
                terminalStatus = exitCode == 0 ? "succeeded" : "failed";
                ViewModel.CompleteTrainingRun(exitCode);
            }
        }
        catch (OperationCanceledException)
        {
            FlushTrainingLogQueue(runId);
            terminalStatus = "cancelled";
            ViewModel.SetTrainingRunCancelled();
        }
        catch (Exception ex)
        {
            FlushTrainingLogQueue(runId);
            terminalStatus = "failed";
            terminalNote = ex.Message;
            ViewModel.SetTrainingRunError(ex.Message);
        }
        finally
        {
            logTimer.Stop();
            checkpointTimer.Stop();
            if (ReferenceEquals(_trainingRunCts, cts))
            {
                _trainingRunCts = null;
            }

            cts.Dispose();

            // A stopped/crashed run is exactly when surviving checkpoints matter.
            await RefreshTrainingCheckpointsAsync();

            // Finalize the durable record with fresh checkpoints + terminal status.
            await FinalizeRunRecord(runProjectPath, runRecord, terminalStatus, terminalExitCode, terminalNote);
        }
    }

    private TrainingRunRecord? CreateAndSaveRunRecord(string? projectPath, IReadOnlyList<string> argv)
    {
        if (string.IsNullOrWhiteSpace(projectPath))
        {
            return null;
        }

        var now = PythonEngineService.UtcNowIso();
        var record = new TrainingRunRecord
        {
            RunId = PythonEngineService.MintTrainingRunId(),
            CreatedAt = now,
            UpdatedAt = now,
            Status = "running",
            Target = ViewModel.TrainingTarget,
            BaseModel = ViewModel.TrainingBaseModel,
            ConfigPath = ViewModel.TrainingConfigPath,
            OutputDir = ViewModel.TrainingOutputDirectory,
            Argv = argv.ToList(),
            BeforeEvalPath = ViewModel.TrainingBaselineReport?.ReportPath,
        };
        TrySaveRunRecord(projectPath, record);
        return record;
    }

    private void RecordRunPid(string? projectPath, TrainingRunRecord? record, int pid, DateTime? startedAt)
    {
        if (string.IsNullOrWhiteSpace(projectPath) || record is null)
        {
            return;
        }

        record.Pid = pid;
        record.ProcessStartedAt = startedAt?.ToString("o", System.Globalization.CultureInfo.InvariantCulture);
        record.UpdatedAt = PythonEngineService.UtcNowIso();
        TrySaveRunRecord(projectPath, record);
    }

    private async Task FinalizeRunRecord(
        string? projectPath,
        TrainingRunRecord? record,
        string status,
        int? exitCode,
        string? note
    )
    {
        if (string.IsNullOrWhiteSpace(projectPath) || record is null)
        {
            return;
        }

        // Enumerate checkpoints against THIS run's captured output dir/config, not
        // the live VM (which the user may have changed by regenerating a config).
        try
        {
            if (!string.IsNullOrWhiteSpace(record.OutputDir))
            {
                var checkpoints = await _engineService.GetTrainingCheckpointsAsync(
                    record.OutputDir,
                    string.IsNullOrWhiteSpace(record.Target) ? "axolotl_yaml" : record.Target,
                    record.ConfigPath
                );
                record.Checkpoints = checkpoints.Checkpoints.ToList();
            }
        }
        catch
        {
            // Leave checkpoints as-is if enumeration fails.
        }

        record.Status = status;
        record.ExitCode = exitCode;
        record.UpdatedAt = PythonEngineService.UtcNowIso();
        if (!string.IsNullOrWhiteSpace(note))
        {
            record.Notes = note;
        }
        TrySaveRunRecord(projectPath, record);
    }

    private void TrySaveRunRecord(string projectPath, TrainingRunRecord record)
    {
        try
        {
            _engineService.SaveTrainingRunRecord(projectPath, record);
        }
        catch
        {
            // Recording must never break or interrupt the training run.
        }
    }

    private async void RefreshTrainingCheckpointsButton_Click(object sender, RoutedEventArgs e)
    {
        await RefreshTrainingCheckpointsAsync();
    }

    private async void GateTrainingRunButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetTrainingRunGateError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Running regression gate...");

            // Link the newest post-training eval (not the baseline) to the newest
            // run, carrying the model id so provenance can be verified.
            var baseline = ViewModel.TrainingBaselineReport;
            var after = _engineService.LoadEvaluationReportHistory(projectPath).FirstOrDefault(item =>
                baseline is null
                || !string.Equals(item.ReportPath, baseline.ReportPath, StringComparison.OrdinalIgnoreCase));

            var runId = after is not null
                ? _engineService.LinkAfterEvalToNewestRun(projectPath, after.ReportPath, after.Report.Model)
                : _engineService.LoadTrainingRunRecords(projectPath).FirstOrDefault()?.RunId;

            if (runId is null)
            {
                ViewModel.SetTrainingRunGateError("No training run has been recorded yet.");
                return;
            }

            var report = await _engineService.RunTrainingRunGateAsync(projectPath, runId);
            ViewModel.ApplyTrainingRunGate(report);
            ViewModel.ApplyTrainingRunHistory(_engineService.LoadTrainingRunRecords(projectPath));
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingRunGateError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private void RegisterArtifactFromRunButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Artifacts.SetArtifactError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            var run = _engineService.LoadTrainingRunRecords(projectPath).FirstOrDefault();
            if (run is null)
            {
                ViewModel.Artifacts.SetArtifactError("No training run has been recorded yet.");
                return;
            }
            if (string.IsNullOrWhiteSpace(run.OutputDir))
            {
                ViewModel.Artifacts.SetArtifactError("The latest run has no output directory to register.");
                return;
            }

            _engineService.RegisterArtifact(projectPath, run.RunId, run.OutputDir);
            RefreshArtifacts();
        }
        catch (Exception ex)
        {
            ViewModel.Artifacts.SetArtifactError(ex.Message);
        }
    }

    private async void KeepArtifactButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Artifacts.SelectedModelArtifact;
        if (selected is null)
        {
            ViewModel.Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Promote-gating artifact...");

            // Preview the promote gate so the user sees the verdict/reason...
            var report = await _engineService.GateArtifactAsync(projectPath, selected.Record.ArtifactId);
            var allowed = ViewModel.Artifacts.ApplyPromoteGate(report);
            if (!allowed)
            {
                return;
            }

            // ...then write through the ENGINE, which re-enforces the gate authoritatively — the
            // keep can never bypass it (a block throws and is surfaced below).
            await _engineService.PromoteArtifactAsync(projectPath, selected.Record.ArtifactId);
            RefreshArtifacts();
        }
        catch (Exception ex)
        {
            ViewModel.Artifacts.SetArtifactError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void ViewArtifactCardButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Artifacts.SelectedModelArtifact;
        if (selected is null)
        {
            ViewModel.Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Rendering weight card...");
            var markdown = await _engineService.GetWeightCardAsync(ViewModel.ActiveProjectPath, selected.Record.ArtifactId);
            ViewModel.Artifacts.SetArtifactDetail(markdown);
        }
        catch (Exception ex)
        {
            ViewModel.Artifacts.SetArtifactError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private void RejectArtifactButton_Click(object sender, RoutedEventArgs e) => SetSelectedArtifactStatus("rejected");

    private void RefreshArtifactsButton_Click(object sender, RoutedEventArgs e) => RefreshArtifacts();

    private void SetSelectedArtifactStatus(string status)
    {
        var selected = ViewModel.Artifacts.SelectedModelArtifact;
        if (selected is null)
        {
            ViewModel.Artifacts.SetArtifactError("Select an artifact first.");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        try
        {
            _engineService.UpdateArtifactStatus(ViewModel.ActiveProjectPath, selected.Record.ArtifactId, status);
            RefreshArtifacts();
        }
        catch (Exception ex)
        {
            ViewModel.Artifacts.SetArtifactError(ex.Message);
        }
    }

    private void RefreshArtifacts()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Artifacts.SetArtifactError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            // Resolve base_model live through run_id (never stored on the artifact).
            var runs = _engineService.LoadTrainingRunRecords(projectPath)
                .ToDictionary(r => r.RunId, r => r, StringComparer.Ordinal);
            var items = _engineService.LoadArtifacts(projectPath)
                .Select(entry => new ArtifactDisplayItem(
                    entry.Record,
                    entry.Integrity,
                    runs.TryGetValue(entry.Record.RunId, out var run) ? run.BaseModel : string.Empty))
                .ToList();
            ViewModel.Artifacts.ApplyArtifacts(items);
        }
        catch (Exception ex)
        {
            ViewModel.Artifacts.SetArtifactError(ex.Message);
        }
    }

    private async void CaptureDatasetVersionButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Capturing dataset version...");

            // Capture goes through the engine so the fingerprint is computed one way
            // (never reimplemented in C#, which would risk a phantom 'drifted').
            var record = await _engineService.CreateDatasetVersionAsync(
                projectPath, ViewModel.Versions.DatasetVersionLabel, "manual");
            ViewModel.Versions.DatasetVersionLabel = string.Empty;
            // Honest confirmation: a fingerprint-less record (missing/unreadable
            // dataset) must not read as a verified success.
            ViewModel.Versions.SetDatasetVersionDetail(VersionsViewModel.FormatCaptureConfirmation(record));
            await RefreshDatasetVersionsAsync();
        }
        catch (Exception ex)
        {
            ViewModel.Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void ViewDatasetVersionCardButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Versions.SelectedDatasetVersion;
        if (selected is null)
        {
            ViewModel.Versions.SetDatasetVersionError("Select a version first.");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Rendering version card...");

            var markdown = await _engineService.GetDatasetVersionCardAsync(
                ViewModel.ActiveProjectPath, selected.Record.VersionId);
            ViewModel.Versions.SetDatasetVersionDetail(markdown);
        }
        catch (Exception ex)
        {
            ViewModel.Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void RestoreDatasetVersionButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Versions.SelectedDatasetVersion;
        if (selected is null)
        {
            ViewModel.Versions.SetDatasetVersionError("Select a version to restore.");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;

        // Confirm — this overwrites the current dataset. The dialog is honest about the
        // undo capture and the canonical caveat.
        var confirm = MessageBox.Show(
            VersionsViewModel.BuildRestoreConfirmation(selected, ViewModel.Examples.Items.Count),
            "Restore version",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning);
        if (confirm != MessageBoxResult.Yes)
        {
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Restoring version (capturing the current dataset first)...");

            // The service captures the current dataset as an undo version, then restores
            // the selected version to a verified temp and atomically swaps it in. Any
            // failure before the swap leaves examples.jsonl untouched.
            var result = await _engineService.RestoreDatasetVersionInPlaceAsync(
                projectPath, selected.Record.VersionId, VersionsViewModel.BuildRestoreUndoLabel(selected));

            // Reflect the restored dataset (and the flipped integrity badges) in the UI.
            ViewModel.SetExamples(_engineService.LoadExamples(projectPath));
            ViewModel.Versions.ApplyRestoreResult(result);
            await RefreshDatasetVersionsAsync();
        }
        catch (Exception ex)
        {
            ViewModel.Versions.SetDatasetVersionError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private void SetDiffBaseButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Versions.SelectedDatasetVersion;
        if (selected is null)
        {
            ViewModel.Versions.SetDatasetVersionError("Select a version to set as the diff base.");
            return;
        }
        ViewModel.Versions.SetDatasetDiffBase(selected);
    }

    private async void DiffVersionsButton_Click(object sender, RoutedEventArgs e)
    {
        var selected = ViewModel.Versions.SelectedDatasetVersion;
        if (selected is null)
        {
            ViewModel.Versions.SetDatasetVersionError("Select a version to diff against the base.");
            return;
        }
        if (string.IsNullOrEmpty(ViewModel.Versions.DatasetDiffBaseId))
        {
            ViewModel.Versions.SetDatasetVersionError("Set a diff base first (select a version and click 'Set diff base').");
            return;
        }
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Diffing versions...");

            // Read-only: the engine compares the two versions' stored manifests and
            // refuses (throws) if either lacks stored rows.
            var markdown = await _engineService.GetDatasetVersionDiffAsync(
                ViewModel.ActiveProjectPath, ViewModel.Versions.DatasetDiffBaseId, selected.Record.VersionId);
            ViewModel.Versions.SetDatasetVersionDetail(markdown);
        }
        catch (Exception ex)
        {
            ViewModel.Versions.SetDatasetVersionError(ex.Message);
            // Replace any prior successful diff so a failure never leaves a stale result.
            ViewModel.Versions.SetDatasetVersionDetail("Diff failed: " + ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void RefreshDatasetVersionsButton_Click(object sender, RoutedEventArgs e)
    {
        await RefreshDatasetVersionsAsync();
    }

    private async Task RefreshDatasetVersionsAsync()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Versions.SetDatasetVersionError("Create or select a dataset project first.");
            return;
        }

        try
        {
            var items = await _engineService.LoadDatasetVersionsAsync(ViewModel.ActiveProjectPath);
            ViewModel.Versions.ApplyDatasetVersions(items);
        }
        catch (Exception ex)
        {
            ViewModel.Versions.SetDatasetVersionError(ex.Message);
        }
    }

    private async void RunDatasetDebtButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Debt.SetDebtError("Create or select a dataset project first.");
            return;
        }

        var projectPath = ViewModel.ActiveProjectPath;
        try
        {
            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Assessing dataset debt...");

            // Read-only: the engine computes and grades; the desktop parses and colors.
            var report = await _engineService.GetDatasetDebtAsync(projectPath);
            ViewModel.Debt.ApplyDebtReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.Debt.SetDebtError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private void RefreshTrainingRunsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.SetTrainingRunHistoryError("Create or select a dataset project first.");
            return;
        }

        try
        {
            var records = _engineService.LoadTrainingRunRecords(ViewModel.ActiveProjectPath);
            ViewModel.ApplyTrainingRunHistory(records);
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingRunHistoryError(ex.Message);
        }
    }

    private void CompareTrainingBaselineButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            return;
        }

        try
        {
            ViewModel.CompareTrainingBaseline(
                _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingConfigError(ex.Message);
        }
    }

    private async Task RefreshTrainingCheckpointsAsync()
    {
        var outputDirectory = ViewModel.TrainingOutputDirectory;
        if (string.IsNullOrWhiteSpace(outputDirectory))
        {
            return;
        }

        try
        {
            var result = await _engineService.GetTrainingCheckpointsAsync(
                outputDirectory,
                string.IsNullOrWhiteSpace(ViewModel.TrainingTarget) ? "axolotl" : ViewModel.TrainingTarget,
                ViewModel.TrainingConfigPath
            );
            ViewModel.ApplyTrainingCheckpoints(result);
        }
        catch
        {
            // Checkpoint refresh is advisory; never let it disrupt a run.
        }
    }

    private void FlushTrainingLogQueue(int runId)
    {
        if (_trainingLogQueue.IsEmpty)
        {
            return;
        }

        var batch = new List<string>();
        while (_trainingLogQueue.TryDequeue(out var line))
        {
            batch.Add(line);
        }

        ViewModel.AppendTrainingRunLogBatch(runId, batch);
    }

    private void StopTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        _trainingCancelRequested = true;
        _trainingRunCts?.Cancel();
    }

    private void MainWindow_Closing(object? sender, CancelEventArgs e)
    {
        if (ViewModel.HasUnsavedWork)
        {
            var unsavedChoice = MessageBox.Show(
                this,
                "You have unsaved changes (an edited draft or open documents). "
                + "Close Corpus Studio and discard them?",
                "Unsaved changes",
                MessageBoxButton.OKCancel,
                MessageBoxImage.Warning,
                MessageBoxResult.Cancel
            );
            if (unsavedChoice != MessageBoxResult.OK)
            {
                e.Cancel = true;
                return;
            }
        }

        if (!ViewModel.IsTrainingRunning)
        {
            return;
        }

        var result = MessageBox.Show(
            this,
            "A training run is in progress. Closing Corpus Studio will stop it. Close anyway?",
            "Training in progress",
            MessageBoxButton.OKCancel,
            MessageBoxImage.Warning
        );
        if (result != MessageBoxResult.OK)
        {
            e.Cancel = true;
            return;
        }

        // Best-effort: cancel the run and synchronously kill the trainer tree so it
        // is not orphaned when the app exits.
        _trainingCancelRequested = true;
        _trainingRunCts?.Cancel();
        _trainingRunner.TryKillCurrent();
    }

    private void CopyLaunchCommandButton_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(ViewModel.TrainingLaunchCommand))
        {
            MessageBox.Show(
                this,
                "Generate a training config first — the launch command is produced with it.",
                "Corpus Studio",
                MessageBoxButton.OK,
                MessageBoxImage.Information
            );
            return;
        }

        try
        {
            Clipboard.SetText(ViewModel.TrainingLaunchCommand);
        }
        catch (Exception ex)
        {
            ViewModel.SetTrainingConfigError($"Could not copy the launch command: {ex.Message}");
        }
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
            ViewModel.SetBusy("Generating training config...");
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
            ViewModel.ClearBusy();
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
            ViewModel.SetBusy("Generating dataset card...");
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
            ViewModel.ClearBusy();
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
        if (ViewModel.Quarantine.SelectedImportQuarantineItem is null)
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
            ViewModel.SetBusy("Exporting JSONL...");
            var removeDuplicates = ExportRemoveDuplicatesCheckBox.IsChecked == true;
            var removeLowInformation = ExportRemoveLowInformationCheckBox.IsChecked == true;
            var exportResult = await _engineService.ExportProjectExamplesAsync(
                ViewModel.ActiveProjectPath,
                ViewModel.ActiveSchemaId,
                removeDuplicates,
                removeLowInformation
            );

            var message = $"Exported {exportResult.OutputRows} row(s) to:\n{exportResult.OutputPath}";
            if (exportResult.Cleaned && exportResult.RemovedRows > 0)
            {
                message += $"\nRemoved {exportResult.RemovedRows} row(s) during cleaning.";
            }
            if (exportResult.Warnings.Count > 0)
            {
                message += "\n\nWarnings:\n- " + string.Join("\n- ", exportResult.Warnings);
            }

            MessageBox.Show(
                this,
                message,
                "Export Complete",
                MessageBoxButton.OK,
                exportResult.Warnings.Count > 0 ? MessageBoxImage.Warning : MessageBoxImage.Information
            );
        }
        catch (Exception ex)
        {
            ViewModel.SetValidationError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }

    private async void GenerateSplitsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Splits.SetSplitError("Create a dataset project before generating splits.");
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
                ViewModel.Splits.SetSplitError(errorMessage);
                return;
            }

            Mouse.OverrideCursor = Cursors.Wait;
            ViewModel.SetBusy("Generating splits...");
            ViewModel.Splits.SetSplitInProgress(trainRatio, validationRatio, seed);
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

            ViewModel.Splits.ApplySplitReport(report);
        }
        catch (Exception ex)
        {
            ViewModel.Splits.SetSplitError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
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
            ViewModel.Splits.SplitTrainPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var trainPercent
        ))
        {
            errorMessage = "Train split must be a number from 1 to 98.";
            return false;
        }

        if (!double.TryParse(
            ViewModel.Splits.SplitValidationPercent,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out var validationPercent
        ))
        {
            errorMessage = "Validation split must be a number from 0 to 98.";
            return false;
        }

        if (!int.TryParse(
            ViewModel.Splits.SplitSeed,
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
        out string errorMessage,
        bool requireModel = true
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

        if (requireModel && string.IsNullOrWhiteSpace(model))
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
        backend = ViewModel.AiAssistConnection.AiAssistBackend.Trim();
        model = ViewModel.AiAssistConnection.AiAssistModel.Trim();
        baseUrl = string.IsNullOrWhiteSpace(ViewModel.AiAssistConnection.AiAssistBaseUrl)
            ? null
            : ViewModel.AiAssistConnection.AiAssistBaseUrl.Trim();
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
            ViewModel.AiAssistConnection.AiAssistTimeoutSeconds,
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

    private bool _suppressProjectSelectionChange;

    private async void ProjectsListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (_suppressProjectSelectionChange || ViewModel.SelectedProject is null)
        {
            return;
        }

        // Guard unsaved work before the switch discards it (the switch resets the draft and
        // clears open documents). On cancel, revert the selection to the previous project.
        if (ViewModel.HasUnsavedWork)
        {
            var choice = MessageBox.Show(
                this,
                "You have unsaved changes (an edited draft or open documents). "
                + "Switch projects and discard them?",
                "Unsaved changes",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning,
                MessageBoxResult.No
            );
            if (choice != MessageBoxResult.Yes)
            {
                var previous = e.RemovedItems.Count > 0
                    ? e.RemovedItems[0] as DatasetProjectListItem
                    : null;
                _suppressProjectSelectionChange = true;
                ViewModel.SelectedProject = previous; // reverts the ListBox; re-fire is suppressed
                _suppressProjectSelectionChange = false;
                return;
            }
        }

        await LoadProjectAsync(ViewModel.SelectedProject);
    }

    private async Task LoadProjectAsync(DatasetProjectListItem project)
    {
        ViewModel.SelectProject(project);
        ViewModel.Splits.ApplySplitSettings(_engineService.LoadProjectSplitSettings(project.ProjectPath));
        ViewModel.ApplyLabSettings(_engineService.LoadProjectLabSettings(project.ProjectPath));
        ViewModel.SetExamples(_engineService.LoadExamples(project.ProjectPath));
        ViewModel.Quarantine.SetItems(_engineService.LoadImportQuarantineItems(project.ProjectPath));
        ViewModel.SetAiAssistReviewQueue(_engineService.LoadAiAssistReviewQueue(project.ProjectPath));
        ViewModel.SetAiAssistQueueViews(_engineService.LoadAiAssistQueueViews(project.ProjectPath));
        ViewModel.RewriteBatches.SetAiAssistRewriteBatches(
            _engineService.LoadAiAssistRewriteBatches(project.ProjectPath)
        );
        ViewModel.SetReviewedFixes(
            _engineService.LoadReviewedFixes(project.ProjectPath)
        );
        ViewModel.SetEvaluationFailureFilters(
            _engineService.LoadEvaluationFailureFilters(project.ProjectPath)
        );
        ClearAiAssistBulkUndoStack();
        ViewModel.SetEvaluationReportHistory(
            _engineService.LoadEvaluationReportHistory(project.ProjectPath)
        );
        await RefreshDatasetVersionsAsync();
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
            ViewModel.SetBusy("Running quality checks...");
            ViewModel.SetQualityInProgress();
            var report = await _engineService.BuildQualityReportAsync(ViewModel.ActiveProjectPath);
            if (recordHistory)
            {
                _engineService.SaveQualityHistoryEntry(ViewModel.ActiveProjectPath, report);
            }

            // Load a wider window than the 5-line text summary uses so the debt-trend chart
            // has enough points; the summary still shows only its most recent few internally.
            var history = _engineService.LoadQualityHistory(ViewModel.ActiveProjectPath, maxEntries: 30);
            ViewModel.ApplyQualityReport(report, history);
        }
        catch (Exception ex)
        {
            ViewModel.SetQualityError(ex.Message);
        }
        finally
        {
            Mouse.OverrideCursor = null;
            ViewModel.ClearBusy();
        }
    }
}
