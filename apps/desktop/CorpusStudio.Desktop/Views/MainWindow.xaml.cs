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
    private readonly PythonEngineService _engineService = new();

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

    // Startup + the engine-setup screen (locate/retry) now live on the shared view-model (#249):
    // the shell just delegates its Loaded event to StartWorkspaceAsync, and the setup-screen
    // buttons bind to LocateEngineCommand / RetryEngineCommand.
    private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
    {
        await ViewModel.StartWorkspaceAsync();
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
            await ViewModel.RefreshQualityAsync(recordHistory: false);

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

        // Build the reveal command through the hardened helper: the path is passed as an escaped
        // argument (not interpolated into a shell-parsed string) and must still exist on disk (#208).
        var startInfo = RevealInFileExplorer.BuildStartInfo(doc.FullPath);
        if (startInfo is null)
        {
            MessageBox.Show(this, "That file or folder no longer exists on disk.", "Reveal",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            System.Diagnostics.Process.Start(startInfo);
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
    private string? PromptForRelativePath(string title, string prompt, string initial = "", string okLabel = "Create")
    {
        var input = new TextBox { MinWidth = 340, Margin = new Thickness(0, 8, 0, 0), Text = initial };
        var ok = new Button { Content = okLabel, IsDefault = true, MinWidth = 84, Margin = new Thickness(0, 0, 8, 0) };
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
        dialog.Loaded += (_, _) => { input.Focus(); input.SelectAll(); };

        return dialog.ShowDialog() == true && !string.IsNullOrWhiteSpace(input.Text)
            ? input.Text.Trim()
            : null;
    }

    // --- Explorer tree node context-menu operations (issue #200). The node is the menu item's
    // inherited DataContext (the ContextMenu is placed on the node template). ---
    private void ExplorerNodeReveal_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is not WorkspaceTreeNode node)
        {
            return;
        }

        var startInfo = RevealInFileExplorer.BuildStartInfo(node.FullPath);
        if (startInfo is null)
        {
            MessageBox.Show(this, "That file or folder no longer exists on disk.", "Reveal",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            System.Diagnostics.Process.Start(startInfo);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Reveal", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void ExplorerNodeRename_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is not WorkspaceTreeNode node)
        {
            return;
        }

        var newName = PromptForRelativePath("Rename", $"New name for “{node.Name}”:", node.Name, "Rename");
        if (newName is null)
        {
            return;
        }

        var error = ViewModel.Explorer.RenameNode(node, newName);
        if (error is not null)
        {
            MessageBox.Show(this, error, "Rename", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private async void ExplorerNodeDelete_Click(object sender, RoutedEventArgs e)
    {
        if ((sender as FrameworkElement)?.DataContext is not WorkspaceTreeNode node)
        {
            return;
        }

        var kind = node.IsDirectory ? "folder" : "file";
        var confirmed = await Dialogs.ConfirmAsync(
            $"Permanently delete the {kind} “{node.Name}”? This cannot be undone.",
            "Delete",
            DialogButtons.YesNo,
            DialogSeverity.Warning,
            defaultAffirmative: false);
        if (!confirmed)
        {
            return;
        }

        var error = ViewModel.Explorer.DeleteNode(node);
        if (error is not null)
        {
            MessageBox.Show(this, error, "Delete", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }


    private void ExportCenterButton_Click(object sender, RoutedEventArgs e)
    {
        ExportJsonlButton.Focus();
    }


    private void CancelEngineButton_Click(object sender, RoutedEventArgs e)
    {
        _engineService.CancelRunningEngineCommand();
        ViewModel.SetBusy("Cancelling...");
    }







    /// <summary>Snapshot the dataset as a version after an import that added rows, so the change
    /// is never silent. Best-effort: the import already succeeded, so a snapshot failure returns
    /// an honest note (never a claim that a snapshot happened) and does not fail the import.
    /// Returns a message line, or null when nothing was captured.</summary>


    // ---- Evaluation Suites tab (v1.3 M2) ---------------------------------------------



    /// <summary>Load the effective gate thresholds into the Settings editor (issue #198).</summary>
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
            ViewModel.Evaluation.SetEvaluationFailureFilterError(
                "Create or select a dataset project before saving a failure filter."
            );
            return;
        }

        try
        {
            var savedFilter = _engineService.SaveEvaluationFailureFilter(
                ViewModel.ActiveProjectPath,
                ViewModel.Evaluation.BuildCurrentEvaluationFailureFilter()
            );
            ViewModel.Evaluation.SetEvaluationFailureFilters(
                _engineService.LoadEvaluationFailureFilters(ViewModel.ActiveProjectPath)
            );
            ViewModel.Evaluation.SelectedEvaluationFailureFilter = ViewModel.Evaluation.EvaluationFailureFilters
                .FirstOrDefault(item => item.Name == savedFilter.Name);
            ViewModel.Evaluation.ApplyEvaluationFailureFilterSaved(savedFilter);
        }
        catch (Exception ex)
        {
            ViewModel.Evaluation.SetEvaluationFailureFilterError(ex.Message);
        }
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




    private void UseAiAssistSuggestionButton_Click(object sender, RoutedEventArgs e)
    {
        // Confirm-on-block: the pre-review candidate gate only INFORMS — a block never
        // auto-rejects, but the human should not pull a blocked candidate into the draft
        // by accident. Confirm-then-allow (never refuse); moving to the draft is not
        // acceptance. Covers both the queue-item and fresh-run paths (the VM checks the
        // active gate: selected queue item's gate, else the current run's gate).
        if (ViewModel.AiAssist.SelectedAiAssistCandidateGateBlocks && !ConfirmMoveBlockedCandidate())
        {
            return;
        }

        if (ViewModel.AiAssist.SelectedAiAssistReviewQueueItem is null
            || string.IsNullOrWhiteSpace(ViewModel.AiAssist.SelectedAiAssistReviewQueueItem.SuggestedJsonl))
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
            ViewModel.AiAssist.SetAiAssistQueueError("Create or select a dataset project before saving an AI Assist queue view.");
            return;
        }

        var view = ViewModel.AiAssist.BuildCurrentAiAssistQueueView();
        if (string.IsNullOrWhiteSpace(view.Name))
        {
            ViewModel.AiAssist.SetAiAssistQueueError("Name the AI Assist queue view before saving.");
            return;
        }

        try
        {
            var savedView = _engineService.SaveAiAssistQueueView(ViewModel.ActiveProjectPath, view);
            ViewModel.AiAssist.SetAiAssistQueueViews(
                _engineService.LoadAiAssistQueueViews(ViewModel.ActiveProjectPath)
            );
            ViewModel.AiAssist.SelectedAiAssistQueueView = ViewModel.AiAssist.AiAssistQueueViews
                .FirstOrDefault(item => string.Equals(
                    item.Name,
                    savedView.Name,
                    StringComparison.OrdinalIgnoreCase
                ));
            ViewModel.AiAssist.ApplyAiAssistQueueViewSaved(savedView);
        }
        catch (Exception ex)
        {
            ViewModel.AiAssist.SetAiAssistQueueError(ex.Message);
        }
    }

    private void LoadAiAssistQueueViewButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.AiAssist.SelectedAiAssistQueueView is null)
        {
            ViewModel.AiAssist.SetAiAssistQueueError("Select a saved AI Assist queue view before loading.");
            return;
        }

        var view = ViewModel.AiAssist.SelectedAiAssistQueueView;
        ViewModel.AiAssist.ApplyAiAssistQueueView(view);
        ViewModel.AiAssist.ApplyAiAssistQueueViewLoaded(view);
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
            ViewModel.AiAssist.SetAiAssistQueueError("Create or select a dataset project before updating AI Assist review state.");
            return false;
        }

        if (ViewModel.AiAssist.SelectedAiAssistReviewQueueItem is null)
        {
            ViewModel.AiAssist.SetAiAssistQueueError("Select an AI Assist review before updating its state.");
            return false;
        }

        try
        {
            var reviewId = ViewModel.AiAssist.SelectedAiAssistReviewQueueItem.ReviewId;
            var updatedItem = _engineService.UpdateAiAssistReviewState(
                ViewModel.ActiveProjectPath,
                reviewId,
                reviewState
            );
            ViewModel.AiAssist.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            ViewModel.AiAssist.SelectedAiAssistReviewQueueItem = ViewModel.AiAssist.AiAssistReviewQueue
                .FirstOrDefault(item => item.ReviewId == reviewId);
            ViewModel.AiAssist.ApplyAiAssistReviewState(updatedItem);
            ViewModel.AiAssist.ClearBulkUndoStack();
            return true;
        }
        catch (Exception ex)
        {
            ViewModel.AiAssist.SetAiAssistQueueError(ex.Message);
            return false;
        }
    }

    private void BulkMarkVisibleAiAssistReviews(string reviewState)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.AiAssist.SetAiAssistQueueError("Create or select a dataset project before updating AI Assist review state.");
            return;
        }

        var reviewIds = ViewModel.AiAssist.GetVisibleAiAssistReviewIds();
        var previousStates = ViewModel.AiAssist.GetVisibleAiAssistReviewStates();
        if (reviewIds.Count == 0)
        {
            ViewModel.AiAssist.SetAiAssistQueueError("No AI Assist reviews match the current filter.");
            return;
        }

        try
        {
            var updatedCount = _engineService.UpdateAiAssistReviewStates(
                ViewModel.ActiveProjectPath,
                reviewIds,
                reviewState
            );
            ViewModel.AiAssist.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            ViewModel.AiAssist.PushBulkUndoStep(previousStates);
            ViewModel.AiAssist.ApplyAiAssistBulkReviewState(
                updatedCount,
                reviewState,
                ViewModel.AiAssist.BulkUndoStackDepth
            );
        }
        catch (Exception ex)
        {
            ViewModel.AiAssist.SetAiAssistQueueError(ex.Message);
        }
    }

    private void UndoBulkAiAssistReviews()
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.AiAssist.SetAiAssistQueueError("Create or select a dataset project before undoing AI Assist bulk triage.");
            return;
        }

        var previousStates = ViewModel.AiAssist.PeekBulkUndoStep();
        if (previousStates is null)
        {
            ViewModel.AiAssist.SetAiAssistQueueError("No AI Assist bulk triage action is available to undo.");
            return;
        }

        try
        {
            var restoredCount = _engineService.UpdateAiAssistReviewStates(
                ViewModel.ActiveProjectPath,
                previousStates
            );
            ViewModel.AiAssist.SetAiAssistReviewQueue(
                _engineService.LoadAiAssistReviewQueue(ViewModel.ActiveProjectPath)
            );
            ViewModel.AiAssist.RemoveLastBulkUndoStep();
            ViewModel.AiAssist.ApplyAiAssistBulkUndoReviewState(
                restoredCount,
                ViewModel.AiAssist.BulkUndoStackDepth
            );
        }
        catch (Exception ex)
        {
            ViewModel.AiAssist.SetAiAssistQueueError(ex.Message);
        }
    }

    private void SaveEvaluationReviewButton_Click(object sender, RoutedEventArgs e)
    {
        if (ViewModel.Evaluation.SelectedEvaluationReportHistoryItem is null)
        {
            ViewModel.Evaluation.SetEvaluationReviewError("Select an evaluation report before saving review notes.");
            return;
        }

        if (ViewModel.Evaluation.SelectedEvaluationExampleResult is null)
        {
            ViewModel.Evaluation.SetEvaluationReviewError("Select an evaluation example before saving review notes.");
            return;
        }

        if (!TryReadEvaluationManualReview(
            out var manualScore,
            out var manualNotes,
            out var errorMessage
        ))
        {
            ViewModel.Evaluation.SetEvaluationReviewError(errorMessage);
            return;
        }

        try
        {
            var exampleId = ViewModel.Evaluation.SelectedEvaluationExampleResult.ExampleId;
            var updatedItem = _engineService.SaveEvaluationManualReview(
                ViewModel.Evaluation.SelectedEvaluationReportHistoryItem,
                exampleId,
                manualScore,
                manualNotes
            );

            if (!string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
            {
                ViewModel.Evaluation.SetEvaluationReportHistory(
                    _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
                );
                ViewModel.Evaluation.SelectedEvaluationReportHistoryItem = ViewModel.Evaluation.EvaluationReportHistory
                    .FirstOrDefault(item => item.ReportPath == updatedItem.ReportPath);
            }

            ViewModel.Evaluation.SelectedEvaluationExampleResult = ViewModel.Evaluation.EvaluationResults
                .FirstOrDefault(result => result.ExampleId == exampleId);
            ViewModel.Evaluation.ApplySavedEvaluationManualReview(updatedItem);
        }
        catch (Exception ex)
        {
            ViewModel.Evaluation.SetEvaluationReviewError(ex.Message);
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


    // Version refresh now lives on the shared view-model (used by CaptureDatasetVersionCommand and the
    // project-switch load); the code-behind callers delegate to it.
    private Task RefreshDatasetVersionsAsync() => ViewModel.RefreshDatasetVersionsAsync();


    private async void RefreshTrainingRunsButton_Click(object sender, RoutedEventArgs e)
    {
        if (!ViewModel.HasActiveProject || string.IsNullOrWhiteSpace(ViewModel.ActiveProjectPath))
        {
            ViewModel.Training.SetTrainingRunHistoryError("Create or select a dataset project first.");
            return;
        }

        IReadOnlyList<TrainingRunRecord> records;
        try
        {
            records = _engineService.LoadTrainingRunRecords(ViewModel.ActiveProjectPath);
            ViewModel.Training.ApplyTrainingRunHistory(records);
        }
        catch (Exception ex)
        {
            ViewModel.Training.SetTrainingRunHistoryError(ex.Message);
            return;
        }

        // Also refresh the close-the-loop plan for the newest run (the plan itself is
        // honest about a run that hasn't succeeded yet).
        if (records.Count > 0)
        {
            await ViewModel.ShowEvalHandoffAsync(ViewModel.ActiveProjectPath, records[0].RunId);
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
            ViewModel.Training.CompareTrainingBaseline(
                _engineService.LoadEvaluationReportHistory(ViewModel.ActiveProjectPath)
            );
        }
        catch (Exception ex)
        {
            ViewModel.Training.SetTrainingConfigError(ex.Message);
        }
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

        if (!ViewModel.Training.IsTrainingRunning)
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
        ViewModel.StopTrainingForShutdown();
    }

    private void CopyLaunchCommandButton_Click(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(ViewModel.Training.TrainingLaunchCommand))
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
            Clipboard.SetText(ViewModel.Training.TrainingLaunchCommand);
        }
        catch (Exception ex)
        {
            ViewModel.Training.SetTrainingConfigError($"Could not copy the launch command: {ex.Message}");
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

    private bool TryReadEvaluationManualReview(
        out double? manualScore,
        out string? manualNotes,
        out string errorMessage
    )
    {
        manualScore = null;
        manualNotes = string.IsNullOrWhiteSpace(ViewModel.Evaluation.EvaluationManualNotes)
            ? null
            : ViewModel.Evaluation.EvaluationManualNotes.Trim();
        errorMessage = string.Empty;

        if (string.IsNullOrWhiteSpace(ViewModel.Evaluation.EvaluationManualScore))
        {
            return true;
        }

        if (!double.TryParse(
            ViewModel.Evaluation.EvaluationManualScore,
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

    // The project-load flow now lives on the shared view-model (startup workspace-init + the
    // project-selection change both use it); the code-behind callers delegate to it.
    private Task LoadProjectAsync(DatasetProjectListItem project) => ViewModel.LoadProjectAsync(project);

}
