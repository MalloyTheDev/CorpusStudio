using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Backs the New Project wizard (v1.2.4 Workspace System, slice 3c-2): schema +
/// template pickers with a live preview of the folder structure that will be scaffolded.
/// Pure/testable — no dialogs or engine. The window does the actual scaffold + open.</summary>
public sealed class WorkspaceWizardViewModel : INotifyPropertyChanged
{
    private static readonly Regex CollapseUnderscores = new("_+", RegexOptions.Compiled);

    private readonly ProjectTemplateService _templates;
    private bool _projectIdTouched;

    public WorkspaceWizardViewModel(IReadOnlyList<DatasetSchema> schemas, ProjectTemplateService? templates = null)
    {
        _templates = templates ?? new ProjectTemplateService();
        Schemas = new ObservableCollection<DatasetSchema>(schemas ?? Array.Empty<DatasetSchema>());
        Templates = new ObservableCollection<WorkspaceTemplateDefinition>(_templates.Templates);

        _selectedSchema = Schemas.FirstOrDefault(s => s.Id == "instruction") ?? Schemas.FirstOrDefault();
        _selectedTemplate = Templates.FirstOrDefault(t => t.Id == "standard") ?? Templates.FirstOrDefault();
        RebuildPreview();
    }

    public ObservableCollection<DatasetSchema> Schemas { get; }
    public ObservableCollection<WorkspaceTemplateDefinition> Templates { get; }
    public ObservableCollection<WorkspacePreviewLine> PreviewLines { get; } = new();

    private string _projectName = string.Empty;
    public string ProjectName
    {
        get => _projectName;
        set
        {
            if (SetField(ref _projectName, value))
            {
                if (!_projectIdTouched)
                {
                    SetProjectId(Slugify(value), markTouched: false);
                }

                OnFormChanged();
            }
        }
    }

    private string _projectId = string.Empty;
    public string ProjectId
    {
        get => _projectId;
        set => SetProjectId(value, markTouched: true);
    }

    private void SetProjectId(string value, bool markTouched)
    {
        if (markTouched)
        {
            _projectIdTouched = true;
        }

        if (SetField(ref _projectId, value, nameof(ProjectId)))
        {
            OnFormChanged();
        }
    }

    private string _location = string.Empty;
    public string Location
    {
        get => _location;
        set
        {
            if (SetField(ref _location, value))
            {
                OnChanged(nameof(TargetFolder));
                OnChanged(nameof(CanCreate));
                OnChanged(nameof(ValidationMessage));
            }
        }
    }

    private DatasetSchema? _selectedSchema;
    public DatasetSchema? SelectedSchema
    {
        get => _selectedSchema;
        set
        {
            if (SetField(ref _selectedSchema, value))
            {
                RebuildPreview();
                OnChanged(nameof(CanCreate));
                OnChanged(nameof(ValidationMessage));
            }
        }
    }

    private WorkspaceTemplateDefinition? _selectedTemplate;
    public WorkspaceTemplateDefinition? SelectedTemplate
    {
        get => _selectedTemplate;
        set
        {
            if (SetField(ref _selectedTemplate, value))
            {
                RebuildPreview();
                OnChanged(nameof(CanCreate));
                OnChanged(nameof(ValidationMessage));
            }
        }
    }

    private string? _previewNote;
    public string? PreviewNote
    {
        get => _previewNote;
        private set
        {
            if (SetField(ref _previewNote, value))
            {
                OnChanged(nameof(HasPreviewNote));
            }
        }
    }

    public bool HasPreviewNote => !string.IsNullOrWhiteSpace(_previewNote);

    /// <summary>The sanitized folder-name segment for the project (from the id, or the name
    /// when the id is blank).</summary>
    public string SafeProjectId =>
        WorkspacePathSafety.SanitizeSegmentName(string.IsNullOrWhiteSpace(_projectId) ? _projectName : _projectId);

    /// <summary>Where the project folder will be created: &lt;location&gt;/&lt;safe id&gt;.</summary>
    public string TargetFolder =>
        string.IsNullOrWhiteSpace(_location) || string.IsNullOrWhiteSpace(SafeProjectId)
            ? string.Empty
            : Path.Combine(_location, SafeProjectId);

    public bool CanCreate =>
        !string.IsNullOrWhiteSpace(_projectName)
        && !string.IsNullOrWhiteSpace(SafeProjectId)
        && _selectedSchema is not null
        && _selectedTemplate is not null
        && !string.IsNullOrWhiteSpace(_location);

    public string ValidationMessage
    {
        get
        {
            if (string.IsNullOrWhiteSpace(_projectName))
            {
                return "Enter a project name.";
            }

            if (string.IsNullOrWhiteSpace(SafeProjectId))
            {
                return "Enter a valid project id (letters, numbers, - or _).";
            }

            if (_selectedSchema is null)
            {
                return "Choose a dataset schema.";
            }

            if (_selectedTemplate is null)
            {
                return "Choose a project template.";
            }

            return string.IsNullOrWhiteSpace(_location)
                ? "Choose a location (Browse…)."
                : $"Will create: {TargetFolder}";
        }
    }

    public WorkspaceProjectManifest BuildManifest() => new()
    {
        ProjectId = SafeProjectId,
        Name = string.IsNullOrWhiteSpace(_projectName) ? SafeProjectId : _projectName.Trim(),
        SchemaId = _selectedSchema?.Id ?? "instruction",
        TemplateId = _selectedTemplate?.Id,
        CreatedAt = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
    };

    /// <summary>The scaffold plan for the current selections (the window uses this for the
    /// actual write; the preview mirrors it).</summary>
    public WorkspaceScaffoldPlan BuildPlan()
    {
        var id = string.IsNullOrWhiteSpace(SafeProjectId) ? "my_dataset" : SafeProjectId;
        var name = string.IsNullOrWhiteSpace(_projectName) ? "My Dataset" : _projectName.Trim();
        return _templates.BuildPlan(
            _selectedTemplate?.Id ?? "standard",
            _selectedSchema?.Id ?? "instruction",
            name,
            id,
            _selectedSchema?.ExampleText);
    }

    private void OnFormChanged()
    {
        RebuildPreview();
        OnChanged(nameof(SafeProjectId));
        OnChanged(nameof(TargetFolder));
        OnChanged(nameof(CanCreate));
        OnChanged(nameof(ValidationMessage));
    }

    private void RebuildPreview()
    {
        PreviewLines.Clear();
        if (_selectedSchema is null || _selectedTemplate is null)
        {
            PreviewNote = null;
            return;
        }

        var rootName = string.IsNullOrWhiteSpace(SafeProjectId) ? "my_dataset" : SafeProjectId;
        var plan = BuildPlan();
        foreach (var line in RenderPlan(plan, rootName))
        {
            PreviewLines.Add(line);
        }

        PreviewNote = plan.Note;
    }

    /// <summary>Flatten a plan into an indented, deterministic folder-first listing.</summary>
    private static IEnumerable<WorkspacePreviewLine> RenderPlan(WorkspaceScaffoldPlan plan, string rootName)
    {
        yield return new WorkspacePreviewLine(rootName + "/", isFolder: true, depth: 0);

        var entries = new List<(string Path, bool IsFolder)>();
        foreach (var dir in plan.Directories)
        {
            var norm = dir.Replace('\\', '/').Trim('/');
            if (norm.Length > 0)
            {
                entries.Add((norm, true));
            }
        }

        foreach (var file in plan.Files)
        {
            var norm = file.RelativePath.Replace('\\', '/').Trim('/');
            if (norm.Length > 0)
            {
                entries.Add((norm, false));
            }
        }

        // Folders before files at each level, then alphabetical — a stable, readable tree.
        foreach (var (path, isFolder) in entries
                     .OrderBy(e => e.Path, StringComparer.OrdinalIgnoreCase)
                     .OrderByDescending(e => e.IsFolder)
                     .ThenBy(e => e.Path, StringComparer.OrdinalIgnoreCase))
        {
            var leaf = path.Split('/').Last();
            var depth = path.Count(c => c == '/') + 1;
            yield return new WorkspacePreviewLine(leaf + (isFolder ? "/" : string.Empty), isFolder, depth);
        }
    }

    /// <summary>Slugify a project name into an id: lowercase, non-alphanumeric → '_',
    /// collapse runs, trimmed.</summary>
    private static string Slugify(string value)
    {
        var lower = (value ?? string.Empty).Trim().ToLowerInvariant();
        var chars = lower.Select(c => char.IsLetterOrDigit(c) ? c : '_').ToArray();
        var slug = CollapseUnderscores.Replace(new string(chars), "_").Trim('_');
        return slug;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnChanged(name);
        return true;
    }

    private void OnChanged(string? name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>One line in the wizard's folder-structure preview.</summary>
public sealed class WorkspacePreviewLine
{
    public WorkspacePreviewLine(string text, bool isFolder, int depth)
    {
        Text = text;
        IsFolder = isFolder;
        Depth = depth;
    }

    public string Text { get; }
    public bool IsFolder { get; }
    public int Depth { get; }

    /// <summary>Left indent in device-independent pixels for the view.</summary>
    public double Indent => Depth * 16.0;
}
