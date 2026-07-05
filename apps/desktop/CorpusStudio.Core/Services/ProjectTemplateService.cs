using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>A file the scaffold will create, with its literal content.</summary>
public sealed class WorkspaceScaffoldFile
{
    public string RelativePath { get; init; } = string.Empty;
    public string Content { get; init; } = string.Empty;
}

/// <summary>The concrete, pure plan a template produces for a given schema — directories
/// and files. Rendered by the New Project wizard as a preview <em>before</em> anything is
/// written to disk.</summary>
public sealed class WorkspaceScaffoldPlan
{
    public List<string> Directories { get; } = new();
    public List<WorkspaceScaffoldFile> Files { get; } = new();

    /// <summary>Human note shown in the wizard (e.g. the image_caption empty-rows rationale).</summary>
    public string? Note { get; set; }
}

/// <summary>Outcome of a scaffold operation. Partial failures are reported, never hidden.</summary>
public sealed class ScaffoldResult
{
    public bool Ok => Error is null;
    public string? Error { get; init; }
    public List<string> CreatedDirectories { get; } = new();
    public List<string> CreatedFiles { get; } = new();
}

/// <summary>Template-driven New Dataset Project scaffolding (v1.2.3 Workspace System,
/// slice 3). <see cref="BuildPlan"/> is pure (previewable + testable); <see cref="Scaffold"/>
/// does the guarded I/O: it refuses to write into a non-empty folder unless explicitly
/// allowed, resolves every path through <see cref="WorkspacePathSafety"/>, refuses to
/// overwrite existing files, and writes the manifest last. Nothing is validated, trained,
/// or exported as a side effect.</summary>
public sealed class ProjectTemplateService
{
    private readonly WorkspaceManifestService _manifests;

    public ProjectTemplateService(WorkspaceManifestService? manifests = null)
        => _manifests = manifests ?? new WorkspaceManifestService();

    public IReadOnlyList<WorkspaceTemplateDefinition> Templates { get; } = new[]
    {
        new WorkspaceTemplateDefinition { Id = "empty", Kind = WorkspaceTemplateKind.Empty, Name = "Empty Workspace", Description = "Just the .corpus manifest. Bring your own layout." },
        new WorkspaceTemplateDefinition { Id = "minimal", Kind = WorkspaceTemplateKind.Minimal, Name = "Minimal Dataset Project", Description = "examples.jsonl + README + assets/." },
        new WorkspaceTemplateDefinition { Id = "standard", Kind = WorkspaceTemplateKind.Standard, Name = "Standard Dataset Project", Description = "Adds imports, splits, reports, exports, training configs." },
        new WorkspaceTemplateDefinition { Id = "full", Kind = WorkspaceTemplateKind.Full, Name = "Full Dataset-to-Model Project", Description = "Everything: asset kinds, report kinds, runs, artifacts, versions." },
        new WorkspaceTemplateDefinition { Id = "schema", Kind = WorkspaceTemplateKind.SchemaSpecific, Name = "Schema-Specific Starter", Description = "Standard + schema asset folders + a valid starter row where safe." },
    };

    /// <summary>Build the (pure) scaffold plan. <paramref name="starterRowJson"/> is the
    /// schema's example row (the app already has this via DatasetSchema.ExampleText); it is
    /// seeded into examples.jsonl only for the Schema-Specific template and only when it is
    /// safe — never for image_caption, whose placeholder paths would fail validation.</summary>
    public WorkspaceScaffoldPlan BuildPlan(string templateId, string schemaId, string projectName, string projectId, string? starterRowJson = null)
    {
        var plan = new WorkspaceScaffoldPlan();
        plan.Directories.Add(WorkspaceProjectManifest.MetadataDirectoryName); // ".corpus"

        if (string.Equals(templateId, "empty", StringComparison.Ordinal))
            return plan;

        var isSchemaStarter = string.Equals(templateId, "schema", StringComparison.Ordinal);
        var isImage = string.Equals(schemaId, "image_caption", StringComparison.Ordinal);
        var seedRow = isSchemaStarter && !isImage && !string.IsNullOrWhiteSpace(starterRowJson);

        plan.Files.Add(new WorkspaceScaffoldFile { RelativePath = "examples.jsonl", Content = seedRow ? starterRowJson!.Trim() + "\n" : string.Empty });
        plan.Files.Add(new WorkspaceScaffoldFile { RelativePath = "README.md", Content = Readme(projectName, schemaId, isSchemaStarter && isImage) });
        plan.Directories.Add("assets");

        if (string.Equals(templateId, "minimal", StringComparison.Ordinal))
            return plan;

        // standard+ : layout-persistence stub, dataset card, working dirs.
        plan.Files.Add(new WorkspaceScaffoldFile { RelativePath = ".corpus/workspace.json", Content = "{\n  \"open_files\": [],\n  \"expanded_folders\": []\n}\n" });
        plan.Files.Add(new WorkspaceScaffoldFile { RelativePath = "dataset_card.json", Content = DatasetCard(projectName, schemaId) });
        plan.Directories.AddRange(new[] { "imports", "imports/quarantine", "splits", "reports", "exports", "training_configs" });

        if (string.Equals(templateId, "full", StringComparison.Ordinal))
        {
            plan.Directories.AddRange(new[]
            {
                "assets/images", "assets/audio", "assets/video", "assets/documents", "assets/code", "assets/misc",
                "reports/validation", "reports/quality", "reports/gates", "reports/leakage",
                "evaluation_reports", "arena_reports", "training_runs", "model_artifacts", "dataset_versions",
            });
        }
        else if (isSchemaStarter)
        {
            plan.Directories.AddRange(SchemaAssetDirs(schemaId));
            plan.Note = isImage
                ? "image_caption: examples.jsonl left empty (placeholder image paths would fail validation). Starter guidance written to README.md."
                : "A valid starter row was seeded into examples.jsonl.";
        }

        // De-dupe directories, preserving first-seen order.
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var deduped = plan.Directories.Where(d => seen.Add(WorkspaceLayout.Normalize(d))).ToList();
        plan.Directories.Clear();
        plan.Directories.AddRange(deduped);
        return plan;
    }

    /// <summary>Apply a plan to disk. Creates the root, refuses a non-empty root unless
    /// <paramref name="allowNonEmpty"/> (the confirmed "initialize existing folder" flow),
    /// resolves every entry within the root, refuses to overwrite files, and writes the
    /// manifest last. Returns a result carrying the created paths, or an <c>Error</c>.</summary>
    public ScaffoldResult Scaffold(string workspaceRoot, WorkspaceScaffoldPlan plan, WorkspaceProjectManifest manifest, bool allowNonEmpty = false)
    {
        if (string.IsNullOrWhiteSpace(workspaceRoot)) return new ScaffoldResult { Error = "Workspace root was empty." };
        if (plan is null) return new ScaffoldResult { Error = "No scaffold plan." };
        if (manifest is null) return new ScaffoldResult { Error = "No manifest." };

        string root;
        try { root = WorkspacePathSafety.NormalizeRoot(workspaceRoot); }
        catch (ArgumentException ex) { return new ScaffoldResult { Error = $"Invalid workspace path: {ex.Message}" }; }

        var result = new ScaffoldResult();
        try
        {
            Directory.CreateDirectory(root);
            if (!allowNonEmpty && Directory.EnumerateFileSystemEntries(root).Any())
                return new ScaffoldResult { Error = "Folder is not empty. Confirm before initializing an existing folder." };

            foreach (var rel in plan.Directories)
            {
                if (!WorkspacePathSafety.TryResolveWithinRoot(root, rel, out var full))
                    return new ScaffoldResult { Error = $"Refused unsafe directory path: {rel}" };
                Directory.CreateDirectory(full);
                result.CreatedDirectories.Add(full);
            }

            foreach (var file in plan.Files)
            {
                if (!WorkspacePathSafety.TryResolveWithinRoot(root, file.RelativePath, out var full))
                    return new ScaffoldResult { Error = $"Refused unsafe file path: {file.RelativePath}" };
                if (File.Exists(full) && !allowNonEmpty)
                    return new ScaffoldResult { Error = $"Refused to overwrite existing file: {file.RelativePath}" };
                Directory.CreateDirectory(Path.GetDirectoryName(full)!);
                File.WriteAllText(full, file.Content);
                result.CreatedFiles.Add(full);
            }

            var manifestError = _manifests.Write(root, manifest);
            if (manifestError is not null)
                return new ScaffoldResult { Error = manifestError };

            return result;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new ScaffoldResult { Error = $"Scaffold failed: {ex.Message}" };
        }
    }

    private static IEnumerable<string> SchemaAssetDirs(string schemaId) => schemaId switch
    {
        "image_caption" => new[] { "assets/images" },
        "code" => new[] { "assets/code" },
        "chat" or "retrieval" or "evaluation" => new[] { "assets/documents" },
        _ => new[] { "assets/misc" },
    };

    private static string Readme(string projectName, string schemaId, bool imageStarter)
    {
        var body = $"# {projectName}\n\n- **Schema:** {schemaId}\n\n";
        return imageStarter
            ? body + "## Getting started (image_caption)\n\nPlace images under `assets/images/` and add one row per image to `examples.jsonl`:\n\n```json\n{\"image\": \"assets/images/your_image.jpg\", \"caption\": \"...\"}\n```\n\n`examples.jsonl` was left empty on purpose — placeholder image paths would fail validation.\n"
            : body + "Author rows in Writing Studio or import a JSONL file, then validate.\n";
    }

    private static string DatasetCard(string projectName, string schemaId) =>
        $"{{\n  \"name\": \"{projectName}\",\n  \"schema\": \"{schemaId}\",\n  \"rows\": 0\n}}\n";
}
