using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>What opening a folder should do (v1.2.4 Workspace System, slice 3c).</summary>
public enum WorkspaceOpenAction
{
    /// <summary>Folder already has a <c>.corpus/project.json</c> — open it directly.</summary>
    OpenManifest,

    /// <summary>Recognizable dataset files but no manifest — offer to initialize (adds the
    /// manifest; existing rows untouched).</summary>
    OfferInitializeDataset,

    /// <summary>Empty folder — offer to create a workspace there.</summary>
    OfferCreateEmpty,

    /// <summary>Random non-empty folder — not a workspace and not a dataset; do not mutate.</summary>
    RejectOther,
}

/// <summary>Pure decision logic for the Open/Initialize Folder flow — kept free of WPF and
/// dialogs so it is unit-testable. The code-behind inspects the folder, asks this what to do,
/// then runs the confirmations and opening.</summary>
public static class WorkspaceOpenRouting
{
    public const string DefaultSchemaId = "instruction";

    /// <summary>Decide the action from three facts about a folder.</summary>
    public static WorkspaceOpenAction Classify(bool hasManifest, bool hasExamples, bool isEmpty)
    {
        if (hasManifest)
        {
            return WorkspaceOpenAction.OpenManifest;
        }

        if (hasExamples)
        {
            return WorkspaceOpenAction.OfferInitializeDataset;
        }

        return isEmpty ? WorkspaceOpenAction.OfferCreateEmpty : WorkspaceOpenAction.RejectOther;
    }

    /// <summary>Inspect a folder on disk and classify it. Tolerant of I/O errors (an
    /// unreadable folder is treated as "reject", never a crash).</summary>
    public static WorkspaceOpenAction Inspect(string folder, WorkspaceManifestService manifests)
    {
        if (string.IsNullOrWhiteSpace(folder))
        {
            return WorkspaceOpenAction.RejectOther;
        }

        var hasManifest = manifests.HasManifest(folder);

        bool hasExamples;
        try
        {
            hasExamples = File.Exists(Path.Combine(folder, "examples.jsonl"));
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or UnauthorizedAccessException)
        {
            hasExamples = false;
        }

        bool isEmpty;
        try
        {
            isEmpty = Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            isEmpty = false;
        }

        return Classify(hasManifest, hasExamples, isEmpty);
    }

    /// <summary>Decide whether an action that REPLACES the current workspace (open folder,
    /// recent, initialize, or new-project) may proceed. Proceeds immediately when there is no
    /// unsaved work; otherwise proceeds only if the user confirms discarding it. Kept free of
    /// dialogs so it is unit-testable — the code-behind supplies the confirmation prompt, which
    /// is invoked ONLY when there is unsaved work (a clean workspace opens with no prompt).</summary>
    public static bool ShouldReplaceWorkspace(bool hasUnsavedWork, Func<bool> confirmDiscard)
    {
        ArgumentNullException.ThrowIfNull(confirmDiscard);
        return !hasUnsavedWork || confirmDiscard();
    }

    /// <summary>Async form for the head-agnostic dialog seam (non-WPF confirms are async). Same
    /// contract: proceed immediately when nothing is unsaved (the prompt is never awaited); else
    /// proceed only if the awaited confirmation returns true.</summary>
    public static async Task<bool> ShouldReplaceWorkspaceAsync(
        bool hasUnsavedWork, Func<Task<bool>> confirmDiscard)
    {
        ArgumentNullException.ThrowIfNull(confirmDiscard);
        return !hasUnsavedWork || await confirmDiscard();
    }

    /// <summary>Derive the (projectId, name, schemaId) used to open a folder as the active
    /// workspace: from a recognized manifest, else from the folder name + default schema.</summary>
    public static (string ProjectId, string Name, string SchemaId) DeriveOpenArgs(
        WorkspaceProjectManifest? manifest, string folderName)
    {
        var safeName = string.IsNullOrWhiteSpace(folderName) ? "workspace" : folderName;

        if (manifest is { IsRecognized: true })
        {
            var id = string.IsNullOrWhiteSpace(manifest.ProjectId) ? safeName : manifest.ProjectId;
            var name = string.IsNullOrWhiteSpace(manifest.Name) ? safeName : manifest.Name;
            var schema = string.IsNullOrWhiteSpace(manifest.SchemaId) ? DefaultSchemaId : manifest.SchemaId;
            return (id, name, schema);
        }

        return (safeName, safeName, DefaultSchemaId);
    }
}
