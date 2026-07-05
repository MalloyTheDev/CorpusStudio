using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>Outcome of opening a document. A missing/unsafe/unreadable path is a
/// non-crashing error result, never an exception.</summary>
public sealed class OpenDocumentResult
{
    public bool Ok => Document is not null && Error is null;
    public string? Error { get; init; }
    public OpenWorkspaceDocument? Document { get; init; }
}

/// <summary>Opens and saves workspace documents safely (v1.2.3 Workspace System, slice 5).
/// Generated artifacts and over-large files open read-only; images/binaries load metadata
/// only; text-editable kinds load their content. Saving is always explicit and atomic, is
/// refused for read-only docs, and writes verbatim — examples.jsonl is never auto-formatted
/// (the desktop is its single writer).</summary>
public sealed class WorkspaceDocumentService
{
    /// <summary>Files larger than this open as a read-only preview (a virtualized editor
    /// is a later slice).</summary>
    public long MaxEditableBytes { get; init; } = 2 * 1024 * 1024;

    /// <summary>How many bytes of an over-large text file to show in the read-only preview.</summary>
    public int PreviewBytes { get; init; } = 64 * 1024;

    /// <summary>Open a document off the UI thread. The read (up to <see cref="MaxEditableBytes"/>,
    /// or a bounded preview for larger files), classification, and metadata all run on the thread
    /// pool so a large or slow file never stalls the UI; the caller marshals the result back.</summary>
    public Task<OpenDocumentResult> OpenAsync(string workspaceRoot, string relativePath)
        => Task.Run(() => Open(workspaceRoot, relativePath));

    public OpenDocumentResult Open(string workspaceRoot, string relativePath)
    {
        if (!WorkspacePathSafety.TryResolveWithinRoot(workspaceRoot, relativePath, out var full))
            return new OpenDocumentResult { Error = "Path is empty, absolute, or escapes the workspace root." };
        if (!File.Exists(full))
            return new OpenDocumentResult { Error = "File not found." };

        var rel = WorkspaceLayout.Normalize(relativePath);
        var kind = WorkspaceFileKinds.Classify(full, isDirectory: false);
        var meta = GetMetadata(workspaceRoot, relativePath);
        var generated = WorkspaceLayout.IsGeneratedArtifact(rel);

        // Non-text: image / audio / video / binary / unknown -> metadata-only viewer.
        if (kind is WorkspaceFileKind.Image or WorkspaceFileKind.AudioFuture or WorkspaceFileKind.VideoFuture
                 or WorkspaceFileKind.Binary or WorkspaceFileKind.Unknown)
        {
            var doc = new OpenWorkspaceDocument
            {
                DisplayName = Path.GetFileName(full),
                FullPath = full,
                RelativePath = rel,
                FileKind = kind,
                IsReadOnly = true,
                Metadata = meta,
                ImagePreviewPath = kind == WorkspaceFileKind.Image ? full : null,
                StatusMessage = kind == WorkspaceFileKind.Image
                    ? "Image preview — opening never creates a dataset row."
                    : "No text preview for this file type.",
            };
            return new OpenDocumentResult { Document = doc };
        }

        long size;
        try { size = new FileInfo(full).Length; }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return new OpenDocumentResult { Error = ex.Message }; }

        var tooLarge = size > MaxEditableBytes;
        string text;
        var status = string.Empty;
        var readOnly = generated || tooLarge;
        try
        {
            if (tooLarge)
            {
                status = $"File is {meta.HumanSize} — opened as a read-only preview (first {PreviewBytes / 1024} KB).";
                using var stream = new FileStream(full, FileMode.Open, FileAccess.Read);
                var buffer = new byte[Math.Min(PreviewBytes, (int)Math.Min(size, int.MaxValue))];
                var read = stream.Read(buffer, 0, buffer.Length);
                text = Encoding.UTF8.GetString(buffer, 0, read);
            }
            else
            {
                text = File.ReadAllText(full);
                if (generated) status = "Generated artifact — opened read-only.";
                else if (WorkspaceLayout.IsDatasetCoreFile(rel)) status = "Dataset core file — the desktop is its single writer. Edits here are explicit and never auto-formatted.";
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return new OpenDocumentResult { Error = ex.Message }; }

        var textDoc = new OpenWorkspaceDocument
        {
            DisplayName = Path.GetFileName(full),
            FullPath = full,
            RelativePath = rel,
            FileKind = kind,
            IsReadOnly = readOnly,
            Metadata = meta,
            StatusMessage = status,
        };
        textDoc.MarkClean(text);
        return new OpenDocumentResult { Document = textDoc };
    }

    /// <summary>Explicitly save an editable text document. Refuses read-only docs; writes
    /// atomically (temp + move); leaves content verbatim. Returns an error string or null
    /// on success.</summary>
    public string? Save(OpenWorkspaceDocument document)
    {
        if (document is null) return "No document.";
        if (document.IsReadOnly) return "This document is read-only and was not modified.";
        if (string.IsNullOrEmpty(document.FullPath)) return "Document has no path.";

        var temp = document.FullPath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            File.WriteAllText(temp, document.TextContent);
            File.Move(temp, document.FullPath, overwrite: true);
            document.MarkClean(document.TextContent);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            try { if (File.Exists(temp)) File.Delete(temp); }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException) { /* best-effort */ }
            return $"Could not save: {ex.Message}";
        }
    }

    /// <summary>Basic file facts for the selected-file panel. Dependency-free: image
    /// dimensions are left null (the WPF layer can fill them from a decoded BitmapFrame).</summary>
    public WorkspaceFileMetadata GetMetadata(string workspaceRoot, string relativePath)
    {
        var rel = WorkspaceLayout.Normalize(relativePath);
        WorkspacePathSafety.TryResolveWithinRoot(workspaceRoot, relativePath, out var full);
        var probe = string.IsNullOrEmpty(full) ? rel : full;
        var kind = WorkspaceFileKinds.Classify(probe, isDirectory: false);

        long size = 0;
        DateTimeOffset? modified = null;
        var exists = false;
        try
        {
            var fi = new FileInfo(full);
            if (fi.Exists) { exists = true; size = fi.Length; modified = fi.LastWriteTimeUtc; }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException) { /* leave defaults */ }

        return new WorkspaceFileMetadata
        {
            RelativePath = rel,
            FileKind = kind,
            Exists = exists,
            SizeBytes = size,
            LastModifiedUtc = modified,
            IsGeneratedArtifact = WorkspaceLayout.IsGeneratedArtifact(rel),
            IsDatasetCoreFile = WorkspaceLayout.IsDatasetCoreFile(rel),
        };
    }
}
