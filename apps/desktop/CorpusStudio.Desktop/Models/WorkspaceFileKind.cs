using System;
using System.Collections.Generic;
using System.IO;

namespace CorpusStudio.Desktop.Models;

/// <summary>How a workspace entry should be viewed/edited. The file <em>kind</em>
/// controls the viewer; the dataset <em>schema</em> controls row behavior (they are
/// separate axes — see the Workspace System design).</summary>
public enum WorkspaceFileKind
{
    Folder,
    Jsonl,
    Json,
    Markdown,
    Text,
    Yaml,
    Toml,
    Code,
    Image,
    AudioFuture,
    VideoFuture,
    Binary,
    Unknown,
}

/// <summary>Pure, deterministic extension-to-kind classification. No I/O, no schema
/// knowledge — just the file kind that selects a viewer. Central so the explorer and
/// the document services agree on classification.</summary>
public static class WorkspaceFileKinds
{
    private static readonly Dictionary<string, WorkspaceFileKind> ByExtension =
        new(StringComparer.OrdinalIgnoreCase)
        {
            [".jsonl"] = WorkspaceFileKind.Jsonl,
            [".json"] = WorkspaceFileKind.Json,
            [".md"] = WorkspaceFileKind.Markdown,
            [".markdown"] = WorkspaceFileKind.Markdown,
            [".txt"] = WorkspaceFileKind.Text,
            [".yaml"] = WorkspaceFileKind.Yaml,
            [".yml"] = WorkspaceFileKind.Yaml,
            [".toml"] = WorkspaceFileKind.Toml,
            // Code
            [".py"] = WorkspaceFileKind.Code,
            [".cs"] = WorkspaceFileKind.Code,
            [".cpp"] = WorkspaceFileKind.Code,
            [".c"] = WorkspaceFileKind.Code,
            [".h"] = WorkspaceFileKind.Code,
            [".hpp"] = WorkspaceFileKind.Code,
            [".js"] = WorkspaceFileKind.Code,
            [".ts"] = WorkspaceFileKind.Code,
            [".rs"] = WorkspaceFileKind.Code,
            [".java"] = WorkspaceFileKind.Code,
            [".go"] = WorkspaceFileKind.Code,
            // Images
            [".png"] = WorkspaceFileKind.Image,
            [".jpg"] = WorkspaceFileKind.Image,
            [".jpeg"] = WorkspaceFileKind.Image,
            [".webp"] = WorkspaceFileKind.Image,
            [".gif"] = WorkspaceFileKind.Image,
            // Future media (classified now, previews are a later slice)
            [".wav"] = WorkspaceFileKind.AudioFuture,
            [".mp3"] = WorkspaceFileKind.AudioFuture,
            [".flac"] = WorkspaceFileKind.AudioFuture,
            [".mp4"] = WorkspaceFileKind.VideoFuture,
            [".webm"] = WorkspaceFileKind.VideoFuture,
            [".mov"] = WorkspaceFileKind.VideoFuture,
        };

    private static readonly HashSet<string> BinaryExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".zip", ".gz", ".tar", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib", ".bin",
            ".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".onnx",
            ".sqlite3", ".db", ".parquet", ".arrow",
            ".pdf", ".ico",
        };

    /// <summary>Classify a path. Directories are <see cref="WorkspaceFileKind.Folder"/>;
    /// known extensions map to their kind; known binary extensions are
    /// <see cref="WorkspaceFileKind.Binary"/>; everything else is
    /// <see cref="WorkspaceFileKind.Unknown"/>.</summary>
    public static WorkspaceFileKind Classify(string path, bool isDirectory)
    {
        if (isDirectory)
        {
            return WorkspaceFileKind.Folder;
        }

        var extension = Path.GetExtension(path ?? string.Empty);
        if (string.IsNullOrEmpty(extension))
        {
            return WorkspaceFileKind.Unknown;
        }

        if (ByExtension.TryGetValue(extension, out var kind))
        {
            return kind;
        }

        return BinaryExtensions.Contains(extension)
            ? WorkspaceFileKind.Binary
            : WorkspaceFileKind.Unknown;
    }

    /// <summary>True for kinds that are safe to open in a plain text editor.</summary>
    public static bool IsTextEditable(WorkspaceFileKind kind) => kind switch
    {
        WorkspaceFileKind.Jsonl => true,
        WorkspaceFileKind.Json => true,
        WorkspaceFileKind.Markdown => true,
        WorkspaceFileKind.Text => true,
        WorkspaceFileKind.Yaml => true,
        WorkspaceFileKind.Toml => true,
        WorkspaceFileKind.Code => true,
        _ => false,
    };
}
