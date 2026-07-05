using System;
using System.Globalization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Basic, read-only file facts for the Explorer's selected-file panel (v1.2.3
/// Workspace System, slice 5). Pure data — produced by
/// <see cref="Services.WorkspaceDocumentService.GetMetadata"/>.</summary>
public sealed class WorkspaceFileMetadata
{
    public string RelativePath { get; init; } = string.Empty;
    public WorkspaceFileKind FileKind { get; init; }
    public bool Exists { get; init; }
    public long SizeBytes { get; init; }
    public DateTimeOffset? LastModifiedUtc { get; init; }

    /// <summary>Image pixel dimensions when known. Left null by the (dependency-free)
    /// service; the WPF layer fills these from a decoded <c>BitmapFrame</c> if needed.</summary>
    public int? ImageWidth { get; init; }
    public int? ImageHeight { get; init; }

    public bool IsGeneratedArtifact { get; init; }
    public bool IsDatasetCoreFile { get; init; }

    /// <summary>Human-readable size (e.g. "214 KB", "1.4 MB"). Invariant culture: a size
    /// always renders "1.4 MB", never "1,4 MB" on a comma-decimal locale (matches the
    /// number-formatting convention used elsewhere in the desktop).</summary>
    public string HumanSize
    {
        get
        {
            var ci = CultureInfo.InvariantCulture;
            return SizeBytes switch
            {
                < 1024 => $"{SizeBytes} B",
                < 1024 * 1024 => $"{(SizeBytes / 1024.0).ToString("0.#", ci)} KB",
                < 1024L * 1024 * 1024 => $"{(SizeBytes / (1024.0 * 1024)).ToString("0.#", ci)} MB",
                _ => $"{(SizeBytes / (1024.0 * 1024 * 1024)).ToString("0.#", ci)} GB",
            };
        }
    }
}
