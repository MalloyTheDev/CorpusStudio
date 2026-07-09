using System.Collections.Generic;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>Head-agnostic seam for the "Import from Hugging Face" flow — the inspect →
/// pick config/split → map columns → stage-to-JSONL dialog. The desktop head shows a modal
/// window; the seam lets the VM own the import command (guard, schema lookup, and handing the
/// staged file to the shared import-preview/quarantine path) without depending on that window,
/// so it is testable with a fake (mirrors <see cref="IDialogService"/> / <see cref="IFilePickerService"/>).</summary>
public interface IHuggingFaceImportDialog
{
    /// <summary>Show the HF import dialog for <paramref name="schemaId"/> and return the path of
    /// the staging JSONL to import, or <c>null</c> if the user cancelled or nothing was staged.
    /// The engine and this dialog never write the project's <c>examples.jsonl</c> — the caller
    /// runs the returned file through the normal import-preview/quarantine flow.</summary>
    Task<string?> ShowAsync(string schemaId, string schemaName, IReadOnlyList<DatasetField> schemaFields);
}
