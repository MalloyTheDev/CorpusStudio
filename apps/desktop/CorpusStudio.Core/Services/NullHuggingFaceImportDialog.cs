using System.Collections.Generic;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>No-op <see cref="IHuggingFaceImportDialog"/> that stages nothing (returns null) — the
/// default for the parameterless design-time constructor so existing <c>new MainWindowViewModel()</c>
/// call sites keep working, and the fallback for any head without a real HF import dialog yet.</summary>
public sealed class NullHuggingFaceImportDialog : IHuggingFaceImportDialog
{
    public Task<string?> ShowAsync(string schemaId, string schemaName, IReadOnlyList<DatasetField> schemaFields) =>
        Task.FromResult<string?>(null);
}
