using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Views;

namespace CorpusStudio.Desktop.Services;

/// <summary>WPF adapter for <see cref="IHuggingFaceImportDialog"/>: shows the modal
/// <see cref="HfImportWindow"/> and returns the staged JSONL path (or null on cancel). The
/// owner window is resolved lazily at show time (the main window exists by then), so the VM can
/// depend on the seam at construction without a window reference.</summary>
public sealed class HuggingFaceImportDialog : IHuggingFaceImportDialog
{
    private readonly PythonEngineService _engine;

    public HuggingFaceImportDialog(PythonEngineService engine)
    {
        _engine = engine;
    }

    public Task<string?> ShowAsync(string schemaId, string schemaName, IReadOnlyList<DatasetField> schemaFields)
    {
        var dialog = new HfImportWindow(_engine, schemaId, schemaName, schemaFields)
        {
            Owner = Application.Current?.MainWindow,
        };
        var confirmed = dialog.ShowDialog() == true;
        return Task.FromResult(confirmed ? dialog.Result?.StagingPath : null);
    }
}
