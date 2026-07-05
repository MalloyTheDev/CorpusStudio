using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The pure WPF filter rendering in the Phase-0 file-picker seam. The dialogs themselves
/// can't run headless, but the portable-filter → WPF-string mapping must exactly reproduce the
/// filter of the import dialog it replaced.</summary>
public sealed class Win32FilePickerServiceTests
{
    [Fact]
    public void ToWpfFilter_ReproducesTheJsonlImportFilter()
    {
        var result = Win32FilePickerService.ToWpfFilter(new[]
        {
            new FilePickerFilter("JSONL files", "jsonl"),
            new FilePickerFilter("All files", "*"),
        });

        // Byte-identical to the old inline dialog's Filter string.
        Assert.Equal("JSONL files (*.jsonl)|*.jsonl|All files (*.*)|*.*", result);
    }

    [Fact]
    public void ToWpfFilter_JoinsMultipleExtensions()
    {
        var result = Win32FilePickerService.ToWpfFilter(new[] { new FilePickerFilter("Data", "jsonl", "csv") });
        Assert.Equal("Data (*.jsonl;*.csv)|*.jsonl;*.csv", result);
    }

    [Fact]
    public void ToWpfFilter_EmptyFallsBackToAllFiles()
    {
        Assert.Equal("All files (*.*)|*.*", Win32FilePickerService.ToWpfFilter(new FilePickerFilter[0]));
    }
}
