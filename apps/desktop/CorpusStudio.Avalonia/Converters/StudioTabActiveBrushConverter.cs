using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Returns the Nocturne accent-soft tint when the bound
/// <see cref="MainWindowViewModel.SelectedStudioTabIndex"/> matches the <see cref="StudioTab"/> named by
/// the parameter, else transparent — the active-row highlight for the grouped-IA nav. (A concrete tint,
/// not the theme resource, so a plain <c>Background</c> binding suffices; a later fidelity slice can make
/// it fully theme-swapping.)</summary>
public sealed class StudioTabActiveBrushConverter : IValueConverter
{
    // --accent-soft ≈ accent (#968AE0) at 16% — the design's active-nav fill.
    private static readonly IBrush Active = new SolidColorBrush(Color.Parse("#29968AE0"));

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is int index
            && parameter is string name
            && Enum.TryParse<StudioTab>(name, out var tab)
            && index == (int)tab)
        {
            return Active;
        }

        return Brushes.Transparent;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
