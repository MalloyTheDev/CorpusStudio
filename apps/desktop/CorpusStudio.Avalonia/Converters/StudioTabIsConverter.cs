using System;
using System.Globalization;
using Avalonia.Data.Converters;
using CorpusStudio.Desktop.ViewModels;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>True when the bound <see cref="MainWindowViewModel.SelectedStudioTabIndex"/> equals the
/// <see cref="StudioTab"/> named by the converter parameter. Drives the grouped-IA content-switcher:
/// each Studio screen panel is <c>IsVisible</c>-bound with its tab name, so exactly one shows. Using
/// the enum name (not a raw index) keeps the XAML readable and immune to enum re-ordering.</summary>
public sealed class StudioTabIsConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value is int index
            && parameter is string name
            && Enum.TryParse<StudioTab>(name, out var tab)
            && index == (int)tab;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
