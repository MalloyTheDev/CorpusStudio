using System;
using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Maps a New Project preview line's indent (px, from Depth×16) to a left
/// <see cref="Thickness"/> so nested scaffold folders/files render stepped in.</summary>
public sealed class PreviewIndentConverter : IValueConverter
{
    public static readonly PreviewIndentConverter Instance = new();

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var left = value switch
        {
            double d => d,
            int i => i,
            _ => 0.0,
        };
        return new Thickness(left, 0, 0, 0);
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
