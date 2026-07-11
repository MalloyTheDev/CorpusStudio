using System;
using System.Globalization;
using Avalonia.Controls;
using Avalonia.Data.Converters;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Maps a fraction (0..1) to a star <see cref="GridLength"/> so two grid columns can
/// render a proportional two-layer bar (an Accent fill over a Track track) without pixel math:
/// the default returns <c>fraction*</c> (the filled portion); with <c>ConverterParameter=rest</c>
/// it returns <c>(1-fraction)*</c> (the remaining track). Used by the Model Arena win-rate bars.
/// Out-of-range or non-numeric values clamp to [0,1]; never throws in a binding.</summary>
public sealed class FractionToStarConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var fraction = value switch
        {
            double d => d,
            float f => f,
            int i => i,
            _ => 0d,
        };

        if (double.IsNaN(fraction) || fraction < 0d)
        {
            fraction = 0d;
        }
        else if (fraction > 1d)
        {
            fraction = 1d;
        }

        var isRest = parameter is string s
            && string.Equals(s, "rest", StringComparison.OrdinalIgnoreCase);
        return new GridLength(isRest ? 1d - fraction : fraction, GridUnitType.Star);
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
