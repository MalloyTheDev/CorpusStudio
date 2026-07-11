using System;
using System.Collections;
using System.Globalization;
using System.Linq;
using Avalonia.Data.Converters;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Projects the shell's engine command log (<c>OutputLog</c> — an append-only, newest-LAST
/// collection of real CLI invocations) into the Dashboard "Recent Activity" feed: the most recent
/// <see cref="MaxRows"/> entries, newest FIRST. A pure view-layer adapter — the shell view-model is not
/// touched and no entries are synthesised; it simply reverses and caps the real log. The Dashboard is a
/// summary surface, so it reflects the log as of the last time the binding evaluated (navigation/refresh).
/// The row count can be overridden via the converter parameter.</summary>
public sealed class RecentActivityConverter : IValueConverter
{
    public const int MaxRows = 6;

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not IEnumerable items)
        {
            return Array.Empty<object>();
        }

        var take = MaxRows;
        if (parameter is string p && int.TryParse(p, out var n) && n > 0)
        {
            take = n;
        }

        return items.Cast<object>().Reverse().Take(take).ToList();
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
