using System;
using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Maps a Dashboard lifecycle-node status token ("done" | "warn" | "bad" | "neutral" |
/// "locked", from <c>MainWindowViewModel.*NodeStatus</c>) to the presentation facet named by the
/// converter parameter, resolving the Nocturne design tokens from the active theme:
/// <list type="bullet">
///   <item><c>"bg"</c> — circle fill brush (Ok / Warn / Bad for done / warn / bad; Track for neutral + locked).</item>
///   <item><c>"border"</c> — circle border brush (the status colour, else Line2 for neutral + locked).</item>
///   <item><c>"glyphfill"</c> — status-glyph fill (Panel on a coloured circle so the glyph inverts for
///     contrast in both themes; T4 for the locked node).</item>
///   <item><c>"glyph"</c> — the Phosphor <see cref="Geometry"/> (check / warning / x-circle / lock),
///     <c>null</c> when neutral so a signal-less node shows a bare circle.</item>
///   <item><c>"hasglyph"</c> — <c>false</c> only for neutral, driving the glyph's <c>IsVisible</c>.</item>
/// </list>
/// Neutral is the honesty default: a node with no real view-model signal renders a bare Track circle —
/// never a colour or a completion glyph. Resolved against <see cref="Application"/> theme resources like
/// <c>StringToBrushConverter</c>; it does not re-run on a live theme switch (acceptable, matching the
/// existing converter precedent).</summary>
public sealed class LifecycleNodeConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var status = (value as string ?? "neutral").ToLowerInvariant();
        var facet = (parameter as string ?? "bg").ToLowerInvariant();

        return facet switch
        {
            "hasglyph" => status is not ("neutral" or ""),
            "glyph" => Geometry(status),
            "glyphfill" => Brush(status switch
            {
                "done" or "warn" or "bad" => "Panel",
                "locked" => "T4",
                _ => "Track",
            }),
            "border" => Brush(status switch
            {
                "done" => "Ok",
                "warn" => "Warn",
                "bad" => "Bad",
                _ => "Line2", // neutral + locked
            }),
            _ => Brush(status switch // "bg"
            {
                "done" => "Ok",
                "warn" => "Warn",
                "bad" => "Bad",
                _ => "Track", // neutral + locked
            }),
        };
    }

    /// <summary>Resolve a themed Nocturne brush by resource key; transparent when unavailable (never throws).</summary>
    private static IBrush Brush(string key)
    {
        var app = Application.Current;
        if (app is not null
            && app.TryGetResource(key, app.ActualThemeVariant, out var res)
            && res is IBrush brush)
        {
            return brush;
        }

        return Brushes.Transparent;
    }

    /// <summary>Resolve the status glyph StreamGeometry; null for neutral (bare circle) or if missing.</summary>
    private static object? Geometry(string status)
    {
        var key = status switch
        {
            "done" => "IcoCheckCircleFill",
            "warn" => "IcoWarningFill",
            "bad" => "IcoXCircleFill",
            "locked" => "IcoLock",
            _ => null,
        };
        if (key is null)
        {
            return null;
        }

        var app = Application.Current;
        return app is not null && app.TryGetResource(key, app.ActualThemeVariant, out var res)
            ? res
            : null;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
