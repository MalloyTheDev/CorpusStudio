using System;
using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Resolves a Dashboard "Recent Activity" row's leading glyph from an
/// <c>EngineLogEntry.ActivityGlyphKey</c> (a Nocturne icon resource name the model computes purely from
/// the real engine verb + outcome). The facet <see cref="IValueConverter"/> parameter selects:
/// <list type="bullet">
///   <item><c>"glyph"</c> — the icon <see cref="Geometry"/> for the key (an unknown/empty key falls back
///     to the neutral <c>IcoListDashes</c>, never a blank).</item>
///   <item><c>"fill"</c> — a themed Nocturne tint keyed off the same glyph so the operation type reads at
///     a glance (quality → Warn, pass → Ok, version → Accent, error → Bad, import → T2, neutral → T4).</item>
/// </list>
/// Resolves against <see cref="Application"/> theme resources exactly like <c>LifecycleNodeConverter</c>;
/// like that precedent it does not re-run on a live theme switch (acceptable, matching the existing
/// converters). Never throws inside a binding.</summary>
public sealed class ActivityGlyphConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var key = value as string;
        if (string.IsNullOrWhiteSpace(key))
        {
            key = "IcoListDashes";
        }

        var facet = (parameter as string ?? "glyph").ToLowerInvariant();
        return facet == "fill"
            ? Brush(TintToken(key))
            : Resource(key) ?? Resource("IcoListDashes");
    }

    /// <summary>Nocturne brush token for a glyph key. Every token exists in both theme variants.</summary>
    private static string TintToken(string key) => key switch
    {
        "IcoBroom" => "Warn",
        "IcoCheckCircleFill" => "Ok",
        "IcoGitCommit" => "Accent",
        "IcoWarningFill" => "Bad",
        "IcoImport" => "T2",
        _ => "T4",
    };

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

    private static object? Resource(string key)
    {
        var app = Application.Current;
        return app is not null && app.TryGetResource(key, app.ActualThemeVariant, out var res)
            ? res
            : null;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
