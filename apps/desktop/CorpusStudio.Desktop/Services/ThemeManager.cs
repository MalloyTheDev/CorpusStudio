using System;
using System.Linq;
using System.Windows;

namespace CorpusStudio.Desktop.Services;

/// <summary>App light/dark theme (#187/#201). The theme is a merged ResourceDictionary at index 0 of
/// the application resources (Light by default, whose values equal the pre-theming hardcoded colors —
/// so light mode is unchanged). Swapping it re-resolves every <c>{DynamicResource ...}</c> live.
/// This is an Application/View concern, kept out of the shared view-models.</summary>
public enum AppTheme
{
    Light,
    Dark,
}

public static class ThemeManager
{
    private static AppTheme _current = AppTheme.Light;

    public static AppTheme Current => _current;

    /// <summary>Apply a theme by swapping the theme dictionary at index 0. No-op without an
    /// Application (design-time / tests).</summary>
    public static void Apply(AppTheme theme)
    {
        _current = theme;
        if (Application.Current is not { } app)
        {
            return;
        }

        var uri = new Uri($"Themes/{theme}.xaml", UriKind.Relative);
        var dict = new ResourceDictionary { Source = uri };
        var merged = app.Resources.MergedDictionaries;

        // Replace the existing theme dictionary (the one that defines our brush tokens) in place,
        // preserving any other merged dictionaries.
        var existing = merged.FirstOrDefault(d =>
            d.Source is { } s && s.OriginalString.StartsWith("Themes/", StringComparison.OrdinalIgnoreCase));
        if (existing is not null)
        {
            merged[merged.IndexOf(existing)] = dict;
        }
        else
        {
            merged.Insert(0, dict);
        }
    }

    /// <summary>Flip Light↔Dark.</summary>
    public static AppTheme Toggle()
    {
        Apply(_current == AppTheme.Light ? AppTheme.Dark : AppTheme.Light);
        return _current;
    }
}
