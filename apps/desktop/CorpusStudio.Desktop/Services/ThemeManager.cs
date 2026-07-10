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

    // The theme dictionary we last installed, tracked by reference so a swap is exact after the first
    // one — independent of how WPF reports ResourceDictionary.Source.
    private static ResourceDictionary? _themeDict;

    public static AppTheme Current => _current;

    /// <summary>Apply a theme by replacing the theme dictionary in the app's merged dictionaries.
    /// No-op without an Application (design-time / tests).</summary>
    public static void Apply(AppTheme theme)
    {
        _current = theme;
        if (Application.Current is not { } app)
        {
            return;
        }

        var dict = new ResourceDictionary { Source = new Uri($"Themes/{theme}.xaml", UriKind.Relative) };
        var merged = app.Resources.MergedDictionaries;

        // Replace the existing theme dictionary in place (preserving any other merged dictionaries).
        // Prefer the one we installed last; on the first swap, find the one App.xaml merged — matching
        // with Contains, since WPF may report Source as the RESOLVED pack URI
        // (pack://application:,,,/Themes/Light.xaml) rather than the relative "Themes/Light.xaml", and
        // both contain "Themes/". A StartsWith check would miss the pack form and wrongly stack a second
        // dictionary (leaving the old one to win), so the toggle would appear to do nothing.
        var existing = _themeDict is not null && merged.Contains(_themeDict)
            ? _themeDict
            : merged.FirstOrDefault(d =>
                d.Source is { } s && s.OriginalString.Contains("Themes/", StringComparison.OrdinalIgnoreCase));
        if (existing is not null)
        {
            merged[merged.IndexOf(existing)] = dict;
        }
        else
        {
            merged.Insert(0, dict);
        }

        _themeDict = dict;
    }

    /// <summary>Flip Light↔Dark.</summary>
    public static AppTheme Toggle()
    {
        Apply(_current == AppTheme.Light ? AppTheme.Dark : AppTheme.Light);
        return _current;
    }
}
