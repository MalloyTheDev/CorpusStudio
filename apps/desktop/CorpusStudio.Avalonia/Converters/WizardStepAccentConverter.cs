using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Stepper coloring for the New Project wizard: returns the accent brush when the current
/// <c>WizardStep</c> has reached (or passed) the step number in the parameter, else a muted brush.
/// Used for both the step dots and their labels so progress reads at a glance. Concrete brushes
/// (accent ≈ same in both themes; muted is a mid-grey that reads on either) — mirrors the pragmatic
/// approach of <see cref="StudioTabActiveBrushConverter"/>.</summary>
public sealed class WizardStepAccentConverter : IValueConverter
{
    private static readonly IBrush Accent = new SolidColorBrush(Color.Parse("#968AE0"));
    private static readonly IBrush Muted = new SolidColorBrush(Color.Parse("#5B6070"));

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is int step && parameter is string s && int.TryParse(s, out var n) && step >= n)
        {
            return Accent;
        }

        return Muted;
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
