using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Binds a hex color STRING (e.g. "#B45309") to a brush property. The shared view-models expose
/// status colors as strings (DebtGradeColor, SuiteOverallColor, QualityStatusColor, GateReport.StatusColor,
/// …) because WPF implicitly converts string→Brush; Avalonia bindings do not, so the ported .axaml views
/// route those bindings through this converter. Unparseable/empty → transparent (never throws in a binding).</summary>
public sealed class StringToBrushConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        => value is string s && Color.TryParse(s, out var color)
            ? new SolidColorBrush(color)
            : Brushes.Transparent;

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
