using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace CorpusStudio.Desktop;

/// <summary>
/// Returns Collapsed when the bound bool is true and Visible when false.
/// Used to show empty-state placeholders only when a list has no items
/// (bind to the list's HasItems).
/// </summary>
public sealed class InverseBooleanToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var flag = value is bool b && b;
        return flag ? Visibility.Collapsed : Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return value is Visibility visibility && visibility != Visibility.Visible;
    }
}
