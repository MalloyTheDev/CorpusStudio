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
        // Only support converting to Visibility (or nullable Visibility).
        if (targetType != typeof(Visibility) && targetType != typeof(Visibility?))
        {
            return DependencyProperty.UnsetValue;
        }

        // Only support bool inputs; surface misconfigurations instead of treating them as false.
        if (value is not bool flag)
        {
            return DependencyProperty.UnsetValue;
        }

        return flag ? Visibility.Collapsed : Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        // Only support converting back to bool (or nullable bool).
        if (targetType != typeof(bool) && targetType != typeof(bool?))
        {
            return DependencyProperty.UnsetValue;
        }

        if (value is not Visibility visibility)
        {
            return DependencyProperty.UnsetValue;
        }

        // Inverse converter: Visible -> false, Collapsed/Hidden -> true.
        return visibility != Visibility.Visible;
    }
}
