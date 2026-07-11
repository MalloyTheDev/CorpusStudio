using System;
using System.Globalization;
using System.Text;
using Avalonia.Data.Converters;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Produces a right-aligned line-number gutter ("1\n2\n3…") for the Explorer's code
/// viewer, computed from the document's real text. Purely presentational — the numbers reflect
/// the actual content, nothing is fabricated. Empty/no-text yields "1" so the gutter never
/// renders blank next to an empty file.</summary>
public sealed class TextToLineNumbersConverter : IValueConverter
{
    public static readonly TextToLineNumbersConverter Instance = new();

    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var text = value as string ?? string.Empty;
        if (text.Length == 0)
        {
            return "1";
        }

        // Count newlines; a trailing newline does not add a visible final line.
        var lines = 1;
        foreach (var ch in text)
        {
            if (ch == '\n')
            {
                lines++;
            }
        }

        if (text.EndsWith("\n", StringComparison.Ordinal))
        {
            lines--;
        }

        if (lines < 1)
        {
            lines = 1;
        }

        var sb = new StringBuilder(lines * 3);
        for (var i = 1; i <= lines; i++)
        {
            if (i > 1)
            {
                sb.Append('\n');
            }

            sb.Append(i.ToString(CultureInfo.InvariantCulture));
        }

        return sb.ToString();
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
