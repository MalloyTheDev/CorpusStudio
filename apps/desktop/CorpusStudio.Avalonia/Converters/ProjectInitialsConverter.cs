using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace CorpusStudio.Avalonia.Converters;

/// <summary>Derives a compact 1–2 character avatar monogram from the active project's title for the
/// Nocturne project-switcher card: the first letter of each of the first two words (e.g.
/// "Support Chatbot" → "SC"), or the first two letters of a single-word title (e.g.
/// "instruction" → "IN"); a blank title falls back to "?". Purely presentational and read-only over
/// <see cref="CorpusStudio.Desktop.ViewModels.MainWindowViewModel.ActiveProjectTitle"/>, so it needs no
/// new view-model state. Declared locally in the switcher's resources rather than app-wide.</summary>
public sealed class ProjectInitialsConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var title = (value as string)?.Trim();
        if (string.IsNullOrEmpty(title))
        {
            return "?";
        }

        var words = title.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length >= 2)
        {
            return string.Concat(
                char.ToUpperInvariant(words[0][0]),
                char.ToUpperInvariant(words[1][0]));
        }

        var word = words[0];
        return (word.Length >= 2 ? word[..2] : word).ToUpperInvariant();
    }

    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
