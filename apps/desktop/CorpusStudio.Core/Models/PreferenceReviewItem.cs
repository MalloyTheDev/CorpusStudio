namespace CorpusStudio.Desktop.Models;

public sealed class PreferenceReviewItem
{
    public int RowNumber { get; init; }

    public string Prompt { get; init; } = string.Empty;

    public string Chosen { get; init; } = string.Empty;

    public string Rejected { get; init; } = string.Empty;

    public string Reason { get; init; } = string.Empty;

    public string Json { get; init; } = string.Empty;

    public string Contrast { get; init; } = "unknown";

    public double TokenOverlap { get; init; }

    public int CharacterDelta { get; init; }

    public string DisplayName =>
        $"Example {RowNumber}: {Contrast} | overlap {TokenOverlap:P0} | {Truncate(Prompt)}";

    /// <summary>Second line for the pair-list item template: the pair's contrast band, derived straight
    /// from the real <see cref="Contrast"/> field (weak = chosen/rejected too similar; strong = clearly
    /// distinct). Kept honest to the field — a "weak" pair is not relabelled "high contrast".</summary>
    public string ContrastLine => $"{Contrast} contrast";

    public string ContrastSummary
    {
        get
        {
            var delta = CharacterDelta >= 0 ? $"+{CharacterDelta}" : CharacterDelta.ToString();
            return $"Contrast: {Contrast}. Token overlap: {TokenOverlap:P0}. Chosen/rejected character delta: {delta}.";
        }
    }

    private static string Truncate(string value, int maxLength = 64)
    {
        var trimmed = value.Trim();
        return trimmed.Length <= maxLength ? trimmed : $"{trimmed[..maxLength]}...";
    }
}
