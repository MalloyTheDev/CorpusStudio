using System.Globalization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One ranked row in the Model Arena standings (display-only projection of a
/// judged <see cref="ArenaModelSummary"/>). Win-rate and the W·L·T record are computed
/// from the real per-model win count and the judged-prompt total — never invented. A run
/// with no judge produces no judgments, so <see cref="IsJudged"/> is false and the row
/// shows a neutral "not yet judged" state instead of a fabricated percentage.</summary>
public sealed class ArenaStandingItem
{
    /// <summary>1-based placement after sorting by wins (descending).</summary>
    public int Rank { get; init; }

    public string Model { get; init; } = string.Empty;

    /// <summary>Prompts the judge picked this model as the winner.</summary>
    public int Wins { get; init; }

    /// <summary>Prompts a different model won.</summary>
    public int Losses { get; init; }

    /// <summary>Prompts with no decisive winner (judge tie or unparsed judgment).</summary>
    public int Ties { get; init; }

    /// <summary>Total judged prompts (0 when the run was not judged).</summary>
    public int JudgedPrompts { get; init; }

    public bool IsJudged => JudgedPrompts > 0;

    /// <summary>The #1 row gets a trophy glyph.</summary>
    public bool IsFirst => Rank == 1;

    /// <summary>Share of judged prompts this model won, 0..1 (drives the bar fill).</summary>
    public double WinRate => IsJudged ? (double)Wins / JudgedPrompts : 0d;

    /// <summary>Big right-aligned percentage; a dash when there is nothing to rank yet.</summary>
    public string WinRateDisplay =>
        IsJudged
            ? (WinRate * 100).ToString("0", CultureInfo.InvariantCulture) + "%"
            : "—";

    /// <summary>Compact win/loss/tie record shown beneath the bar.</summary>
    public string RecordDisplay =>
        IsJudged
            ? string.Format(CultureInfo.InvariantCulture, "{0}W · {1}L · {2}T", Wins, Losses, Ties)
            : "not yet judged";

    public string RankDisplay => Rank.ToString(CultureInfo.InvariantCulture);
}

/// <summary>One prompt row in the head-to-head strip comparing the two top-ranked models.
/// The judge records a single overall winner per prompt; this projects that pick onto the
/// A-vs-B lens: a win chip when the judge favoured either compared model, a muted "tie"
/// chip when it favoured neither.</summary>
public sealed class ArenaHeadToHeadItem
{
    public string Prompt { get; init; } = string.Empty;

    /// <summary>Chip text, e.g. "llama3.1 wins" or "tie".</summary>
    public string ResultLabel { get; init; } = "tie";

    /// <summary>True → accent win chip; false → muted tie chip.</summary>
    public bool IsWin { get; init; }
}
