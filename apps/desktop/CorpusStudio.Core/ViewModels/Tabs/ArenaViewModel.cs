using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Model Arena tab view-model, extracted from the shell (backlog #4).
/// Behaviour and formatting moved verbatim.</summary>
public sealed class ArenaViewModel : ViewModelBase, IArenaViewModel
{
    private string _arenaPromptsInput = string.Empty;
    private string _arenaModelsInput = string.Empty;
    private string _arenaJudgeModelInput = string.Empty;
    private string _arenaSummary =
        "Enter prompts and models, then run the arena to compare responses side by side.";
    private bool _hasArenaReport;
    private string _arenaResultsLine = "Configure prompts and models below, then run the arena.";
    private string _headToHeadLabel = string.Empty;
    private bool _hasHeadToHead;

    public string ArenaPromptsInput
    {
        get => _arenaPromptsInput;
        set => SetField(ref _arenaPromptsInput, value);
    }

    public string ArenaModelsInput
    {
        get => _arenaModelsInput;
        set => SetField(ref _arenaModelsInput, value);
    }

    public string ArenaJudgeModelInput
    {
        get => _arenaJudgeModelInput;
        set => SetField(ref _arenaJudgeModelInput, value);
    }

    public string ArenaSummary
    {
        get => _arenaSummary;
        private set => SetField(ref _arenaSummary, value);
    }

    public bool HasArenaReport
    {
        get => _hasArenaReport;
        private set => SetField(ref _hasArenaReport, value);
    }

    public string ArenaResultsLine
    {
        get => _arenaResultsLine;
        private set => SetField(ref _arenaResultsLine, value);
    }

    public string HeadToHeadLabel
    {
        get => _headToHeadLabel;
        private set => SetField(ref _headToHeadLabel, value);
    }

    public bool HasHeadToHead
    {
        get => _hasHeadToHead;
        private set => SetField(ref _hasHeadToHead, value);
    }

    /// <summary>Ranked standings cards (top three by win count) for the current judged run.</summary>
    public ObservableCollection<ArenaStandingItem> Standings { get; } = [];

    /// <summary>Per-prompt head-to-head rows comparing the two top-ranked models.</summary>
    public ObservableCollection<ArenaHeadToHeadItem> HeadToHead { get; } = [];

    /// <summary>Split a comma/newline-separated model list into a trimmed, de-duplicated
    /// (case-insensitive, order-preserving) list.</summary>
    public static IReadOnlyList<string> ParseModelList(string text)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var models = new List<string>();
        foreach (var token in (text ?? string.Empty).Split(
                     [',', '\n', '\r'],
                     StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (seen.Add(token))
            {
                models.Add(token);
            }
        }

        return models;
    }

    public void SetArenaInProgress()
    {
        ArenaSummary = "Running arena...";
        ResetResults("Running arena...");
    }

    public void SetArenaError(string message)
    {
        ArenaSummary = $"Arena could not run.{Environment.NewLine}{message}";
        ResetResults("Arena could not run — see the transcript below.");
    }

    /// <summary>Clear the results-first surfaces back to a neutral pre-report state.</summary>
    private void ResetResults(string resultsLine)
    {
        Standings.Clear();
        HeadToHead.Clear();
        HeadToHeadLabel = string.Empty;
        HasHeadToHead = false;
        HasArenaReport = false;
        ArenaResultsLine = resultsLine;
    }

    public void ApplyArenaReport(ArenaReport report)
    {
        var judged = !string.IsNullOrWhiteSpace(report.JudgeModel);
        var judgmentByPrompt = report.Judgments
            .GroupBy(judgment => judgment.PromptId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Last(), StringComparer.Ordinal);

        var header = $"Arena: {report.Models.Count} model(s) × {report.PromptCount} prompt(s)";
        if (judged)
        {
            header += $"   judge: {report.JudgeModel}";
        }

        var lines = new List<string> { header, string.Empty, "Models:" };
        foreach (var summary in report.ModelSummaries)
        {
            var parts = new List<string> { $"{summary.WinCount} win(s)" };
            if (summary.AverageJudgeScore is { } score)
            {
                parts.Add($"avg {score:0.#}");
            }
            if (summary.EmptyResponseCount > 0)
            {
                parts.Add($"{summary.EmptyResponseCount} empty");
            }
            if (summary.ErrorCount > 0)
            {
                parts.Add($"⚠ {summary.ErrorCount} error(s)");
            }
            lines.Add($"  {summary.Model} — {string.Join(", ", parts)}");
        }

        foreach (var prompt in report.Prompts)
        {
            lines.Add(string.Empty);
            lines.Add($"── {prompt.Id}: {SingleLine(prompt.Prompt)}");
            judgmentByPrompt.TryGetValue(prompt.Id, out var judgment);
            foreach (var response in report.Responses.Where(r => r.PromptId == prompt.Id))
            {
                var win = judgment is not null && judgment.Winner == response.Model ? "🏆 " : "   ";
                lines.Add($"  {win}{response.Model}:");
                var body = !string.IsNullOrWhiteSpace(response.Error)
                    ? $"⚠ (backend error: {SingleLine(response.Error)})"
                    : string.IsNullOrWhiteSpace(response.Text) ? "(empty response)" : response.Text;
                lines.Add(IndentBlock(body));
            }

            if (judged && judgment is not null)
            {
                lines.Add(judgment.Parsed
                    ? $"  judge: {judgment.Winner} — {SingleLine(judgment.Rationale)}"
                    : "  judge: (could not parse judge output)");
            }
        }

        ArenaSummary = string.Join(Environment.NewLine, lines);

        BuildResultsSurfaces(report);
    }

    /// <summary>Populate the results-first surfaces (compact summary line, ranked standings
    /// cards, and the head-to-head strip) from real report fields.</summary>
    private void BuildResultsSurfaces(ArenaReport report)
    {
        var judged = report.Judgments.Count > 0;
        var judgedPrompts = report.Judgments.Count;
        var ties = report.Judgments.Count(j => string.IsNullOrEmpty(j.Winner));

        ArenaResultsLine = string.Format(
            System.Globalization.CultureInfo.InvariantCulture,
            "{0} prompt{1} · {2} model{3} · {4}",
            report.PromptCount,
            report.PromptCount == 1 ? string.Empty : "s",
            report.Models.Count,
            report.Models.Count == 1 ? string.Empty : "s",
            judged ? "pairwise judged" : "not yet judged");

        // Standings: rank by wins (desc), then average judge score (desc), then discovery order.
        var ordered = report.ModelSummaries
            .Select((summary, index) => (summary, index))
            .OrderByDescending(x => x.summary.WinCount)
            .ThenByDescending(x => x.summary.AverageJudgeScore ?? double.MinValue)
            .ThenBy(x => x.index)
            .Select(x => x.summary)
            .ToList();

        Standings.Clear();
        var rank = 0;
        foreach (var summary in ordered.Take(3))
        {
            rank++;
            Standings.Add(new ArenaStandingItem
            {
                Rank = rank,
                Model = summary.Model,
                Wins = summary.WinCount,
                Ties = ties,
                Losses = Math.Max(0, judgedPrompts - summary.WinCount - ties),
                JudgedPrompts = judgedPrompts,
            });
        }

        // Head-to-head: project the judge's single per-prompt winner onto the top-two lens.
        HeadToHead.Clear();
        if (judged && ordered.Count >= 2)
        {
            var a = ordered[0].Model;
            var b = ordered[1].Model;
            HeadToHeadLabel = $"HEAD-TO-HEAD · {ShortName(a)} vs {ShortName(b)}";

            var judgmentByPrompt = report.Judgments
                .GroupBy(j => j.PromptId, StringComparer.Ordinal)
                .ToDictionary(g => g.Key, g => g.Last(), StringComparer.Ordinal);

            foreach (var prompt in report.Prompts)
            {
                judgmentByPrompt.TryGetValue(prompt.Id, out var judgment);
                var winner = judgment?.Winner ?? string.Empty;
                string label;
                bool isWin;
                if (winner == a)
                {
                    label = $"{ShortName(a)} wins";
                    isWin = true;
                }
                else if (winner == b)
                {
                    label = $"{ShortName(b)} wins";
                    isWin = true;
                }
                else
                {
                    label = "tie";
                    isWin = false;
                }

                HeadToHead.Add(new ArenaHeadToHeadItem
                {
                    Prompt = SingleLine(prompt.Prompt),
                    ResultLabel = label,
                    IsWin = isWin,
                });
            }
        }
        else
        {
            HeadToHeadLabel = string.Empty;
        }

        HasHeadToHead = HeadToHead.Count > 0;
        HasArenaReport = true;
    }

    /// <summary>Drop the provider tag (":8b", ":7b", …) for a compact display name.</summary>
    private static string ShortName(string model)
    {
        if (string.IsNullOrEmpty(model))
        {
            return model;
        }

        var colon = model.IndexOf(':');
        return colon > 0 ? model[..colon] : model;
    }

    private static string SingleLine(string text)
    {
        return string.Join(" ", (text ?? string.Empty).Split(
            ['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
    }

    private static string IndentBlock(string text)
    {
        var body = (text ?? string.Empty).Replace("\r\n", "\n").Split('\n');
        return string.Join(Environment.NewLine, body.Select(line => "       " + line));
    }
}
