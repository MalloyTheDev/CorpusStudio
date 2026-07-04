using System;
using System.Collections.Generic;
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
    }

    public void SetArenaError(string message)
    {
        ArenaSummary = $"Arena could not run.{Environment.NewLine}{message}";
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
                lines.Add(IndentBlock(string.IsNullOrWhiteSpace(response.Text) ? "(empty response)" : response.Text));
            }

            if (judged && judgment is not null)
            {
                lines.Add(judgment.Parsed
                    ? $"  judge: {judgment.Winner} — {SingleLine(judgment.Rationale)}"
                    : "  judge: (could not parse judge output)");
            }
        }

        ArenaSummary = string.Join(Environment.NewLine, lines);
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
