using System.Collections.ObjectModel;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Model Arena tab's own view-model (backlog #4 decomposition). Self-contained:
/// inputs + a formatted report summary, no cross-tab lifecycle. Behind an interface so the
/// shell/tests/DI depend on the contract.</summary>
public interface IArenaViewModel : INotifyPropertyChanged
{
    string ArenaPromptsInput { get; set; }
    string ArenaModelsInput { get; set; }
    string ArenaJudgeModelInput { get; set; }
    string ArenaSummary { get; }

    /// <summary>Whether a report has been applied (drives results vs. empty state).</summary>
    bool HasArenaReport { get; }

    /// <summary>Compact results-first summary line, e.g. "14 prompts · 3 models · pairwise judged".</summary>
    string ArenaResultsLine { get; }

    /// <summary>Ranked standings cards (top models by win count) for the judged run.</summary>
    ObservableCollection<ArenaStandingItem> Standings { get; }

    /// <summary>Eyebrow label for the head-to-head strip, e.g. "HEAD-TO-HEAD · llama3.1 vs qwen2.5".</summary>
    string HeadToHeadLabel { get; }

    /// <summary>Whether the head-to-head strip has content (needs a judged run with two models).</summary>
    bool HasHeadToHead { get; }

    /// <summary>Per-prompt head-to-head rows comparing the two top-ranked models.</summary>
    ObservableCollection<ArenaHeadToHeadItem> HeadToHead { get; }

    void SetArenaInProgress();
    void SetArenaError(string message);

    /// <summary>Format an arena report: per-model win/score summary, then each prompt's
    /// side-by-side responses with the judge's winner marked.</summary>
    void ApplyArenaReport(ArenaReport report);
}
