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

    void SetArenaInProgress();
    void SetArenaError(string message);

    /// <summary>Format an arena report: per-model win/score summary, then each prompt's
    /// side-by-side responses with the judge's winner marked.</summary>
    void ApplyArenaReport(ArenaReport report);
}
