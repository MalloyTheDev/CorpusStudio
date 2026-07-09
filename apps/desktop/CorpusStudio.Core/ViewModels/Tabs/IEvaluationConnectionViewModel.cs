using System.Collections.ObjectModel;
using System.ComponentModel;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Evaluation model-backend connection sub-view-model (backend-cluster slice 3, PR 3a).
/// Holds the backend connection config (backend/model/base-url/timeout — two-way bound and persisted
/// per project via the Lab settings), the discovered model list, and the model-list refresh summary.
/// Mirrors <see cref="IAiAssistConnectionViewModel"/>.
///
/// <para>The backend OPERATIONS stay on the shell for now: health-check + the model-list refresh write
/// the Evaluation summary or use shared backend helpers, and the Lab-settings orchestrator applies/reads
/// these fields. They reach in via this contract (and <see cref="SetModelListSummary"/>). Behind an
/// interface so the shell/tests/DI depend on the contract.</para></summary>
public interface IEvaluationConnectionViewModel : INotifyPropertyChanged
{
    string EvaluationBackend { get; set; }
    string EvaluationModel { get; set; }
    string EvaluationBaseUrl { get; set; }
    string EvaluationTimeoutSeconds { get; set; }

    /// <summary>Optional evaluator model for the opt-in LLM-judge scorer (metric <c>llm_judge</c>). Blank =
    /// keyword-overlap only. When set, the eval run scores each answer 0–100 with a rationale using this
    /// model (evaluator-only; not a quality guarantee).</summary>
    string EvaluationJudgeModel { get; set; }
    string EvaluationJudgeBackend { get; set; }
    string EvaluationJudgeBaseUrl { get; set; }
    ObservableCollection<string> EvaluationAvailableModels { get; }
    string EvaluationModelListSummary { get; }

    /// <summary>Set the model-list refresh summary. Used by the shell's model-list operations.</summary>
    void SetModelListSummary(string message);
}
