using System.Collections.ObjectModel;
using System.ComponentModel;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The AI-Assist model-backend connection sub-view-model (backend-cluster slice 2, PR 2/3).
/// Holds the backend connection config (backend/model/base-url/timeout — two-way bound and persisted
/// per project via the Lab settings), the discovered model list, and the model-list refresh summary.
///
/// <para>The backend OPERATIONS stay on the shell for now: health-check + the model-list refresh write
/// the run-core summary or use shared backend helpers, and the Lab-settings orchestrator applies/reads
/// these fields. They reach in via this contract (and <see cref="SetModelListSummary"/>). Behind an
/// interface so the shell/tests/DI depend on the contract.</para></summary>
public interface IAiAssistConnectionViewModel : INotifyPropertyChanged
{
    string AiAssistBackend { get; set; }
    string AiAssistModel { get; set; }
    string AiAssistBaseUrl { get; set; }
    string AiAssistTimeoutSeconds { get; set; }
    ObservableCollection<string> AiAssistAvailableModels { get; }
    string AiAssistModelListSummary { get; }

    /// <summary>Set the model-list refresh summary. Used by the shell's model-list operations.</summary>
    void SetModelListSummary(string message);
}
