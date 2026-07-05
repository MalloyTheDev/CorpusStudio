using System.Collections.ObjectModel;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete AI-Assist model-backend connection sub-view-model. Behaviour moved verbatim from
/// the shell (<c>MainWindowViewModel</c>) — the backend/model/base-url/timeout are two-way bound and
/// persisted per project; the discovered model list + its refresh summary populate on demand.</summary>
public sealed class AiAssistConnectionViewModel : ViewModelBase, IAiAssistConnectionViewModel
{
    private string _aiAssistBackend = "ollama";
    private string _aiAssistModel = "qwen2.5-coder:7b";
    private string _aiAssistBaseUrl = "http://localhost:11434";
    private string _aiAssistTimeoutSeconds = "120";
    private string _aiAssistModelListSummary =
        "Refresh models to load running Ollama or OpenAI-compatible models.";

    public string AiAssistBackend
    {
        get => _aiAssistBackend;
        set => SetField(ref _aiAssistBackend, value);
    }

    public string AiAssistModel
    {
        get => _aiAssistModel;
        set => SetField(ref _aiAssistModel, value);
    }

    public string AiAssistBaseUrl
    {
        get => _aiAssistBaseUrl;
        set => SetField(ref _aiAssistBaseUrl, value);
    }

    public string AiAssistTimeoutSeconds
    {
        get => _aiAssistTimeoutSeconds;
        set => SetField(ref _aiAssistTimeoutSeconds, value);
    }

    public ObservableCollection<string> AiAssistAvailableModels { get; } = [];

    public string AiAssistModelListSummary
    {
        get => _aiAssistModelListSummary;
        private set => SetField(ref _aiAssistModelListSummary, value);
    }

    public void SetModelListSummary(string message)
    {
        AiAssistModelListSummary = message;
    }
}
