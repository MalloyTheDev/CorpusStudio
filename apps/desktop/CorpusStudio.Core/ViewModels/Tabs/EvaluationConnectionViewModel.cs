using System.Collections.ObjectModel;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Evaluation model-backend connection sub-view-model. Behaviour moved verbatim from
/// the shell (<c>MainWindowViewModel</c>) — the backend/model/base-url/timeout are two-way bound and
/// persisted per project; the discovered model list + its refresh summary populate on demand.</summary>
public sealed class EvaluationConnectionViewModel : ViewModelBase, IEvaluationConnectionViewModel
{
    private string _evaluationBackend = "ollama";
    private string _evaluationModel = "qwen2.5-coder:7b";
    private string _evaluationBaseUrl = "http://localhost:11434";
    private string _evaluationTimeoutSeconds = "120";
    private string _evaluationJudgeModel = string.Empty;
    private string _evaluationModelListSummary =
        "Refresh models to load running Ollama or OpenAI-compatible models.";

    public string EvaluationBackend
    {
        get => _evaluationBackend;
        set => SetField(ref _evaluationBackend, value);
    }

    public string EvaluationModel
    {
        get => _evaluationModel;
        set => SetField(ref _evaluationModel, value);
    }

    public string EvaluationBaseUrl
    {
        get => _evaluationBaseUrl;
        set => SetField(ref _evaluationBaseUrl, value);
    }

    public string EvaluationTimeoutSeconds
    {
        get => _evaluationTimeoutSeconds;
        set => SetField(ref _evaluationTimeoutSeconds, value);
    }

    public string EvaluationJudgeModel
    {
        get => _evaluationJudgeModel;
        set => SetField(ref _evaluationJudgeModel, value);
    }

    // Optional separate judge provider (#224): blank reuses the eval run's own backend/base-url.
    // The real use case is a local eval (Ollama) scored by a cloud judge (openai-compatible).
    private string _evaluationJudgeBackend = string.Empty;
    private string _evaluationJudgeBaseUrl = string.Empty;

    public string EvaluationJudgeBackend
    {
        get => _evaluationJudgeBackend;
        set => SetField(ref _evaluationJudgeBackend, value);
    }

    public string EvaluationJudgeBaseUrl
    {
        get => _evaluationJudgeBaseUrl;
        set => SetField(ref _evaluationJudgeBaseUrl, value);
    }

    public ObservableCollection<string> EvaluationAvailableModels { get; } = [];

    public string EvaluationModelListSummary
    {
        get => _evaluationModelListSummary;
        private set => SetField(ref _evaluationModelListSummary, value);
    }

    public void SetModelListSummary(string message)
    {
        EvaluationModelListSummary = message;
    }
}
