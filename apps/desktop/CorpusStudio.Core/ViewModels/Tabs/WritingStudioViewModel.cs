using System;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Writing Studio tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>) — the draft editor buffer with baseline dirty-tracking, so loading a
/// known draft (template, saved example, retried row) is not reported as unsaved work until the user
/// edits it. The shell folds <see cref="IsDraftDirty"/> into its aggregate HasUnsavedWork.</summary>
public sealed class WritingStudioViewModel : ViewModelBase, IWritingStudioViewModel
{
    private const string InitialDraftTemplate =
        "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}";

    private string _draftText = InitialDraftTemplate;

    // The last programmatically loaded/saved draft. The buffer is "dirty" (unsaved user edits)
    // when DraftText diverges from this. Set via LoadDraft/MarkDraftClean so that loading a
    // known draft (template, saved example, retried row) is not reported as unsaved work.
    private string _draftBaseline = InitialDraftTemplate;

    public string DraftText
    {
        get => _draftText;
        set
        {
            if (SetField(ref _draftText, value))
            {
                OnPropertyChanged(nameof(IsDraftDirty));
            }
        }
    }

    public bool IsDraftDirty => !string.Equals(_draftText, _draftBaseline, StringComparison.Ordinal);

    public void LoadDraft(string text)
    {
        _draftBaseline = text ?? string.Empty;
        DraftText = text ?? string.Empty; // the setter re-raises IsDraftDirty
    }

    public void MarkDraftClean()
    {
        _draftBaseline = _draftText;
        OnPropertyChanged(nameof(IsDraftDirty));
    }
}
