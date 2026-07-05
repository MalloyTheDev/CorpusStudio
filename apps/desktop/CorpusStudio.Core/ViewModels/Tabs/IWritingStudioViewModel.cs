using System.ComponentModel;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Writing Studio tab's own view-model (Phase-2 decomposition, backend-cluster slice 1).
/// Owns the draft editor buffer and its dirty-tracking: the buffer is "dirty" when
/// <see cref="DraftText"/> diverges from the last programmatically loaded/saved draft.
///
/// <para><see cref="LoadDraft"/> is the shared "load text into the editor" seam that many other
/// features call (edit a saved example, retry a quarantined row, prepare an AI-Assist review, a
/// synthetic/failure rewrite). The shell keeps the aggregate <c>HasUnsavedWork</c> (draft OR a dirty
/// Explorer document) and the draft-construction helpers. Behind an interface so the shell/tests/DI
/// depend on the contract.</para></summary>
public interface IWritingStudioViewModel : INotifyPropertyChanged
{
    string DraftText { get; set; }

    /// <summary>True when the editor buffer has unsaved user edits (differs from the last
    /// loaded/saved draft).</summary>
    bool IsDraftDirty { get; }

    /// <summary>Load a known draft (template, saved example, retried row) as the clean baseline, so it
    /// is not reported as unsaved until the user edits it.</summary>
    void LoadDraft(string text);

    /// <summary>Mark the current draft as saved — its content becomes the clean baseline.</summary>
    void MarkDraftClean();
}
