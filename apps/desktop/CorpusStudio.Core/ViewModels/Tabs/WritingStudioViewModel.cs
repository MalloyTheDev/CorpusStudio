using System;
using System.Text.Json;
using System.Text.Json.Nodes;

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
                // The structured form is a live projection of this JSON buffer — re-raise it so a
                // raw-editor edit (or a loaded draft) refreshes the Instruction/Input/Output fields.
                OnPropertyChanged(nameof(IsInstructionShapedDraft));
                OnPropertyChanged(nameof(DraftInstruction));
                OnPropertyChanged(nameof(DraftInput));
                OnPropertyChanged(nameof(DraftOutput));
            }
        }
    }

    public bool IsDraftDirty => !string.Equals(_draftText, _draftBaseline, StringComparison.Ordinal);

    /// <summary>Whether the draft parses as an instruction-shaped JSON object (has an
    /// <c>instruction</c> or <c>output</c> key). Only then is the structured Instruction/Input/Output
    /// form safe to edit — a chat/preference/malformed draft (which this returns false for) keeps the
    /// raw JSON editor so a structured-field edit can never corrupt a draft it can't round-trip. This
    /// is the honesty gate: <see cref="DraftText"/> stays the single source of truth the WPF head, save,
    /// and validation all read.</summary>
    public bool IsInstructionShapedDraft
    {
        get
        {
            try
            {
                return JsonNode.Parse(_draftText) is JsonObject o
                    && (o.ContainsKey("instruction") || o.ContainsKey("output"));
            }
            catch (JsonException)
            {
                return false;
            }
        }
    }

    /// <summary>The instruction field, projected from / written back to the <see cref="DraftText"/>
    /// JSON (so save/validate/dirty-tracking are unchanged).</summary>
    public string DraftInstruction
    {
        get => GetDraftField("instruction");
        set => SetDraftField("instruction", value);
    }

    /// <summary>The optional input/context field, projected from / written back to the JSON.</summary>
    public string DraftInput
    {
        get => GetDraftField("input");
        set => SetDraftField("input", value);
    }

    /// <summary>The output field, projected from / written back to the JSON.</summary>
    public string DraftOutput
    {
        get => GetDraftField("output");
        set => SetDraftField("output", value);
    }

    private string GetDraftField(string key)
    {
        try
        {
            if (JsonNode.Parse(_draftText) is JsonObject o
                && o.TryGetPropertyValue(key, out var v)
                && v is JsonValue jv
                && jv.TryGetValue<string>(out var s))
            {
                return s;
            }
        }
        catch (JsonException)
        {
            // fall through — a malformed draft has no structured value
        }
        return string.Empty;
    }

    private void SetDraftField(string key, string? value)
    {
        JsonObject obj;
        try
        {
            obj = JsonNode.Parse(_draftText) as JsonObject ?? new JsonObject();
        }
        catch (JsonException)
        {
            // Never structurally rewrite a draft we can't parse — the raw editor owns that case.
            return;
        }
        obj[key] = value ?? string.Empty;
        var json = obj.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        if (string.Equals(_draftText, json, StringComparison.Ordinal))
        {
            return;
        }
        // Update the buffer directly and re-raise only DraftText + IsDraftDirty + the shape gate.
        // Deliberately NOT re-raising the Draft{Instruction,Input,Output} projections here: the edited
        // TextBox already holds the value and the siblings are unchanged, so re-raising would risk a
        // mid-edit caret reset. LoadDraft / a raw-editor edit go through the DraftText SETTER, which
        // does re-raise all three so switching editors stays in sync.
        _draftText = json;
        OnPropertyChanged(nameof(DraftText));
        OnPropertyChanged(nameof(IsDraftDirty));
        OnPropertyChanged(nameof(IsInstructionShapedDraft));
    }

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
