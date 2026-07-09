using System.Threading.Tasks;

namespace CorpusStudio.Desktop.Services;

/// <summary>Which buttons a confirm dialog shows.</summary>
public enum DialogButtons
{
    YesNo,
    OkCancel,
}

/// <summary>Dialog severity → icon.</summary>
public enum DialogSeverity
{
    Information,
    Warning,
    Error,
    Question,
}

/// <summary>Head-agnostic dialog seam so shared logic (and, during the cross-platform port, either
/// UI head) can prompt without a hard WPF dependency. Async because non-WPF dialogs (Avalonia) are
/// async; the WPF adapter completes synchronously. See docs/AVALONIA_MIGRATION_PLAN.md (Phase 0).
///
/// NOTE: <c>Window.Closing</c> confirmations must stay synchronous (they set <c>e.Cancel</c> before
/// returning), so those are intentionally NOT routed through this async seam.</summary>
public interface IDialogService
{
    /// <summary>Show a confirm dialog; returns true on the affirmative button (Yes / OK).
    /// <paramref name="defaultAffirmative"/> makes the affirmative the focused default button;
    /// leave false for the safe/negative default (No / Cancel).</summary>
    Task<bool> ConfirmAsync(
        string message,
        string title,
        DialogButtons buttons = DialogButtons.YesNo,
        DialogSeverity severity = DialogSeverity.Question,
        bool defaultAffirmative = false);

    /// <summary>Show an informational acknowledge dialog (single OK).</summary>
    Task ShowAsync(string message, string title, DialogSeverity severity = DialogSeverity.Information);

    /// <summary>Prompt for a single line of text (OK/Cancel). Returns the entered text, or null if the
    /// user cancelled. Used for explorer new-file / new-folder / rename.</summary>
    Task<string?> PromptAsync(string title, string message, string defaultValue = "");
}
