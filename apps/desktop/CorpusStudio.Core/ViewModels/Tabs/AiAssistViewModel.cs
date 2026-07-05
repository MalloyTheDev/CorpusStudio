using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete AI-Assist tab core view-model (backend-cluster slice 2, PR 3/3). Behaviour moved
/// verbatim from the shell (<c>MainWindowViewModel</c>) — the run + result panes, the honest candidate
/// gate (informational, never approval), and the review queue (filter/search/sort + saved views + bulk
/// triage). Reads the connection child for the run's backend/model; a split failure surfaces via
/// <see cref="ErrorReported"/> (the shell forwards it to its error banner). Honesty invariants intact:
/// the gate only informs, an errored run never keeps a stale verdict, and per-metric-style panes stay
/// truthful.</summary>
public sealed class AiAssistViewModel : ViewModelBase, IAiAssistViewModel
{
    private readonly IAiAssistConnectionViewModel _connection;

    /// <summary>Raised when an AI-Assist run fails; the shell forwards it to its shared error banner.</summary>
    public event Action<string>? ErrorReported;

    public AiAssistViewModel(IAiAssistConnectionViewModel connection)
    {
        _connection = connection;
    }

    private string _aiAssistAction = "review";
    private string _aiAssistInstruction =
        "Review the current draft and suggest safer tags or a stronger output.";
    private string _aiAssistSummary =
        "Run AI Assist on the current draft. Suggestions require human review.";
    private string _aiAssistReviewText = "AI Assist review output appears here.";
    private string _aiAssistSourceDraftText =
        "Source draft appears here after selecting an AI Assist review.";
    private string _aiAssistSuggestedJsonlText =
        "Suggested JSONL appears here after selecting an AI Assist review.";
    private string _aiAssistDiffSummary =
        "Select a queued AI Assist review to compare source and suggestion.";
    private string _aiAssistSuggestionJsonl = string.Empty;
    // The candidate gate for the current fresh run (before a queue item is selected).
    // When a queue item is selected, its persisted gate takes precedence (see
    // ActiveAiAssistCandidateGate). Never means "approved".
    private GateReport? _aiAssistCandidateGate;
    private string _aiAssistCandidateGateStatus = "—";
    private string _aiAssistCandidateGateColor = "#64748B";  // neutral gray until a run/selection sets it
    private string _aiAssistQueueSummary = "AI Assist review queue appears after suggestions are generated.";
    private string _aiAssistQueueFilter = "All";
    private string _aiAssistQueueSearch = string.Empty;
    private string _aiAssistQueueSort = "Newest";
    private string _aiAssistQueueViewName = "Review View";

    private AiAssistReviewQueueItem? _selectedAiAssistReviewQueueItem;

    private AiAssistQueueView? _selectedAiAssistQueueView;

    private readonly List<AiAssistReviewQueueItem> _allAiAssistReviewQueue = [];

    public ObservableCollection<AiAssistReviewQueueItem> AiAssistReviewQueue { get; } = [];

    public ObservableCollection<AiAssistQueueView> AiAssistQueueViews { get; } = [];

    public ObservableCollection<string> AiAssistQueueFilterOptions { get; } =
    [
        "All",
        "Pending",
        "Accepted",
        "Rejected",
    ];

    public ObservableCollection<string> AiAssistQueueSortOptions { get; } =
    [
        "Newest",
        "Oldest",
        "State",
        "Model",
        "Action",
    ];

    public ObservableCollection<string> AiAssistActionPresets { get; } =
    [
        "review",
        "suggest-tags",
        "rewrite-output",
        "draft-example",
    ];

    public string AiAssistAction
    {
        get => _aiAssistAction;
        set => SetField(ref _aiAssistAction, value);
    }

    public string AiAssistInstruction
    {
        get => _aiAssistInstruction;
        set => SetField(ref _aiAssistInstruction, value);
    }

    public string AiAssistSummary
    {
        get => _aiAssistSummary;
        // public set: the shell's backend health-check / move-to-draft bridges write it.
        set => SetField(ref _aiAssistSummary, value);
    }

    /// <summary>The current run's suggested JSONL (empty until a run produces one). Read by the shell's
    /// MoveAiAssistSuggestionToDraft bridge as a fallback when no queue item is selected.</summary>
    public string AiAssistSuggestionJsonl => _aiAssistSuggestionJsonl;

    public string AiAssistReviewText
    {
        get => _aiAssistReviewText;
        private set => SetField(ref _aiAssistReviewText, value);
    }

    public string AiAssistSourceDraftText
    {
        get => _aiAssistSourceDraftText;
        private set => SetField(ref _aiAssistSourceDraftText, value);
    }

    public string AiAssistSuggestedJsonlText
    {
        get => _aiAssistSuggestedJsonlText;
        private set => SetField(ref _aiAssistSuggestedJsonlText, value);
    }

    public string AiAssistDiffSummary
    {
        get => _aiAssistDiffSummary;
        private set => SetField(ref _aiAssistDiffSummary, value);
    }

    /// <summary>Short candidate-gate status label for the AI Assist tab header
    /// (e.g. "PASS", "BLOCK", "not run", "n/a", "—"). Informational — never approval.</summary>
    public string AiAssistCandidateGateStatus
    {
        get => _aiAssistCandidateGateStatus;
        private set => SetField(ref _aiAssistCandidateGateStatus, value);
    }

    /// <summary>Foreground hex for the candidate-gate status label. Neutral gray for
    /// null/unknown/"not run"/"n/a" — never green (see GateReport.StatusColor).</summary>
    public string AiAssistCandidateGateColor
    {
        get => _aiAssistCandidateGateColor;
        private set => SetField(ref _aiAssistCandidateGateColor, value);
    }

    /// <summary>The gate that applies to the suggestion that would move to the draft:
    /// the selected queue item's persisted gate, else the current run's gate. Mirrors
    /// the same fallback MoveAiAssistSuggestionToDraft uses.</summary>
    private GateReport? ActiveAiAssistCandidateGate =>
        SelectedAiAssistReviewQueueItem?.CandidateGate ?? _aiAssistCandidateGate;

    /// <summary>True only when the active candidate gate BLOCKS. Drives the
    /// confirm-on-block prompt before the suggestion is moved to the draft. Pure —
    /// the gate only informs; the human still decides (never auto-rejected).</summary>
    public bool SelectedAiAssistCandidateGateBlocks =>
        string.Equals(ActiveAiAssistCandidateGate?.OverallStatus, "block", StringComparison.OrdinalIgnoreCase);

    public AiAssistReviewQueueItem? SelectedAiAssistReviewQueueItem
    {
        get => _selectedAiAssistReviewQueueItem;
        set
        {
            if (SetField(ref _selectedAiAssistReviewQueueItem, value))
            {
                if (value is null)
                {
                    ClearAiAssistComparison();
                    // No item selected: the header reverts to the current fresh-run gate
                    // (ActiveAiAssistCandidateGate falls back to it), so the block-confirm
                    // and label stay consistent with what would move to the draft.
                    ApplyCandidateGateStatus(
                        _aiAssistCandidateGate,
                        !string.IsNullOrWhiteSpace(_aiAssistSuggestionJsonl));
                }
                else
                {
                    ApplyAiAssistReviewQueueItem(value);
                }
            }
        }
    }

    public string AiAssistQueueSummary
    {
        get => _aiAssistQueueSummary;
        private set => SetField(ref _aiAssistQueueSummary, value);
    }

    public string AiAssistQueueFilter
    {
        get => _aiAssistQueueFilter;
        set
        {
            if (SetField(ref _aiAssistQueueFilter, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueSearch
    {
        get => _aiAssistQueueSearch;
        set
        {
            if (SetField(ref _aiAssistQueueSearch, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueSort
    {
        get => _aiAssistQueueSort;
        set
        {
            if (SetField(ref _aiAssistQueueSort, value))
            {
                RebuildAiAssistReviewQueue();
            }
        }
    }

    public string AiAssistQueueViewName
    {
        get => _aiAssistQueueViewName;
        set => SetField(ref _aiAssistQueueViewName, value);
    }

    public AiAssistQueueView? SelectedAiAssistQueueView
    {
        get => _selectedAiAssistQueueView;
        set => SetField(ref _selectedAiAssistQueueView, value);
    }

    public void SetAiAssistInProgress()
    {
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                "Running AI Assist...",
                $"Action: {AiAssistAction}",
                $"Backend: {_connection.AiAssistBackend}",
                $"Model: {_connection.AiAssistModel}",
            ]
        );
        AiAssistReviewText = "Waiting for local model response.";
        _aiAssistSuggestionJsonl = string.Empty;
        ResetCandidateGateState();  // the prior run's verdict must not linger during a new run
        AiAssistSourceDraftText = "Current draft is being sent to AI Assist.";
        AiAssistSuggestedJsonlText = "Waiting for suggested JSONL.";
        AiAssistDiffSummary = "Comparison will appear after the review is queued.";
    }

    public void ApplyAiAssistRunResult(AiAssistRunResult result)
    {
        _aiAssistSuggestionJsonl = result.SuggestedJsonl;
        _aiAssistCandidateGate = result.CandidateGate;
        var hasSuggestedContent = !string.IsNullOrWhiteSpace(result.SuggestedJsonl);
        ApplyCandidateGateStatus(result.CandidateGate, hasSuggestedContent);
        var suggestionStatus = hasSuggestedContent
            ? "available for review"
            : "none";
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {result.Action}",
                $"Model: {result.Model}",
                $"Review state: {result.ReviewState}",
                $"Review required: {(result.ReviewRequired ? "yes" : "no")}",
                $"Suggested JSONL: {suggestionStatus}",
                $"Candidate gate: {AiAssistCandidateGateStatus}",
                $"Warnings: {result.Warnings.Count}",
                $"Validation errors: {result.ValidationErrors.Count}",
            ]
        );

        var lines = new List<string>
        {
            "Model output:",
            result.ModelOutput,
        };

        if (!string.IsNullOrWhiteSpace(result.SuggestedJsonl))
        {
            lines.Add("");
            lines.Add("Suggested JSONL:");
            lines.Add(result.SuggestedJsonl.TrimEnd());
        }

        lines.Add("");
        lines.Add(GateReport.RenderCandidateGate(result.CandidateGate, hasSuggestedContent));

        if (result.Warnings.Count > 0)
        {
            lines.Add("");
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        if (result.ValidationErrors.Count > 0)
        {
            lines.Add("");
            lines.Add("Suggested JSONL validation errors:");
            lines.AddRange(result.ValidationErrors.Select(error => $"- {error}"));
        }

        AiAssistReviewText = string.Join(Environment.NewLine, lines);
        AiAssistSourceDraftText = "Queued review will show the source draft after it is saved.";
        AiAssistSuggestedJsonlText = string.IsNullOrWhiteSpace(result.SuggestedJsonl)
            ? "No suggested JSONL for this review."
            : result.SuggestedJsonl.TrimEnd();
        AiAssistDiffSummary = string.IsNullOrWhiteSpace(result.SuggestedJsonl)
            ? "No suggested JSONL to compare."
            : "Suggested JSONL is available for side-by-side review after queue selection.";
    }

    private void ApplyCandidateGateStatus(GateReport? gate, bool hasSuggestedContent)
    {
        if (gate is not null)
        {
            AiAssistCandidateGateStatus = (gate.OverallStatus ?? string.Empty).ToUpperInvariant();
            AiAssistCandidateGateColor = GateReport.StatusColor(gate.OverallStatus);
        }
        else
        {
            AiAssistCandidateGateStatus = hasSuggestedContent ? "not run" : "n/a";
            AiAssistCandidateGateColor = GateReport.StatusColor(null);
        }
    }

    private void ResetCandidateGateState()
    {
        _aiAssistCandidateGate = null;
        AiAssistCandidateGateStatus = "—";
        AiAssistCandidateGateColor = GateReport.StatusColor(null);
    }

    public void SetAiAssistReviewQueue(IEnumerable<AiAssistReviewQueueItem> items)
    {
        _allAiAssistReviewQueue.Clear();
        _allAiAssistReviewQueue.AddRange(items);
        RebuildAiAssistReviewQueue();
    }

    public void SetAiAssistQueueViews(IEnumerable<AiAssistQueueView> views)
    {
        var selectedName = SelectedAiAssistQueueView?.Name;
        AiAssistQueueViews.Clear();
        foreach (var view in views)
        {
            AiAssistQueueViews.Add(view);
        }

        SelectedAiAssistQueueView = AiAssistQueueViews
            .FirstOrDefault(view => string.Equals(view.Name, selectedName, StringComparison.OrdinalIgnoreCase))
            ?? AiAssistQueueViews.FirstOrDefault();
    }

    public AiAssistQueueView BuildCurrentAiAssistQueueView()
    {
        return new AiAssistQueueView
        {
            Name = AiAssistQueueViewName.Trim(),
            Filter = AiAssistQueueFilter,
            Search = AiAssistQueueSearch.Trim(),
            Sort = AiAssistQueueSort,
        };
    }

    public void ApplyAiAssistQueueView(AiAssistQueueView view)
    {
        AiAssistQueueViewName = view.Name;
        AiAssistQueueFilter = IsAiAssistQueueFilterOption(view.Filter) ? view.Filter : "All";
        AiAssistQueueSearch = view.Search;
        AiAssistQueueSort = IsAiAssistQueueSortOption(view.Sort) ? view.Sort : "Newest";
    }

    public void ApplyAiAssistQueueViewSaved(AiAssistQueueView view)
    {
        AiAssistQueueSummary = $"AI Assist queue view saved: {view.Name}.";
    }

    public void ApplyAiAssistQueueViewLoaded(AiAssistQueueView view)
    {
        AiAssistQueueSummary = $"AI Assist queue view loaded: {view.Name}.";
    }

    public void ApplyAiAssistReviewQueueItem(AiAssistReviewQueueItem item)
    {
        _aiAssistSuggestionJsonl = item.SuggestedJsonl;
        var hasSuggestedContent = !string.IsNullOrWhiteSpace(item.SuggestedJsonl);
        ApplyCandidateGateStatus(item.CandidateGate, hasSuggestedContent);
        AiAssistSummary = string.Join(
            Environment.NewLine,
            [
                $"Action: {item.Action}",
                $"Model: {item.Model}",
                $"Review state: {item.ReviewState}",
                $"Suggested JSONL: {(hasSuggestedContent ? "available for review" : "none")}",
                $"Candidate gate: {AiAssistCandidateGateStatus}",
                $"Warnings: {item.Warnings.Count}",
                $"Validation errors: {item.ValidationErrors.Count}",
            ]
        );
        AiAssistReviewText = item.DetailText;
        AiAssistSourceDraftText = string.IsNullOrWhiteSpace(item.SourceDraft)
            ? "No source draft was recorded for this review."
            : item.SourceDraft.TrimEnd();
        AiAssistSuggestedJsonlText = string.IsNullOrWhiteSpace(item.SuggestedJsonl)
            ? "No suggested JSONL for this review."
            : item.SuggestedJsonl.TrimEnd();
        AiAssistDiffSummary = BuildAiAssistDiffSummary(item);
    }

    public void ApplyAiAssistReviewState(AiAssistReviewQueueItem item)
    {
        ApplyAiAssistReviewQueueItem(item);
        AiAssistQueueSummary = $"AI Assist review marked {item.ReviewState}.";
    }

    public void ApplyAiAssistBulkReviewState(int updatedCount, string reviewState, int undoStepsAvailable)
    {
        AiAssistQueueSummary =
            $"AI Assist bulk triage marked {updatedCount} review(s) {reviewState}. Undo steps available: {undoStepsAvailable}.";
    }

    public void ApplyAiAssistBulkUndoReviewState(int updatedCount, int undoStepsAvailable)
    {
        AiAssistQueueSummary =
            $"AI Assist bulk triage undo restored {updatedCount} review(s). Undo steps remaining: {undoStepsAvailable}.";
    }

    public IReadOnlyList<string> GetVisibleAiAssistReviewIds()
    {
        return AiAssistReviewQueue.Select(item => item.ReviewId).ToList();
    }

    public IReadOnlyDictionary<string, string> GetVisibleAiAssistReviewStates()
    {
        return AiAssistReviewQueue.ToDictionary(
            item => item.ReviewId,
            item => item.ReviewState,
            StringComparer.Ordinal
        );
    }

    public void SetAiAssistError(string message)
    {
        AiAssistSummary = $"AI Assist could not run.{Environment.NewLine}{message}";
        ErrorReported?.Invoke(message);
        AiAssistReviewText = "No AI Assist suggestion was produced.";
        _aiAssistSuggestionJsonl = string.Empty;
        ResetCandidateGateState();  // a failed run must not keep the previous run's verdict
        ClearAiAssistComparison(
            "No source draft was compared.",
            "No suggested JSONL was produced.",
            "No comparison is available."
        );
    }

    public void SetAiAssistQueueError(string message)
    {
        AiAssistQueueSummary = $"AI Assist review queue could not be updated.{Environment.NewLine}{message}";
    }

    private void ClearAiAssistComparison(
        string sourceMessage = "Source draft appears here after selecting an AI Assist review.",
        string suggestionMessage = "Suggested JSONL appears here after selecting an AI Assist review.",
        string summaryMessage = "Select a queued AI Assist review to compare source and suggestion."
    )
    {
        AiAssistSourceDraftText = sourceMessage;
        AiAssistSuggestedJsonlText = suggestionMessage;
        AiAssistDiffSummary = summaryMessage;
    }

    private void RebuildAiAssistReviewQueue()
    {
        var selectedReviewId = SelectedAiAssistReviewQueueItem?.ReviewId;
        AiAssistReviewQueue.Clear();
        SelectedAiAssistReviewQueueItem = null;

        foreach (var item in SortAiAssistReviewQueue(
            _allAiAssistReviewQueue
                .Where(MatchesAiAssistQueueFilter)
                .Where(MatchesAiAssistQueueSearch)
        ))
        {
            AiAssistReviewQueue.Add(item);
        }

        AiAssistQueueSummary = BuildAiAssistQueueSummary();
        SelectedAiAssistReviewQueueItem = AiAssistReviewQueue
            .FirstOrDefault(item => item.ReviewId == selectedReviewId)
            ?? AiAssistReviewQueue.FirstOrDefault();

        if (SelectedAiAssistReviewQueueItem is null)
        {
            _aiAssistSuggestionJsonl = string.Empty;
            AiAssistReviewText = _allAiAssistReviewQueue.Count == 0
                ? "AI Assist review output appears here."
                : "No AI Assist reviews match the current queue controls.";
            ClearAiAssistComparison(
                "No source draft is selected.",
                "No suggested JSONL is selected.",
                "No comparison is available for the current queue controls."
            );
        }
    }

    private bool MatchesAiAssistQueueFilter(AiAssistReviewQueueItem item)
    {
        return AiAssistQueueFilter switch
        {
            "Pending" => item.ReviewState == "review_required",
            "Accepted" => item.ReviewState == "accepted",
            "Rejected" => item.ReviewState == "rejected",
            _ => true,
        };
    }

    private bool MatchesAiAssistQueueSearch(AiAssistReviewQueueItem item)
    {
        var search = AiAssistQueueSearch.Trim();
        if (string.IsNullOrWhiteSpace(search))
        {
            return true;
        }

        return ContainsSearch(item.ReviewId, search)
            || ContainsSearch(item.ReviewState, search)
            || ContainsSearch(item.Action, search)
            || ContainsSearch(item.Model, search)
            || ContainsSearch(item.PromptTemplateId, search)
            || ContainsSearch(item.SourceDraft, search)
            || ContainsSearch(item.ModelOutput, search)
            || ContainsSearch(item.SuggestedJsonl, search)
            || item.Warnings.Any(warning => ContainsSearch(warning, search))
            || item.ValidationErrors.Any(error => ContainsSearch(error, search));
    }

    private IEnumerable<AiAssistReviewQueueItem> SortAiAssistReviewQueue(
        IEnumerable<AiAssistReviewQueueItem> items
    )
    {
        return AiAssistQueueSort switch
        {
            "Oldest" => items
                .OrderBy(item => item.CreatedAt)
                .ThenBy(item => item.Model, StringComparer.OrdinalIgnoreCase),
            "State" => items
                .OrderBy(item => item.ReviewState, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            "Model" => items
                .OrderBy(item => item.Model, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            "Action" => items
                .OrderBy(item => item.Action, StringComparer.OrdinalIgnoreCase)
                .ThenByDescending(item => item.CreatedAt),
            _ => items
                .OrderByDescending(item => item.CreatedAt)
                .ThenBy(item => item.Model, StringComparer.OrdinalIgnoreCase),
        };
    }

    private string BuildAiAssistQueueSummary()
    {
        if (_allAiAssistReviewQueue.Count == 0)
        {
            return "No AI Assist reviews are queued for this project.";
        }

        var pending = _allAiAssistReviewQueue.Count(item => item.ReviewState == "review_required");
        var accepted = _allAiAssistReviewQueue.Count(item => item.ReviewState == "accepted");
        var rejected = _allAiAssistReviewQueue.Count(item => item.ReviewState == "rejected");
        var search = string.IsNullOrWhiteSpace(AiAssistQueueSearch)
            ? "none"
            : AiAssistQueueSearch.Trim();
        return $"Queue: {pending} pending, {accepted} accepted, {rejected} rejected. Filter: {AiAssistQueueFilter}, search: {search}, sort: {AiAssistQueueSort}, showing {AiAssistReviewQueue.Count} of {_allAiAssistReviewQueue.Count}.";
    }

    private bool IsAiAssistQueueFilterOption(string value)
    {
        return AiAssistQueueFilterOptions.Any(option =>
            string.Equals(option, value, StringComparison.OrdinalIgnoreCase)
        );
    }

    private bool IsAiAssistQueueSortOption(string value)
    {
        return AiAssistQueueSortOptions.Any(option =>
            string.Equals(option, value, StringComparison.OrdinalIgnoreCase)
        );
    }

    public void ApplyAiAssistActionPresets(string schemaId)
    {
        var presets = schemaId switch
        {
            "instruction" or "chat" or "code" => new[]
            {
                "review",
                "suggest-tags",
                "rewrite-output",
                "draft-example",
            },
            "preference" => new[]
            {
                "review",
                "judge-preference-strength",
                "rewrite-output",
                "suggest-tags",
            },
            "raw_text" => new[]
            {
                "review",
                "suggest-tags",
            },
            _ => new[]
            {
                "review",
                "suggest-tags",
                "draft-example",
            },
        };

        AiAssistActionPresets.Clear();
        foreach (var preset in presets)
        {
            AiAssistActionPresets.Add(preset);
        }

        if (!AiAssistActionPresets.Contains(AiAssistAction))
        {
            AiAssistAction = AiAssistActionPresets.FirstOrDefault() ?? "review";
        }
    }

    private static string BuildAiAssistDiffSummary(AiAssistReviewQueueItem item)
    {
        if (string.IsNullOrWhiteSpace(item.SuggestedJsonl))
        {
            return "No suggested JSONL to compare.";
        }

        if (string.IsNullOrWhiteSpace(item.SourceDraft))
        {
            return "Suggested JSONL is available, but no source draft was recorded.";
        }

        var source = item.SourceDraft.TrimEnd();
        var suggestion = item.SuggestedJsonl.TrimEnd();
        if (string.Equals(source, suggestion, StringComparison.Ordinal))
        {
            return "Suggestion matches the source draft exactly.";
        }

        var delta = suggestion.Length - source.Length;
        var deltaText = delta >= 0 ? $"+{delta}" : delta.ToString();
        return $"Source lines: {CountLines(source)}; suggested lines: {CountLines(suggestion)}; character delta: {deltaText}.";
    }

    private static bool ContainsSearch(string value, string search)
    {
        return value.Contains(search, StringComparison.OrdinalIgnoreCase);
    }

    private static int CountLines(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return 0;
        }

        return text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n').Length;
    }

    /// <summary>Reset all run/queue/gate state on a project switch so nothing leaks across projects.</summary>
    public void Reset()
    {
        AiAssistSummary = "Run AI Assist on the current draft. Suggestions require human review.";
        AiAssistReviewText = "AI Assist review output appears here.";
        AiAssistQueueSummary = "AI Assist review queue appears after suggestions are generated.";
        AiAssistQueueViewName = "Review View";
        AiAssistQueueViews.Clear();
        SelectedAiAssistQueueView = null;
        _aiAssistSuggestionJsonl = string.Empty;
        ClearAiAssistComparison();
        _allAiAssistReviewQueue.Clear();
        AiAssistReviewQueue.Clear();
        SelectedAiAssistReviewQueueItem = null;
        ResetCandidateGateState();
    }
}
