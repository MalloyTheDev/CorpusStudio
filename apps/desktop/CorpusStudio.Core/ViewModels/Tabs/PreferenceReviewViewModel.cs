using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text.Json;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Preference Review tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>); honesty invariants unchanged — pairs are ranked by contrast (weak
/// pairs surface first as the likeliest DPO problems), the ranking summary counts weak/moderate/strong,
/// and the review is scoped to preference projects.</summary>
public sealed class PreferenceReviewViewModel : ViewModelBase, IPreferenceReviewViewModel
{
    private string _preferencePromptText = "Select a saved preference example to inspect the prompt.";
    private string _preferenceChosenText = "Chosen response appears here.";
    private string _preferenceRejectedText = "Rejected response appears here.";
    private string _preferenceReasonText = "Reason or preference notes appear here.";
    private string _preferenceRankingSummary =
        "Preference ranking appears after saved preference pairs are loaded.";
    private string _preferenceContrastFilter = "All";
    private string _preferenceExportFormat = "dpo";
    private string _preferenceReviewSummary =
        "Preference pair review is available for preference projects.";
    private PreferenceReviewItem? _selectedPreferenceReviewItem;
    private string _activeSchemaId = string.Empty;

    private readonly List<PreferenceReviewItem> _allPreferenceReviewItems = [];

    public ObservableCollection<PreferenceReviewItem> PreferenceReviewItems { get; } = [];

    public ObservableCollection<string> PreferenceContrastFilterOptions { get; } =
    [
        "All",
        "Weak",
        "Moderate",
        "Strong",
    ];

    public ObservableCollection<string> PreferenceExportFormatOptions { get; } =
    [
        "dpo",
        "kto",
        "reward",
    ];

    public PreferenceReviewItem? SelectedPreferenceReviewItem
    {
        get => _selectedPreferenceReviewItem;
        set
        {
            if (SetField(ref _selectedPreferenceReviewItem, value))
            {
                ApplyPreferenceReviewItem(value);
            }
        }
    }

    public string PreferencePromptText
    {
        get => _preferencePromptText;
        private set => SetField(ref _preferencePromptText, value);
    }

    public string PreferenceChosenText
    {
        get => _preferenceChosenText;
        private set => SetField(ref _preferenceChosenText, value);
    }

    public string PreferenceRejectedText
    {
        get => _preferenceRejectedText;
        private set => SetField(ref _preferenceRejectedText, value);
    }

    public string PreferenceReasonText
    {
        get => _preferenceReasonText;
        private set => SetField(ref _preferenceReasonText, value);
    }

    public string PreferenceRankingSummary
    {
        get => _preferenceRankingSummary;
        private set => SetField(ref _preferenceRankingSummary, value);
    }

    public string PreferenceContrastFilter
    {
        get => _preferenceContrastFilter;
        set
        {
            if (SetField(ref _preferenceContrastFilter, value))
            {
                RebuildPreferenceReviewItems();
            }
        }
    }

    public string PreferenceReviewSummary
    {
        get => _preferenceReviewSummary;
        private set => SetField(ref _preferenceReviewSummary, value);
    }

    public string PreferenceExportFormat
    {
        get => _preferenceExportFormat;
        set => SetField(ref _preferenceExportFormat, value);
    }

    /// <summary>Mirror of the shell's ActiveSchemaId, pushed on project switch.</summary>
    public string ActiveSchemaId
    {
        get => _activeSchemaId;
        set => SetField(ref _activeSchemaId, value);
    }

    public IReadOnlyList<PreferenceReviewItem> GetVisiblePreferenceReviewItems()
    {
        return PreferenceReviewItems.ToList();
    }

    public void ApplyPreferenceRankingExport(string outputPath, int itemCount)
    {
        PreferenceReviewSummary =
            $"Exported {itemCount} visible preference ranking item(s) for DPO review: {outputPath}";
    }

    public void ApplyPreferenceTrainingExport(PreferenceExportResult result)
    {
        var lines = new List<string>
        {
            $"Exported {result.OutputRows} row(s) as {result.Format}: {result.OutputPath}",
        };

        if (result.DroppedDegenerate > 0)
        {
            lines.Add($"Dropped {result.DroppedDegenerate} degenerate pair(s).");
        }

        if (result.Warnings.Count > 0)
        {
            lines.Add("Warnings:");
            lines.AddRange(result.Warnings.Select(warning => $"- {warning}"));
        }

        PreferenceReviewSummary = string.Join(Environment.NewLine, lines);
    }

    public void SetPreferenceRankingExportError(string message)
    {
        PreferenceReviewSummary = $"Preference ranking export failed.{Environment.NewLine}{message}";
    }

    /// <summary>Set the review summary pane. Used by the shell's AI-Assist handoff actions (the
    /// preference↔AI-Assist bridge) until AI Assist is decomposed.</summary>
    public void SetReviewSummary(string message)
    {
        PreferenceReviewSummary = message;
    }

    public void SetItems(IEnumerable<SavedExampleItem> examples)
    {
        _allPreferenceReviewItems.Clear();
        if (ActiveSchemaId == "preference")
        {
            foreach (var example in examples)
            {
                if (TryCreatePreferenceReviewItem(example, out var item))
                {
                    _allPreferenceReviewItems.Add(item);
                }
            }
        }

        RebuildPreferenceReviewItems();
    }

    /// <summary>Reset all pair/selection/filter/pane state on a project switch so nothing leaks
    /// across projects. The shell pushes the new schema via <see cref="ActiveSchemaId"/> first.</summary>
    public void Reset()
    {
        _allPreferenceReviewItems.Clear();
        PreferenceReviewItems.Clear();
        SelectedPreferenceReviewItem = null;
        PreferenceContrastFilter = "All";
        ClearPreferenceReview();
    }

    private void RebuildPreferenceReviewItems()
    {
        var selectedRowNumber = SelectedPreferenceReviewItem?.RowNumber;
        PreferenceReviewItems.Clear();
        SelectedPreferenceReviewItem = null;

        foreach (var item in _allPreferenceReviewItems
            .Where(MatchesPreferenceContrastFilter)
            .OrderBy(item => PreferenceContrastRank(item.Contrast))
            .ThenByDescending(item => item.TokenOverlap)
            .ThenBy(item => item.RowNumber))
        {
            PreferenceReviewItems.Add(item);
        }

        PreferenceRankingSummary = BuildPreferenceRankingSummary();
        SelectedPreferenceReviewItem = PreferenceReviewItems
            .FirstOrDefault(item => item.RowNumber == selectedRowNumber)
            ?? PreferenceReviewItems.FirstOrDefault();

        if (PreferenceReviewItems.Count == 0)
        {
            ClearPreferenceReview();
        }
    }

    private bool MatchesPreferenceContrastFilter(PreferenceReviewItem item)
    {
        return PreferenceContrastFilter switch
        {
            "Weak" => item.Contrast == "weak",
            "Moderate" => item.Contrast == "moderate",
            "Strong" => item.Contrast == "strong",
            _ => true,
        };
    }

    private string BuildPreferenceRankingSummary()
    {
        if (ActiveSchemaId != "preference")
        {
            return "Preference ranking is available for preference projects.";
        }

        if (_allPreferenceReviewItems.Count == 0)
        {
            return "No valid preference pairs are loaded.";
        }

        var weak = _allPreferenceReviewItems.Count(item => item.Contrast == "weak");
        var moderate = _allPreferenceReviewItems.Count(item => item.Contrast == "moderate");
        var strong = _allPreferenceReviewItems.Count(item => item.Contrast == "strong");
        return $"Ranking: {weak} weak, {moderate} moderate, {strong} strong. Filter: {PreferenceContrastFilter}, showing {PreferenceReviewItems.Count} of {_allPreferenceReviewItems.Count}.";
    }

    private void ApplyPreferenceReviewItem(PreferenceReviewItem? item)
    {
        if (item is null)
        {
            ClearPreferenceReview();
            return;
        }

        if (ActiveSchemaId != "preference")
        {
            PreferencePromptText = "Preference review is intended for preference projects.";
            PreferenceChosenText = "Create or select a preference project to inspect chosen responses.";
            PreferenceRejectedText = "Create or select a preference project to inspect rejected responses.";
            PreferenceReasonText = "No preference reason is selected.";
            PreferenceReviewSummary = "Current project is not a preference dataset.";
            return;
        }

        PreferencePromptText = item.Prompt;
        PreferenceChosenText = item.Chosen;
        PreferenceRejectedText = item.Rejected;
        PreferenceReasonText = string.IsNullOrWhiteSpace(item.Reason)
            ? "No reason field is present for this pair."
            : item.Reason;
        PreferenceReviewSummary = $"Example {item.RowNumber}. {item.ContrastSummary}";
    }

    private void ClearPreferenceReview()
    {
        PreferencePromptText = "Select a saved preference example to inspect the prompt.";
        PreferenceChosenText = "Chosen response appears here.";
        PreferenceRejectedText = "Rejected response appears here.";
        PreferenceReasonText = "Reason or preference notes appear here.";
        PreferenceReviewSummary = ActiveSchemaId == "preference"
            ? "Save or select a preference pair to review chosen/rejected contrast."
            : "Preference pair review is available for preference projects.";
    }

    private static bool TryCreatePreferenceReviewItem(
        SavedExampleItem example,
        out PreferenceReviewItem item
    )
    {
        item = new PreferenceReviewItem();
        if (!TryReadPreferencePair(
            example.Json,
            out var prompt,
            out var chosen,
            out var rejected,
            out var reason,
            out _,
            out _
        ))
        {
            return false;
        }

        var metrics = BuildPreferenceContrastMetrics(chosen, rejected);
        item = new PreferenceReviewItem
        {
            RowNumber = example.RowNumber,
            Prompt = prompt,
            Chosen = chosen,
            Rejected = rejected,
            Reason = reason,
            Json = example.Json,
            Contrast = metrics.Contrast,
            TokenOverlap = metrics.TokenOverlap,
            CharacterDelta = metrics.CharacterDelta,
        };
        return true;
    }

    private static bool TryReadPreferencePair(
        string json,
        out string prompt,
        out string chosen,
        out string rejected,
        out string reason,
        out string contrastSummary,
        out string errorMessage
    )
    {
        prompt = string.Empty;
        chosen = string.Empty;
        rejected = string.Empty;
        reason = string.Empty;
        contrastSummary = string.Empty;
        errorMessage = string.Empty;

        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind != JsonValueKind.Object)
            {
                errorMessage = "Preference row must be a JSON object.";
                return false;
            }

            var root = document.RootElement;
            prompt = ReadStringProperty(root, "prompt");
            chosen = ReadStringProperty(root, "chosen");
            rejected = ReadStringProperty(root, "rejected");
            reason = ReadStringProperty(root, "reason");
        }
        catch (JsonException)
        {
            errorMessage = "Preference row contains invalid JSON.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(prompt)
            || string.IsNullOrWhiteSpace(chosen)
            || string.IsNullOrWhiteSpace(rejected))
        {
            errorMessage = "Preference row must include non-empty prompt, chosen, and rejected fields.";
            return false;
        }

        contrastSummary = BuildPreferenceContrastSummary(chosen, rejected);
        return true;
    }

    private static string ReadStringProperty(JsonElement root, string propertyName)
    {
        return root.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static string BuildPreferenceContrastSummary(string chosen, string rejected)
    {
        var metrics = BuildPreferenceContrastMetrics(chosen, rejected);
        var deltaText = metrics.CharacterDelta >= 0
            ? $"+{metrics.CharacterDelta}"
            : metrics.CharacterDelta.ToString();
        return $"Contrast: {metrics.Contrast}. Token overlap: {metrics.TokenOverlap:P0}. Chosen/rejected character delta: {deltaText}.";
    }

    private static (string Contrast, double TokenOverlap, int CharacterDelta) BuildPreferenceContrastMetrics(
        string chosen,
        string rejected
    )
    {
        var chosenTokens = TokenizeForPreference(chosen).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var rejectedTokens = TokenizeForPreference(rejected).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var overlap = 0.0;
        if (chosenTokens.Count > 0 && rejectedTokens.Count > 0)
        {
            var shared = chosenTokens.Intersect(rejectedTokens, StringComparer.OrdinalIgnoreCase).Count();
            overlap = shared / (double)Math.Max(chosenTokens.Count, rejectedTokens.Count);
        }

        var contrast = overlap switch
        {
            >= 0.9 => "weak",
            >= 0.65 => "moderate",
            _ => "strong",
        };
        return (contrast, overlap, chosen.Length - rejected.Length);
    }

    private static int PreferenceContrastRank(string contrast)
    {
        return contrast switch
        {
            "weak" => 0,
            "moderate" => 1,
            "strong" => 2,
            _ => 3,
        };
    }

    private static IEnumerable<string> TokenizeForPreference(string value)
    {
        return value
            .ToLowerInvariant()
            .Split(
                [' ', '\r', '\n', '\t', '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']'],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries
            )
            .Where(token => token.Length > 1);
    }
}
