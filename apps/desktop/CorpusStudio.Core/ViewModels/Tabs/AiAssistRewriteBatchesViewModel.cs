using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete AI-Assist saved rewrite-batches sub-view-model. Behaviour moved verbatim from the
/// shell (<c>MainWindowViewModel</c>) — synthetic batch triage prepares a rewrite batch that can be
/// saved and later resumed; the summary names the affected rows.</summary>
public sealed class AiAssistRewriteBatchesViewModel : ViewModelBase, IAiAssistRewriteBatchesViewModel
{
    private const string DefaultSummary =
        "Prepared rewrite batches appear here after synthetic batch triage.";

    private AiAssistRewriteBatch? _selectedAiAssistRewriteBatch;
    private AiAssistRewriteBatch? _lastPreparedAiAssistRewriteBatch;
    private string _aiAssistRewriteBatchSummary = DefaultSummary;

    public ObservableCollection<AiAssistRewriteBatch> AiAssistRewriteBatches { get; } = [];

    public AiAssistRewriteBatch? SelectedAiAssistRewriteBatch
    {
        get => _selectedAiAssistRewriteBatch;
        set
        {
            if (SetField(ref _selectedAiAssistRewriteBatch, value) && value is not null)
            {
                AiAssistRewriteBatchSummary =
                    $"Selected rewrite batch for rows {FormatRowNumbers(value.RowNumbers)}.";
            }
        }
    }

    public string AiAssistRewriteBatchSummary
    {
        get => _aiAssistRewriteBatchSummary;
        private set => SetField(ref _aiAssistRewriteBatchSummary, value);
    }

    public void SetAiAssistRewriteBatches(IEnumerable<AiAssistRewriteBatch> batches)
    {
        var selectedBatchId = SelectedAiAssistRewriteBatch?.BatchId;
        AiAssistRewriteBatches.Clear();
        foreach (var batch in batches)
        {
            AiAssistRewriteBatches.Add(batch);
        }

        SelectedAiAssistRewriteBatch = AiAssistRewriteBatches
            .FirstOrDefault(batch => string.Equals(batch.BatchId, selectedBatchId, StringComparison.Ordinal))
            ?? AiAssistRewriteBatches.FirstOrDefault();

        AiAssistRewriteBatchSummary = AiAssistRewriteBatches.Count == 0
            ? "No prepared rewrite batches are saved for this project."
            : $"Saved rewrite batches: {AiAssistRewriteBatches.Count}. Select one to resume.";
    }

    public bool TryGetLastPreparedAiAssistRewriteBatch(out AiAssistRewriteBatch batch, out string errorMessage)
    {
        if (_lastPreparedAiAssistRewriteBatch is null)
        {
            batch = new AiAssistRewriteBatch();
            errorMessage = "Prepare a synthetic batch rewrite before saving it.";
            return false;
        }

        batch = _lastPreparedAiAssistRewriteBatch;
        errorMessage = string.Empty;
        return true;
    }

    public void SetLastPrepared(AiAssistRewriteBatch batch)
    {
        _lastPreparedAiAssistRewriteBatch = batch;
    }

    public void ApplyAiAssistRewriteBatchSaved(AiAssistRewriteBatch batch)
    {
        AiAssistRewriteBatchSummary =
            $"Saved rewrite batch for rows {FormatRowNumbers(batch.RowNumbers)}.";
    }

    public void SetAiAssistRewriteBatchError(string message)
    {
        AiAssistRewriteBatchSummary = $"AI Assist rewrite batch could not be updated.{Environment.NewLine}{message}";
    }

    public void SetRewriteBatchSummary(string message)
    {
        AiAssistRewriteBatchSummary = message;
    }

    /// <summary>Clear the batch list/selection/last-prepared/summary on a project switch so nothing
    /// leaks across projects.</summary>
    public void Reset()
    {
        AiAssistRewriteBatches.Clear();
        SelectedAiAssistRewriteBatch = null;
        _lastPreparedAiAssistRewriteBatch = null;
        AiAssistRewriteBatchSummary = DefaultSummary;
    }

    /// <summary>Format a distinct, ordered, positive row-number list ("none" if empty). Public so the
    /// shell's Resume bridge can render its "Resumed ..." summary consistently.</summary>
    public static string FormatRowNumbers(IEnumerable<int> rowNumbers)
    {
        var rows = rowNumbers
            .Where(rowNumber => rowNumber > 0)
            .Distinct()
            .Order()
            .ToList();
        return rows.Count == 0 ? "none" : string.Join(", ", rows);
    }
}
