using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

public sealed class PythonEngineService
{
    private sealed record EngineProcessResult(int ExitCode, string Output, string Error);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    private readonly string _repositoryRoot;
    private readonly string _engineDirectory;
    private readonly string _pythonExecutable;
    private readonly IReadOnlyDictionary<string, string> _localEnvironment;

    public PythonEngineService()
    {
        _repositoryRoot = FindRepositoryRoot();
        _localEnvironment = LoadLocalEnvironment(_repositoryRoot);
        _engineDirectory = ResolveEngineDirectory(_repositoryRoot, _localEnvironment);
        _pythonExecutable = ResolvePythonExecutable(_engineDirectory);
    }

    public async Task<IReadOnlyList<DatasetSchema>> GetSchemasAsync()
    {
        var output = await RunEngineCommandAsync("schemas");
        return JsonSerializer.Deserialize<List<DatasetSchema>>(output, JsonOptions) ?? [];
    }

    public Task<string> CreateProjectAsync(string projectId, string name, string schemaId)
    {
        return RunEngineCommandAsync("new-project", projectId, name, schemaId);
    }

    public IReadOnlyList<DatasetProjectListItem> LoadProjects()
    {
        var projectRoot = ResolveProjectRoot();
        if (!Directory.Exists(projectRoot))
        {
            return [];
        }

        var projects = new List<DatasetProjectListItem>();
        foreach (var projectFile in Directory.EnumerateFiles(projectRoot, "project.json", SearchOption.AllDirectories))
        {
            try
            {
                var json = File.ReadAllText(projectFile);
                var project = JsonSerializer.Deserialize<DatasetProject>(json, JsonOptions);
                if (project is not null)
                {
                    projects.Add(new DatasetProjectListItem(
                        project,
                        Path.GetDirectoryName(projectFile) ?? projectRoot
                    ));
                }
            }
            catch (JsonException)
            {
                continue;
            }
        }

        return projects
            .OrderBy(project => project.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public async Task<ProjectIndexRebuildResult> RebuildProjectIndexAsync()
    {
        var projectRoot = ResolveProjectRoot();
        var output = await RunEngineCommandAsync("project-index-rebuild", "--root", projectRoot);
        return JsonSerializer.Deserialize<ProjectIndexRebuildResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The engine returned an invalid project index result.");
    }

    public async Task<IReadOnlyList<DatasetProjectListItem>> LoadProjectsFromIndexAsync()
    {
        var projectRoot = ResolveProjectRoot();
        var output = await RunEngineCommandAsync("project-list", "--root", projectRoot);
        var report = JsonSerializer.Deserialize<ProjectIndexListReport>(output, JsonOptions);
        if (report?.Projects is null)
        {
            return [];
        }

        var items = new List<DatasetProjectListItem>();
        foreach (var entry in report.Projects)
        {
            if (string.IsNullOrWhiteSpace(entry.Id))
            {
                continue;
            }

            if (!DateTime.TryParse(entry.CreatedAt, out var createdAt))
            {
                createdAt = DateTime.UtcNow;
            }
            if (!DateTime.TryParse(entry.UpdatedAt, out var updatedAt))
            {
                updatedAt = createdAt;
            }
            var project = new DatasetProject(
                entry.Id,
                string.IsNullOrWhiteSpace(entry.Name) ? entry.Id : entry.Name,
                entry.SchemaId,
                createdAt,
                updatedAt
            );
            items.Add(new DatasetProjectListItem(
                project,
                string.IsNullOrWhiteSpace(entry.Path)
                    ? Path.Combine(projectRoot, entry.Id)
                    : entry.Path
            ));
        }

        return items
            .OrderBy(item => item.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public IReadOnlyList<SavedExampleItem> LoadExamples(string projectPath)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            return [];
        }

        var examples = new List<SavedExampleItem>();
        var rowNumber = 0;
        foreach (var rawLine in File.ReadLines(examplesPath, Encoding.UTF8))
        {
            if (string.IsNullOrWhiteSpace(rawLine))
            {
                continue;
            }

            rowNumber++;
            examples.Add(BuildSavedExampleItem(rowNumber, rawLine));
        }

        return examples;
    }

    public IReadOnlyList<QualityHistoryEntry> LoadQualityHistory(
        string projectPath,
        int maxEntries = 5
    )
    {
        var historyPath = Path.Combine(projectPath, "quality_history.jsonl");
        if (!File.Exists(historyPath))
        {
            return [];
        }

        return File.ReadLines(historyPath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(TryDeserializeQualityHistoryEntry)
            .Where(entry => entry is not null)
            .Cast<QualityHistoryEntry>()
            .OrderByDescending(entry => entry.RecordedAt)
            .Take(maxEntries)
            .ToList();
    }

    public QualityHistoryEntry SaveQualityHistoryEntry(string projectPath, QualityReport report)
    {
        Directory.CreateDirectory(projectPath);
        var entry = QualityHistoryEntry.FromReport(report);
        var historyPath = Path.Combine(projectPath, "quality_history.jsonl");
        var json = JsonSerializer.Serialize(entry, JsonOptions) + Environment.NewLine;
        File.AppendAllText(historyPath, json, encoding: Utf8NoBom);
        return entry;
    }

    public IReadOnlyList<ImportQuarantineItem> LoadImportQuarantineItems(string projectPath)
    {
        var quarantineDirectory = Path.Combine(projectPath, "import_quarantine");
        if (!Directory.Exists(quarantineDirectory))
        {
            return [];
        }

        var items = new List<ImportQuarantineItem>();
        foreach (var path in Directory
            .EnumerateFiles(quarantineDirectory, "*_rejected.jsonl")
            .OrderByDescending(File.GetLastWriteTimeUtc))
        {
            foreach (var rawLine in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(rawLine))
                {
                    continue;
                }

                try
                {
                    var item = JsonSerializer.Deserialize<ImportQuarantineItem>(rawLine, JsonOptions);
                    if (item is not null)
                    {
                        item.QuarantinePath = path;
                        items.Add(item);
                    }
                }
                catch (JsonException)
                {
                    continue;
                }
            }
        }

        return items;
    }

    public IReadOnlyList<AiAssistReviewQueueItem> LoadAiAssistReviewQueue(
        string projectPath,
        int maxItems = 50
    )
    {
        var queuePath = GetAiAssistReviewQueuePath(projectPath);
        if (!File.Exists(queuePath))
        {
            return [];
        }

        return File.ReadLines(queuePath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(TryDeserializeAiAssistReviewQueueItem)
            .Where(item => item is not null)
            .Cast<AiAssistReviewQueueItem>()
            .OrderByDescending(item => item.CreatedAt)
            .Take(maxItems)
            .ToList();
    }

    public IReadOnlyList<AiAssistQueueView> LoadAiAssistQueueViews(string projectPath)
    {
        var viewsPath = GetAiAssistQueueViewsPath(projectPath);
        if (!File.Exists(viewsPath))
        {
            return [];
        }

        try
        {
            var views = JsonSerializer.Deserialize<List<AiAssistQueueView>>(
                File.ReadAllText(viewsPath, Encoding.UTF8),
                JsonOptions
            ) ?? [];

            return views
                .Where(view => !string.IsNullOrWhiteSpace(view.Name))
                .OrderBy(view => view.Name, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch (JsonException)
        {
            return [];
        }
    }

    public AiAssistQueueView SaveAiAssistQueueView(string projectPath, AiAssistQueueView view)
    {
        var name = view.Name.Trim();
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new InvalidOperationException("AI Assist queue view name is required.");
        }

        var savedView = new AiAssistQueueView
        {
            Name = name,
            Filter = string.IsNullOrWhiteSpace(view.Filter) ? "All" : view.Filter,
            Search = view.Search.Trim(),
            Sort = string.IsNullOrWhiteSpace(view.Sort) ? "Newest" : view.Sort,
        };
        var views = LoadAiAssistQueueViews(projectPath).ToList();
        var existingIndex = views.FindIndex(existing =>
            string.Equals(existing.Name, savedView.Name, StringComparison.OrdinalIgnoreCase)
        );

        if (existingIndex >= 0)
        {
            views[existingIndex] = savedView;
        }
        else
        {
            views.Add(savedView);
        }

        views = views
            .OrderBy(existing => existing.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        Directory.CreateDirectory(projectPath);
        File.WriteAllText(
            GetAiAssistQueueViewsPath(projectPath),
            JsonSerializer.Serialize(views, new JsonSerializerOptions(JsonOptions)
            {
                WriteIndented = true
            }) + Environment.NewLine,
            encoding: Utf8NoBom
        );

        return savedView;
    }

    public IReadOnlyList<AiAssistRewriteBatch> LoadAiAssistRewriteBatches(
        string projectPath,
        int maxBatches = 20
    )
    {
        var batchesPath = GetAiAssistRewriteBatchesPath(projectPath);
        if (!File.Exists(batchesPath))
        {
            return [];
        }

        try
        {
            var batches = JsonSerializer.Deserialize<List<AiAssistRewriteBatch>>(
                File.ReadAllText(batchesPath, Encoding.UTF8),
                JsonOptions
            ) ?? [];

            return batches
                .Where(batch => !string.IsNullOrWhiteSpace(batch.BatchId))
                .OrderByDescending(batch => batch.CreatedAt)
                .Take(maxBatches)
                .ToList();
        }
        catch (JsonException)
        {
            return [];
        }
    }

    public AiAssistRewriteBatch SaveAiAssistRewriteBatch(
        string projectPath,
        AiAssistRewriteBatch batch
    )
    {
        if (string.IsNullOrWhiteSpace(batch.SourceDraft))
        {
            throw new InvalidOperationException("AI Assist rewrite batch source draft is required.");
        }

        if (string.IsNullOrWhiteSpace(batch.Instruction))
        {
            throw new InvalidOperationException("AI Assist rewrite batch instruction is required.");
        }

        var savedBatch = new AiAssistRewriteBatch
        {
            BatchId = string.IsNullOrWhiteSpace(batch.BatchId)
                ? Guid.NewGuid().ToString("N")
                : batch.BatchId,
            CreatedAt = batch.CreatedAt == default ? DateTime.UtcNow : batch.CreatedAt,
            SchemaId = batch.SchemaId,
            Action = string.IsNullOrWhiteSpace(batch.Action) ? "rewrite-output" : batch.Action,
            RowNumbers = batch.RowNumbers
                .Where(rowNumber => rowNumber > 0)
                .Distinct()
                .Order()
                .ToList(),
            IssueCount = batch.IssueCount,
            IssueSummary = batch.IssueSummary.Trim(),
            SourceDraft = batch.SourceDraft.TrimEnd(),
            Instruction = batch.Instruction.Trim(),
        };

        var batches = LoadAiAssistRewriteBatches(projectPath, maxBatches: 100).ToList();
        var existingIndex = batches.FindIndex(existing =>
            string.Equals(existing.BatchId, savedBatch.BatchId, StringComparison.Ordinal)
        );
        if (existingIndex >= 0)
        {
            batches[existingIndex] = savedBatch;
        }
        else
        {
            batches.Add(savedBatch);
        }

        batches = batches
            .OrderByDescending(existing => existing.CreatedAt)
            .Take(20)
            .ToList();

        Directory.CreateDirectory(projectPath);
        File.WriteAllText(
            GetAiAssistRewriteBatchesPath(projectPath),
            JsonSerializer.Serialize(batches, new JsonSerializerOptions(JsonOptions)
            {
                WriteIndented = true
            }) + Environment.NewLine,
            encoding: Utf8NoBom
        );

        return savedBatch;
    }

    public IReadOnlyList<ReviewedFixRecord> LoadReviewedFixes(
        string projectPath,
        int maxRecords = 200
    )
    {
        var fixesPath = GetReviewedFixesPath(projectPath);
        if (!File.Exists(fixesPath))
        {
            return [];
        }

        try
        {
            var records = JsonSerializer.Deserialize<List<ReviewedFixRecord>>(
                File.ReadAllText(fixesPath, Encoding.UTF8),
                JsonOptions
            ) ?? [];

            return records
                .Where(record => !string.IsNullOrWhiteSpace(record.FixId)
                    && !string.IsNullOrWhiteSpace(record.ExampleId))
                .OrderByDescending(record => record.CreatedAt)
                .Take(maxRecords)
                .ToList();
        }
        catch (JsonException)
        {
            return [];
        }
    }

    public ReviewedFixRecord RecordReviewedFix(string projectPath, ReviewedFixRecord fix)
    {
        if (string.IsNullOrWhiteSpace(fix.ExampleId))
        {
            throw new InvalidOperationException("Reviewed fix requires an evaluation example id.");
        }

        var records = LoadReviewedFixes(projectPath, maxRecords: 500).ToList();
        var nextVersion = records
            .Where(record => string.Equals(record.ExampleId, fix.ExampleId, StringComparison.Ordinal))
            .Select(record => record.Version)
            .DefaultIfEmpty(0)
            .Max() + 1;

        var savedFix = new ReviewedFixRecord
        {
            FixId = string.IsNullOrWhiteSpace(fix.FixId)
                ? Guid.NewGuid().ToString("N")
                : fix.FixId,
            ExampleId = fix.ExampleId,
            RowNumber = fix.RowNumber,
            SchemaId = fix.SchemaId,
            Version = nextVersion,
            Status = ReviewedFixRecord.StatusEdited,
            OriginalScore = fix.OriginalScore,
            LatestScore = null,
            FailureReason = fix.FailureReason.Trim(),
            SourceReport = fix.SourceReport.Trim(),
            CreatedAt = fix.CreatedAt == default ? DateTime.UtcNow : fix.CreatedAt,
            UpdatedAt = DateTime.UtcNow,
        };

        records.Add(savedFix);
        PersistReviewedFixes(projectPath, records);
        return savedFix;
    }

    /// <summary>
    /// Reconciles open reviewed fixes against a fresh set of evaluation results.
    /// Only the latest version per example is updated: a passing re-test marks the
    /// fix resolved, a failing re-test marks it still-failing. Older versions and
    /// examples absent from the run are left untouched.
    /// </summary>
    public IReadOnlyList<ReviewedFixRecord> ReconcileReviewedFixes(
        string projectPath,
        IReadOnlyList<EvaluationExampleResult> results
    )
    {
        var records = LoadReviewedFixes(projectPath, maxRecords: 500).ToList();
        if (records.Count == 0)
        {
            return records;
        }

        var latestResultByExample = results
            .GroupBy(result => result.ExampleId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Last(), StringComparer.Ordinal);

        var latestVersionByExample = records
            .GroupBy(record => record.ExampleId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Max(record => record.Version), StringComparer.Ordinal);

        var changed = false;
        foreach (var record in records)
        {
            var isLatestVersion =
                latestVersionByExample.TryGetValue(record.ExampleId, out var latestVersion)
                && latestVersion == record.Version;

            if (!isLatestVersion || record.Status != ReviewedFixRecord.StatusEdited)
            {
                continue;
            }

            if (!latestResultByExample.TryGetValue(record.ExampleId, out var result))
            {
                continue;
            }

            record.Status = result.Passed
                ? ReviewedFixRecord.StatusResolved
                : ReviewedFixRecord.StatusStillFailing;
            record.LatestScore = result.Score;
            record.UpdatedAt = DateTime.UtcNow;
            changed = true;
        }

        if (changed)
        {
            PersistReviewedFixes(projectPath, records);
        }

        return LoadReviewedFixes(projectPath);
    }

    private void PersistReviewedFixes(string projectPath, List<ReviewedFixRecord> records)
    {
        var trimmed = records
            .OrderByDescending(record => record.CreatedAt)
            .Take(200)
            .ToList();

        Directory.CreateDirectory(projectPath);
        File.WriteAllText(
            GetReviewedFixesPath(projectPath),
            JsonSerializer.Serialize(trimmed, new JsonSerializerOptions(JsonOptions)
            {
                WriteIndented = true
            }) + Environment.NewLine,
            encoding: Utf8NoBom
        );
    }

    public IReadOnlyList<EvaluationFailureFilter> LoadEvaluationFailureFilters(string projectPath)
    {
        var filtersPath = GetEvaluationFailureFiltersPath(projectPath);
        if (!File.Exists(filtersPath))
        {
            return [];
        }

        try
        {
            var filters = JsonSerializer.Deserialize<List<EvaluationFailureFilter>>(
                File.ReadAllText(filtersPath, Encoding.UTF8),
                JsonOptions
            ) ?? [];

            return filters
                .Where(filter => !string.IsNullOrWhiteSpace(filter.Name))
                .OrderBy(filter => filter.Name, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch (JsonException)
        {
            return [];
        }
    }

    public EvaluationFailureFilter SaveEvaluationFailureFilter(
        string projectPath,
        EvaluationFailureFilter filter
    )
    {
        var name = filter.Name.Trim();
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new InvalidOperationException("Evaluation failure filter name is required.");
        }

        var savedFilter = new EvaluationFailureFilter
        {
            Name = name,
            Status = string.IsNullOrWhiteSpace(filter.Status) ? "All" : filter.Status,
            Tag = string.IsNullOrWhiteSpace(filter.Tag) ? "All" : filter.Tag,
            FailureReason = string.IsNullOrWhiteSpace(filter.FailureReason) ? "All" : filter.FailureReason,
            ScoreBand = string.IsNullOrWhiteSpace(filter.ScoreBand) ? "All" : filter.ScoreBand,
        };

        var filters = LoadEvaluationFailureFilters(projectPath).ToList();
        var existingIndex = filters.FindIndex(existing =>
            string.Equals(existing.Name, savedFilter.Name, StringComparison.OrdinalIgnoreCase)
        );

        if (existingIndex >= 0)
        {
            filters[existingIndex] = savedFilter;
        }
        else
        {
            filters.Add(savedFilter);
        }

        filters = filters
            .OrderBy(existing => existing.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();

        Directory.CreateDirectory(projectPath);
        File.WriteAllText(
            GetEvaluationFailureFiltersPath(projectPath),
            JsonSerializer.Serialize(filters, new JsonSerializerOptions(JsonOptions)
            {
                WriteIndented = true
            }) + Environment.NewLine,
            encoding: Utf8NoBom
        );

        return savedFilter;
    }

    public AiAssistReviewQueueItem SaveAiAssistReviewQueueItem(
        string projectPath,
        string sourceDraft,
        AiAssistRunResult result
    )
    {
        var item = AiAssistReviewQueueItem.FromRunResult(sourceDraft, result);
        var queuePath = GetAiAssistReviewQueuePath(projectPath);
        Directory.CreateDirectory(projectPath);
        File.AppendAllText(
            queuePath,
            JsonSerializer.Serialize(item, JsonOptions) + Environment.NewLine,
            encoding: Utf8NoBom
        );
        return item;
    }

    public AiAssistReviewQueueItem UpdateAiAssistReviewState(
        string projectPath,
        string reviewId,
        string reviewState
    )
    {
        if (reviewState is not ("accepted" or "rejected" or "review_required"))
        {
            throw new InvalidOperationException("AI Assist review state must be accepted, rejected, or review_required.");
        }

        var queuePath = GetAiAssistReviewQueuePath(projectPath);
        if (!File.Exists(queuePath))
        {
            throw new FileNotFoundException("AI Assist review queue was not found.", queuePath);
        }

        var items = File.ReadLines(queuePath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => JsonSerializer.Deserialize<AiAssistReviewQueueItem>(line, JsonOptions))
            .Where(item => item is not null)
            .Cast<AiAssistReviewQueueItem>()
            .ToList();

        var target = items.FirstOrDefault(item => item.ReviewId == reviewId)
            ?? throw new InvalidOperationException($"AI Assist review was not found: {reviewId}");
        target.ReviewState = reviewState;
        target.DecidedAt = reviewState == "review_required" ? null : DateTime.UtcNow;

        var lines = items.Select(item => JsonSerializer.Serialize(item, JsonOptions));
        File.WriteAllText(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
        return target;
    }

    public int UpdateAiAssistReviewStates(
        string projectPath,
        IReadOnlyCollection<string> reviewIds,
        string reviewState
    )
    {
        if (reviewState is not ("accepted" or "rejected" or "review_required"))
        {
            throw new InvalidOperationException("AI Assist review state must be accepted, rejected, or review_required.");
        }

        var reviewIdSet = reviewIds
            .Where(reviewId => !string.IsNullOrWhiteSpace(reviewId))
            .ToHashSet(StringComparer.Ordinal);
        if (reviewIdSet.Count == 0)
        {
            return 0;
        }

        var queuePath = GetAiAssistReviewQueuePath(projectPath);
        if (!File.Exists(queuePath))
        {
            throw new FileNotFoundException("AI Assist review queue was not found.", queuePath);
        }

        var items = File.ReadLines(queuePath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => JsonSerializer.Deserialize<AiAssistReviewQueueItem>(line, JsonOptions))
            .Where(item => item is not null)
            .Cast<AiAssistReviewQueueItem>()
            .ToList();

        var updatedCount = 0;
        foreach (var item in items.Where(item => reviewIdSet.Contains(item.ReviewId)))
        {
            item.ReviewState = reviewState;
            item.DecidedAt = reviewState == "review_required" ? null : DateTime.UtcNow;
            updatedCount++;
        }

        if (updatedCount == 0)
        {
            throw new InvalidOperationException("No matching AI Assist reviews were found.");
        }

        var lines = items.Select(item => JsonSerializer.Serialize(item, JsonOptions));
        File.WriteAllText(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
        return updatedCount;
    }

    public int UpdateAiAssistReviewStates(
        string projectPath,
        IReadOnlyDictionary<string, string> reviewStatesById
    )
    {
        var requestedStates = reviewStatesById
            .Where(pair => !string.IsNullOrWhiteSpace(pair.Key))
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        if (requestedStates.Count == 0)
        {
            return 0;
        }

        foreach (var reviewState in requestedStates.Values)
        {
            if (reviewState is not ("accepted" or "rejected" or "review_required"))
            {
                throw new InvalidOperationException("AI Assist review state must be accepted, rejected, or review_required.");
            }
        }

        var queuePath = GetAiAssistReviewQueuePath(projectPath);
        if (!File.Exists(queuePath))
        {
            throw new FileNotFoundException("AI Assist review queue was not found.", queuePath);
        }

        var items = File.ReadLines(queuePath, Encoding.UTF8)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => JsonSerializer.Deserialize<AiAssistReviewQueueItem>(line, JsonOptions))
            .Where(item => item is not null)
            .Cast<AiAssistReviewQueueItem>()
            .ToList();

        var updatedCount = 0;
        foreach (var item in items.Where(item => requestedStates.ContainsKey(item.ReviewId)))
        {
            var reviewState = requestedStates[item.ReviewId];
            item.ReviewState = reviewState;
            item.DecidedAt = reviewState == "review_required" ? null : DateTime.UtcNow;
            updatedCount++;
        }

        if (updatedCount == 0)
        {
            throw new InvalidOperationException("No matching AI Assist reviews were found.");
        }

        var lines = items.Select(item => JsonSerializer.Serialize(item, JsonOptions));
        File.WriteAllText(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
        return updatedCount;
    }

    public DesktopSettings GetSettings()
    {
        return new DesktopSettings(
            _repositoryRoot,
            _engineDirectory,
            _pythonExecutable,
            ResolveProjectRoot(),
            ResolveExportRoot()
        );
    }

    public Task<string> ValidateAsync(string datasetPath, string schemaId)
    {
        return RunEngineCommandAsync("validate", datasetPath, schemaId);
    }

    public async Task<ImportPreviewReport> PreviewImportAsync(string importPath, string schemaId)
    {
        var output = await RunEngineCommandAsync("import-preview", importPath, schemaId);
        return JsonSerializer.Deserialize<ImportPreviewReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid import preview.");
    }

    public async Task<QualityReport> BuildQualityReportAsync(string projectPath)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var output = await RunEngineCommandAsync("quality", examplesPath);
        return JsonSerializer.Deserialize<QualityReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid quality report.");
    }

    public async Task<ValidationReport> ValidateDraftAsync(string draftText, string schemaId)
    {
        var tempPath = WriteDraftToTempJsonl(draftText);

        try
        {
            var result = await RunEngineProcessAsync(_engineDirectory, "validate", tempPath, schemaId);
            var payload = string.IsNullOrWhiteSpace(result.Output) ? result.Error : result.Output;

            if (string.IsNullOrWhiteSpace(payload))
            {
                throw new InvalidOperationException("The Python engine returned an empty validation report.");
            }

            return JsonSerializer.Deserialize<ValidationReport>(payload, JsonOptions)
                ?? throw new InvalidOperationException("The Python engine returned an invalid validation report.");
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    public int AppendDraftToProjectExamples(string projectPath, string draftText)
    {
        var jsonl = NormalizeDraftToJsonl(draftText);
        var rowCount = jsonl.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries).Length;

        if (rowCount == 0)
        {
            return 0;
        }

        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        Directory.CreateDirectory(projectPath);
        File.AppendAllText(examplesPath, jsonl, encoding: Utf8NoBom);
        return rowCount;
    }

    public ImportCommitResult CommitJsonlImportToProjectExamples(
        string projectPath,
        string importPath,
        ImportPreviewReport report
    )
    {
        var failedRowNumbers = report.FailedRows
            .Select(row => row.RowNumber)
            .ToHashSet();
        var rows = new List<string>();
        var rowNumber = 0;
        foreach (var rawLine in File.ReadLines(importPath, Encoding.UTF8))
        {
            rowNumber++;
            if (string.IsNullOrWhiteSpace(rawLine))
            {
                continue;
            }

            if (failedRowNumbers.Contains(rowNumber))
            {
                continue;
            }

            using var document = JsonDocument.Parse(rawLine);
            rows.Add(JsonSerializer.Serialize(document.RootElement));
        }

        var quarantinePath = report.FailedRows.Count == 0
            ? null
            : WriteImportQuarantine(projectPath, importPath, report);

        if (rows.Count > 0)
        {
            var examplesPath = Path.Combine(projectPath, "examples.jsonl");
            Directory.CreateDirectory(projectPath);
            var jsonl = string.Join(Environment.NewLine, rows) + Environment.NewLine;
            File.AppendAllText(examplesPath, jsonl, encoding: Utf8NoBom);
        }

        return new ImportCommitResult(
            rows.Count,
            report.FailedRows.Count,
            quarantinePath
        );
    }

    public async Task<string> ExportProjectExamplesAsync(string projectPath, string schemaId)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputPath = Path.Combine(ResolveExportRoot(), projectId, "export.jsonl");
        await RunEngineCommandAsync("export", examplesPath, outputPath, schemaId);
        return outputPath;
    }

    public string ExportPreferenceRanking(
        string projectPath,
        IReadOnlyList<PreferenceReviewItem> items
    )
    {
        if (items.Count == 0)
        {
            throw new InvalidOperationException("No visible preference ranking items are available to export.");
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputDirectory = Path.Combine(ResolveExportRoot(), projectId, "preference_review");
        Directory.CreateDirectory(outputDirectory);

        var timestamp = DateTime.Now.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
        var outputPath = Path.Combine(outputDirectory, $"{timestamp}_preference_ranking.json");
        var payload = new
        {
            project_id = projectId,
            exported_at = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            purpose = "DPO and reward-model preference-pair review",
            item_count = items.Count,
            items = items.Select(item => new
            {
                row_number = item.RowNumber,
                prompt = item.Prompt,
                chosen = item.Chosen,
                rejected = item.Rejected,
                reason = item.Reason,
                contrast = item.Contrast,
                token_overlap = Math.Round(item.TokenOverlap, 4),
                character_delta = item.CharacterDelta,
            }),
        };

        File.WriteAllText(
            outputPath,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions(JsonOptions)
            {
                WriteIndented = true
            }) + Environment.NewLine,
            encoding: Utf8NoBom
        );
        return outputPath;
    }

    public async Task<SplitReport> GenerateProjectSplitsAsync(
        string projectPath,
        string schemaId,
        double trainRatio,
        double validationRatio,
        int seed
    )
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputDirectory = Path.Combine(ResolveExportRoot(), projectId, "splits");
        var output = await RunEngineCommandAsync(
            "split",
            examplesPath,
            outputDirectory,
            schemaId,
            "--train-ratio",
            trainRatio.ToString(CultureInfo.InvariantCulture),
            "--validation-ratio",
            validationRatio.ToString(CultureInfo.InvariantCulture),
            "--seed",
            seed.ToString(CultureInfo.InvariantCulture)
        );
        return JsonSerializer.Deserialize<SplitReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid split report.");
    }

    public async Task<EvaluationRunResult> RunEvaluationAsync(
        string projectPath,
        string schemaId,
        string backend,
        string model,
        string? baseUrl,
        int? limit,
        double scoreThreshold,
        int timeoutSeconds
    )
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var reportDirectory = Path.Combine(ResolveExportRoot(), projectId, "evaluation");
        var timestamp = DateTime.Now.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
        var reportPath = Path.Combine(reportDirectory, $"{timestamp}_evaluation_report.json");

        var arguments = new List<string>
        {
            "eval-run",
            examplesPath,
            schemaId,
            "--model",
            model,
            "--backend",
            backend,
            "--output-path",
            reportPath,
            "--score-threshold",
            scoreThreshold.ToString(CultureInfo.InvariantCulture),
            "--timeout-seconds",
            timeoutSeconds.ToString(CultureInfo.InvariantCulture),
        };

        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            arguments.Add("--base-url");
            arguments.Add(baseUrl);
        }

        if (limit is not null)
        {
            arguments.Add("--limit");
            arguments.Add(limit.Value.ToString(CultureInfo.InvariantCulture));
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        var reportJson = File.Exists(reportPath)
            ? File.ReadAllText(reportPath, Encoding.UTF8)
            : output;
        var report = JsonSerializer.Deserialize<EvaluationReport>(reportJson, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid evaluation report.");

        return new EvaluationRunResult(report, reportPath, reportJson);
    }

    public IReadOnlyList<EvaluationReportHistoryItem> LoadEvaluationReportHistory(
        string projectPath,
        int maxReports = 20
    )
    {
        var projectId = new DirectoryInfo(projectPath).Name;
        var reportDirectory = Path.Combine(ResolveExportRoot(), projectId, "evaluation");
        if (!Directory.Exists(reportDirectory))
        {
            return [];
        }

        var reports = new List<EvaluationReportHistoryItem>();
        foreach (var path in Directory
            .EnumerateFiles(reportDirectory, "*_evaluation_report.json")
            .OrderByDescending(File.GetLastWriteTimeUtc))
        {
            try
            {
                var reportJson = File.ReadAllText(path, Encoding.UTF8);
                var report = JsonSerializer.Deserialize<EvaluationReport>(reportJson, JsonOptions);
                if (report is not null)
                {
                    reports.Add(new EvaluationReportHistoryItem(
                        report,
                        path,
                        reportJson,
                        File.GetLastWriteTime(path)
                    ));
                }
            }
            catch (IOException)
            {
                continue;
            }
            catch (JsonException)
            {
                continue;
            }
        }

        return reports.Take(maxReports).ToList();
    }

    public EvaluationReportHistoryItem SaveEvaluationManualReview(
        EvaluationReportHistoryItem reportItem,
        string exampleId,
        double? manualScore,
        string? manualNotes
    )
    {
        var target = reportItem.Report.Results.FirstOrDefault(result => result.ExampleId == exampleId)
            ?? throw new InvalidOperationException($"Evaluation example was not found: {exampleId}");

        target.ManualScore = manualScore;
        target.ManualNotes = string.IsNullOrWhiteSpace(manualNotes) ? null : manualNotes.Trim();

        var manualScores = reportItem.Report.Results
            .Where(result => result.ManualScore is not null)
            .Select(result => result.ManualScore!.Value)
            .ToList();

        var root = JsonNode.Parse(reportItem.ReportJson)?.AsObject()
            ?? throw new InvalidOperationException("Evaluation report is not a JSON object.");
        root["manually_scored_examples"] = manualScores.Count;
        root["average_manual_score"] = manualScores.Count == 0
            ? null
            : Math.Round(manualScores.Average(), 2);

        var results = root["results"]?.AsArray()
            ?? throw new InvalidOperationException("Evaluation report does not contain a results list.");
        foreach (var node in results)
        {
            if (node is not JsonObject resultObject
                || !string.Equals(
                    resultObject["example_id"]?.GetValue<string>(),
                    exampleId,
                    StringComparison.Ordinal
                ))
            {
                continue;
            }

            resultObject["manual_score"] = manualScore;
            resultObject["manual_notes"] = target.ManualNotes;
            break;
        }

        var updatedJson = root.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(reportItem.ReportPath, updatedJson + Environment.NewLine, encoding: Utf8NoBom);

        var updatedReport = JsonSerializer.Deserialize<EvaluationReport>(updatedJson, JsonOptions)
            ?? throw new InvalidOperationException("The saved evaluation report could not be reloaded.");
        return new EvaluationReportHistoryItem(
            updatedReport,
            reportItem.ReportPath,
            updatedJson,
            File.GetLastWriteTime(reportItem.ReportPath)
        );
    }

    public async Task<AiAssistRunResult> RunAiAssistAsync(
        string draftText,
        string schemaId,
        string action,
        string backend,
        string model,
        string? baseUrl,
        int timeoutSeconds,
        string? instruction
    )
    {
        var tempPath = WriteDraftToTempJsonl(draftText);

        try
        {
            var arguments = new List<string>
            {
                "ai-assist",
                tempPath,
                schemaId,
                "--action",
                action,
                "--model",
                model,
                "--backend",
                backend,
                "--timeout-seconds",
                timeoutSeconds.ToString(CultureInfo.InvariantCulture),
            };

            if (!string.IsNullOrWhiteSpace(baseUrl))
            {
                arguments.Add("--base-url");
                arguments.Add(baseUrl);
            }

            if (!string.IsNullOrWhiteSpace(instruction))
            {
                arguments.Add("--instruction");
                arguments.Add(instruction);
            }

            var output = await RunEngineCommandAsync(arguments.ToArray());
            return JsonSerializer.Deserialize<AiAssistRunResult>(output, JsonOptions)
                ?? throw new InvalidOperationException("The Python engine returned an invalid AI Assist result.");
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    public async Task<BackendHealthReport> CheckBackendHealthAsync(
        string backend,
        string model,
        string? baseUrl,
        int timeoutSeconds
    )
    {
        var arguments = new List<string>
        {
            "backend-health",
            "--model",
            model,
            "--backend",
            backend,
            "--timeout-seconds",
            timeoutSeconds.ToString(CultureInfo.InvariantCulture),
        };

        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            arguments.Add("--base-url");
            arguments.Add(baseUrl);
        }

        var result = await RunEngineProcessAsync(_engineDirectory, arguments.ToArray());
        var payload = string.IsNullOrWhiteSpace(result.Output) ? result.Error : result.Output;
        if (string.IsNullOrWhiteSpace(payload))
        {
            throw new InvalidOperationException("The Python engine returned an empty backend health report.");
        }

        if (!payload.TrimStart().StartsWith('{'))
        {
            throw new InvalidOperationException(payload);
        }

        var report = JsonSerializer.Deserialize<BackendHealthReport>(payload, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid backend health report.");

        if (result.ExitCode != 0 && report.Error is null)
        {
            throw new InvalidOperationException(payload);
        }

        return report;
    }

    public async Task<BackendModelListReport> ListBackendModelsAsync(
        string backend,
        string? baseUrl,
        int timeoutSeconds
    )
    {
        var arguments = new List<string>
        {
            "model-list",
            "--backend",
            backend,
            "--timeout-seconds",
            timeoutSeconds.ToString(CultureInfo.InvariantCulture),
        };

        if (!string.IsNullOrWhiteSpace(baseUrl))
        {
            arguments.Add("--base-url");
            arguments.Add(baseUrl);
        }

        var result = await RunEngineProcessAsync(_engineDirectory, arguments.ToArray());
        var payload = string.IsNullOrWhiteSpace(result.Output) ? result.Error : result.Output;
        if (string.IsNullOrWhiteSpace(payload))
        {
            throw new InvalidOperationException("The Python engine returned an empty model list report.");
        }

        if (!payload.TrimStart().StartsWith('{'))
        {
            throw new InvalidOperationException(payload);
        }

        var report = JsonSerializer.Deserialize<BackendModelListReport>(payload, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid model list report.");

        if (result.ExitCode != 0 && report.Error is null)
        {
            throw new InvalidOperationException(payload);
        }

        return report;
    }

    public async Task<TrainingConfigExportResult> GenerateTrainingConfigAsync(
        string projectPath,
        string schemaId,
        string target,
        string baseModel,
        string datasetFormat,
        int sequenceLen,
        int loraR,
        int loraAlpha,
        int microBatchSize,
        int gradientAccumulationSteps,
        double learningRate
    )
    {
        var projectId = new DirectoryInfo(projectPath).Name;
        var projectExportRoot = Path.Combine(ResolveExportRoot(), projectId);
        var splitDirectory = Path.Combine(projectExportRoot, "splits");
        var trainSplitPath = Path.Combine(splitDirectory, "train.jsonl");
        var validationSplitPath = Path.Combine(splitDirectory, "validation.jsonl");
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");

        var datasetPath = File.Exists(trainSplitPath) ? trainSplitPath : examplesPath;
        if (!File.Exists(datasetPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var evalDatasetPath = File.Exists(validationSplitPath) ? validationSplitPath : null;
        var trainingDirectory = Path.Combine(projectExportRoot, "training");
        var timestamp = DateTime.Now.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
        var outputPath = Path.Combine(
            trainingDirectory,
            $"{timestamp}_{SanitizeFileNamePart(target)}_config{GetTrainingConfigExtension(target)}"
        );

        var arguments = new List<string>
        {
            "training-config",
            datasetPath,
            schemaId,
            "--output-path",
            outputPath,
            "--base-model",
            baseModel,
            "--target",
            target,
            "--format",
            datasetFormat,
            "--sequence-len",
            sequenceLen.ToString(CultureInfo.InvariantCulture),
            "--lora-r",
            loraR.ToString(CultureInfo.InvariantCulture),
            "--lora-alpha",
            loraAlpha.ToString(CultureInfo.InvariantCulture),
            "--micro-batch-size",
            microBatchSize.ToString(CultureInfo.InvariantCulture),
            "--gradient-accumulation-steps",
            gradientAccumulationSteps.ToString(CultureInfo.InvariantCulture),
            "--learning-rate",
            learningRate.ToString(CultureInfo.InvariantCulture),
        };

        if (!string.IsNullOrWhiteSpace(evalDatasetPath))
        {
            arguments.Add("--eval-dataset-path");
            arguments.Add(evalDatasetPath);
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<TrainingConfigExportResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid training config result.");
    }

    public async Task<DatasetCardResult> GenerateDatasetCardAsync(
        string projectPath,
        string schemaId
    )
    {
        var projectFile = Path.Combine(projectPath, "project.json");
        if (!File.Exists(projectFile))
        {
            throw new FileNotFoundException("Project metadata file was not found.", projectFile);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var exportDirectory = Path.Combine(ResolveExportRoot(), projectId);
        var outputPath = Path.Combine(exportDirectory, "dataset_card.md");

        var output = await RunEngineCommandAsync(
            "dataset-card",
            projectPath,
            "--schema",
            schemaId,
            "--export-dir",
            exportDirectory,
            "--output-path",
            outputPath
        );

        return JsonSerializer.Deserialize<DatasetCardResult>(output, JsonOptions)
            ?? throw new InvalidOperationException(
                "The Python engine returned an invalid dataset card result."
            );
    }

    public SplitSettings LoadProjectSplitSettings(string projectPath)
    {
        var project = LoadProjectFromPath(projectPath);
        return project?.Project.SplitSettings ?? SplitSettings.Default;
    }

    public void SaveProjectSplitSettings(string projectPath, SplitSettings settings)
    {
        var projectFile = Path.Combine(projectPath, "project.json");
        if (!File.Exists(projectFile))
        {
            throw new FileNotFoundException("Project metadata file was not found.", projectFile);
        }

        var root = JsonNode.Parse(File.ReadAllText(projectFile, Encoding.UTF8))?.AsObject()
            ?? throw new InvalidOperationException("Project metadata is not a JSON object.");
        root["split_settings"] = JsonSerializer.SerializeToNode(settings, JsonOptions);
        root["updated_at"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);

        File.WriteAllText(
            projectFile,
            root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
            encoding: Utf8NoBom
        );
    }

    public LabBackendSettings LoadProjectLabSettings(string projectPath)
    {
        var project = LoadProjectFromPath(projectPath);
        return project?.Project.LabSettings ?? LabBackendSettings.Default;
    }

    public void SaveProjectLabSettings(string projectPath, LabBackendSettings settings)
    {
        var projectFile = Path.Combine(projectPath, "project.json");
        if (!File.Exists(projectFile))
        {
            throw new FileNotFoundException("Project metadata file was not found.", projectFile);
        }

        var root = JsonNode.Parse(File.ReadAllText(projectFile, Encoding.UTF8))?.AsObject()
            ?? throw new InvalidOperationException("Project metadata is not a JSON object.");
        root["lab_settings"] = JsonSerializer.SerializeToNode(settings, JsonOptions);
        root["updated_at"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);

        File.WriteAllText(
            projectFile,
            root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
            encoding: Utf8NoBom
        );
    }

    public async Task<string> ValidateAsync(string engineDirectory, string datasetPath, string schemaId)
    {
        return await RunEngineCommandInDirectoryAsync(engineDirectory, "validate", datasetPath, schemaId);
    }

    private static DatasetProjectListItem? LoadProjectFromPath(string projectPath)
    {
        var projectFile = Path.Combine(projectPath, "project.json");
        if (!File.Exists(projectFile))
        {
            return null;
        }

        try
        {
            var json = File.ReadAllText(projectFile, Encoding.UTF8);
            var project = JsonSerializer.Deserialize<DatasetProject>(json, JsonOptions);
            return project is null ? null : new DatasetProjectListItem(project, projectPath);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static QualityHistoryEntry? TryDeserializeQualityHistoryEntry(string rawLine)
    {
        try
        {
            return JsonSerializer.Deserialize<QualityHistoryEntry>(rawLine, JsonOptions);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static AiAssistReviewQueueItem? TryDeserializeAiAssistReviewQueueItem(string rawLine)
    {
        try
        {
            return JsonSerializer.Deserialize<AiAssistReviewQueueItem>(rawLine, JsonOptions);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string GetAiAssistReviewQueuePath(string projectPath)
    {
        return Path.Combine(projectPath, "ai_assist_reviews.jsonl");
    }

    private static string GetAiAssistQueueViewsPath(string projectPath)
    {
        return Path.Combine(projectPath, "ai_assist_queue_views.json");
    }

    private static string GetAiAssistRewriteBatchesPath(string projectPath)
    {
        return Path.Combine(projectPath, "ai_assist_rewrite_batches.json");
    }

    private static string GetReviewedFixesPath(string projectPath)
    {
        return Path.Combine(projectPath, "reviewed_fixes.json");
    }

    private static string GetEvaluationFailureFiltersPath(string projectPath)
    {
        return Path.Combine(projectPath, "evaluation_failure_filters.json");
    }

    private Task<string> RunEngineCommandAsync(params string[] arguments)
    {
        return RunEngineCommandInDirectoryAsync(_engineDirectory, arguments);
    }

    private async Task<string> RunEngineCommandInDirectoryAsync(
        string engineDirectory,
        params string[] arguments
    )
    {
        var result = await RunEngineProcessAsync(engineDirectory, arguments);

        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(
                string.IsNullOrWhiteSpace(result.Error) ? result.Output : result.Error
            );
        }

        return result.Output;
    }

    private async Task<EngineProcessResult> RunEngineProcessAsync(
        string engineDirectory,
        params string[] arguments
    )
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExecutable,
            WorkingDirectory = engineDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add("corpus_studio.cli");

        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        foreach (var (key, value) in _localEnvironment)
        {
            if (!startInfo.Environment.TryGetValue(key, out var existingValue)
                || string.IsNullOrWhiteSpace(existingValue))
            {
                startInfo.Environment[key] = value;
            }
        }

        using var process = Process.Start(startInfo)
            ?? throw new InvalidOperationException("Failed to start Python engine process.");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        return new EngineProcessResult(process.ExitCode, output, error);
    }

    private static string WriteDraftToTempJsonl(string draftText)
    {
        var directory = Path.Combine(Path.GetTempPath(), "CorpusStudio");
        Directory.CreateDirectory(directory);

        var path = Path.Combine(directory, $"{Guid.NewGuid():N}.jsonl");
        File.WriteAllText(path, NormalizeDraftToJsonl(draftText), encoding: Utf8NoBom);
        return path;
    }

    private static string NormalizeDraftToJsonl(string draftText)
    {
        var trimmed = draftText.Trim();
        if (trimmed.Length == 0)
        {
            return string.Empty;
        }

        try
        {
            using var document = JsonDocument.Parse(trimmed);
            if (document.RootElement.ValueKind == JsonValueKind.Array)
            {
                var rows = document.RootElement.EnumerateArray()
                    .Select(element => JsonSerializer.Serialize(element));
                return string.Join(Environment.NewLine, rows) + Environment.NewLine;
            }

            return JsonSerializer.Serialize(document.RootElement) + Environment.NewLine;
        }
        catch (JsonException)
        {
            return draftText.EndsWith(Environment.NewLine, StringComparison.Ordinal)
                ? draftText
                : draftText + Environment.NewLine;
        }
    }

    private static string WriteImportQuarantine(
        string projectPath,
        string importPath,
        ImportPreviewReport report
    )
    {
        var rawLinesByRowNumber = File.ReadLines(importPath, Encoding.UTF8)
            .Select((line, index) => new { RowNumber = index + 1, Line = line.TrimEnd('\r', '\n') })
            .ToDictionary(row => row.RowNumber, row => row.Line);

        var quarantineDirectory = Path.Combine(projectPath, "import_quarantine");
        Directory.CreateDirectory(quarantineDirectory);

        var sourceName = SanitizeFileNamePart(Path.GetFileNameWithoutExtension(importPath));
        var timestamp = DateTime.Now.ToString("yyyyMMddHHmmss", CultureInfo.InvariantCulture);
        var quarantinePath = Path.Combine(
            quarantineDirectory,
            $"{timestamp}_{sourceName}_rejected.jsonl"
        );

        var entries = report.FailedRows.Select(failedRow =>
        {
            rawLinesByRowNumber.TryGetValue(failedRow.RowNumber, out var rawLine);
            return JsonSerializer.Serialize(
                new Dictionary<string, object?>
                {
                    ["source_path"] = importPath,
                    ["row_number"] = failedRow.RowNumber,
                    ["raw"] = rawLine ?? failedRow.RawPreview,
                    ["errors"] = failedRow.Errors,
                }
            );
        });

        File.WriteAllText(
            quarantinePath,
            string.Join(Environment.NewLine, entries) + Environment.NewLine,
            encoding: Utf8NoBom
        );
        return quarantinePath;
    }

    private static string SanitizeFileNamePart(string value)
    {
        var invalidCharacters = Path.GetInvalidFileNameChars().ToHashSet();
        var safeCharacters = value
            .Select(character => invalidCharacters.Contains(character) ? '_' : character)
            .ToArray();
        var safeValue = new string(safeCharacters).Trim();
        return string.IsNullOrWhiteSpace(safeValue) ? "import" : safeValue;
    }

    private static string GetTrainingConfigExtension(string target)
    {
        var normalized = target.Trim().Replace('-', '_').ToLowerInvariant();
        return normalized switch
        {
            "axolotl" or "axolotl_yaml" or "llama_factory" or "llamafactory" => ".yaml",
            "unsloth" or "unsloth_script" => ".py",
            _ => ".json",
        };
    }

    private static SavedExampleItem BuildSavedExampleItem(int rowNumber, string rawLine)
    {
        try
        {
            using var document = JsonDocument.Parse(rawLine);
            var json = JsonSerializer.Serialize(document.RootElement, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            return new SavedExampleItem(rowNumber, BuildPreview(document.RootElement), json);
        }
        catch (JsonException)
        {
            return new SavedExampleItem(rowNumber, "Invalid JSON row", rawLine);
        }
    }

    private static string BuildPreview(JsonElement row)
    {
        if (row.ValueKind != JsonValueKind.Object)
        {
            return Truncate(row.ToString());
        }

        foreach (var fieldName in new[] { "instruction", "text", "prompt", "output", "chosen" })
        {
            if (row.TryGetProperty(fieldName, out var value) && value.ValueKind == JsonValueKind.String)
            {
                return Truncate(value.GetString() ?? string.Empty);
            }
        }

        if (row.TryGetProperty("messages", out var messages) && messages.ValueKind == JsonValueKind.Array)
        {
            foreach (var message in messages.EnumerateArray())
            {
                if (message.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String)
                {
                    return Truncate(content.GetString() ?? string.Empty);
                }
            }
        }

        return "JSON example";
    }

    private static string Truncate(string value)
    {
        const int maxLength = 80;
        var normalized = string.Join(" ", value.Split(default(string[]), StringSplitOptions.RemoveEmptyEntries));
        return normalized.Length <= maxLength
            ? normalized
            : normalized[..(maxLength - 3)] + "...";
    }

    private static string FindRepositoryRoot()
    {
        foreach (var startPath in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var root = FindRepositoryRootFrom(startPath);
            if (root is not null)
            {
                return root;
            }
        }

        throw new DirectoryNotFoundException("Could not find the Corpus Studio repository root.");
    }

    private static string? FindRepositoryRootFrom(string startPath)
    {
        var directory = new DirectoryInfo(startPath);
        while (directory is not null)
        {
            var engineCli = Path.Combine(directory.FullName, "engine", "corpus_studio", "cli.py");
            if (File.Exists(engineCli))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    private static string ResolveEngineDirectory(
        string repositoryRoot,
        IReadOnlyDictionary<string, string> localEnvironment
    )
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_ENGINE_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            localEnvironment.TryGetValue("CORPUS_STUDIO_ENGINE_DIR", out configured);
        }

        var engineDirectory = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(repositoryRoot, "engine")
            : ResolvePath(repositoryRoot, configured);

        if (!Directory.Exists(engineDirectory))
        {
            throw new DirectoryNotFoundException($"Python engine directory not found: {engineDirectory}");
        }

        return engineDirectory;
    }

    private static string ResolvePythonExecutable(string engineDirectory)
    {
        var venvPython = Path.Combine(engineDirectory, ".venv", "Scripts", "python.exe");
        return File.Exists(venvPython) ? venvPython : "python";
    }

    private string ResolveProjectRoot()
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_DATA_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            _localEnvironment.TryGetValue("CORPUS_STUDIO_DATA_DIR", out configured);
        }

        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(_repositoryRoot, "data", "projects")
            : ResolvePath(_repositoryRoot, configured);
    }

    private string ResolveExportRoot()
    {
        var configured = Environment.GetEnvironmentVariable("CORPUS_STUDIO_EXPORT_DIR");
        if (string.IsNullOrWhiteSpace(configured))
        {
            _localEnvironment.TryGetValue("CORPUS_STUDIO_EXPORT_DIR", out configured);
        }

        return string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(_repositoryRoot, "exports")
            : ResolvePath(_repositoryRoot, configured);
    }

    private static IReadOnlyDictionary<string, string> LoadLocalEnvironment(string repositoryRoot)
    {
        var path = Path.Combine(repositoryRoot, ".env");
        if (!File.Exists(path))
        {
            return new Dictionary<string, string>();
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var rawLine in File.ReadLines(path))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            var separatorIndex = line.IndexOf('=');
            if (separatorIndex <= 0)
            {
                continue;
            }

            var key = line[..separatorIndex].Trim();
            var value = line[(separatorIndex + 1)..].Trim().Trim('"');
            values[key] = value;
        }

        return values;
    }

    private static string ResolvePath(string repositoryRoot, string path)
    {
        return Path.IsPathRooted(path) ? path : Path.GetFullPath(Path.Combine(repositoryRoot, path));
    }
}
