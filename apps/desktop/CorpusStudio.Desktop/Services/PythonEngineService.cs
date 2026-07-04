using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;

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

    private CancellationTokenSource? _currentRunCts;

    /// <summary>Raised after every engine CLI invocation completes (success, failure, or
    /// cancellation) so the Output / Logs panel can record engine activity (v1.2.7). Fires on a
    /// background thread — subscribers must marshal to the UI thread. Never carries secrets
    /// (API keys travel via the environment, not argv).</summary>
    public event EventHandler<EngineLogEntry>? CommandCompleted;

    /// <summary>Cancel the engine command currently running, if any, killing its
    /// process tree. Backs the desktop Cancel affordance for long local runs.</summary>
    public void CancelRunningEngineCommand() => _currentRunCts?.Cancel();

    /// <summary>Whether an engine command is currently running (and thus cancellable).</summary>
    public bool IsEngineCommandRunning => _currentRunCts is not null;

    /// <summary>Write text by writing a sibling temp file and atomically replacing the
    /// target, so a crash mid-write cannot truncate or corrupt a live data file.</summary>
    private static void WriteAllTextAtomic(string path, string content, Encoding? encoding = null)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var tempPath = path + ".tmp-" + Guid.NewGuid().ToString("N");
        if (encoding is null)
        {
            File.WriteAllText(tempPath, content);
        }
        else
        {
            File.WriteAllText(tempPath, content, encoding);
        }

        try
        {
            if (File.Exists(path))
            {
                File.Replace(tempPath, path, destinationBackupFileName: null);
            }
            else
            {
                File.Move(tempPath, path);
            }
        }
        catch
        {
            try
            {
                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }
            }
            catch
            {
                // Ignore cleanup failure; surface the original write error.
            }

            throw;
        }
    }

    /// <summary>Append text atomically: read the current file, append, and write the whole
    /// file back through the temp+File.Replace swap (<see cref="WriteAllTextAtomic"/>), so a
    /// crash mid-append cannot tear the last line of the live dataset. Rewrites the whole file —
    /// acceptable for these low-frequency, user-initiated dataset writes.</summary>
    private static void AppendAllTextAtomic(string path, string content, Encoding encoding)
    {
        var existing = File.Exists(path) ? File.ReadAllText(path, encoding) : string.Empty;
        WriteAllTextAtomic(path, existing + content, encoding);
    }

    /// <summary>A canonical, order-independent identity for a JSON row (object keys sorted,
    /// compact), so re-importing the same rows is idempotent. Mirrors the engine's
    /// exact_row_signature (json.dumps sort_keys) in spirit; used only for desktop-side dedupe,
    /// so it need only be internally consistent. Returns null for an unparseable line.</summary>
    private static string? CanonicalRowSignature(string jsonLine)
    {
        try
        {
            using var document = JsonDocument.Parse(jsonLine);
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                WriteCanonical(document.RootElement, writer);
            }
            return Encoding.UTF8.GetString(stream.ToArray());
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static void WriteCanonical(JsonElement element, Utf8JsonWriter writer)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                writer.WriteStartObject();
                foreach (var property in element.EnumerateObject().OrderBy(p => p.Name, StringComparer.Ordinal))
                {
                    writer.WritePropertyName(property.Name);
                    WriteCanonical(property.Value, writer);
                }
                writer.WriteEndObject();
                break;
            case JsonValueKind.Array:
                writer.WriteStartArray();
                foreach (var item in element.EnumerateArray())
                {
                    WriteCanonical(item, writer);
                }
                writer.WriteEndArray();
                break;
            default:
                element.WriteTo(writer);
                break;
        }
    }

    /// <summary>Remove a single quarantine entry after its row was repaired and saved, so a
    /// retried row doesn't orphan in quarantine. Rewrites the entry's *_rejected.jsonl atomically
    /// without the matching line, and deletes the file when it becomes empty.</summary>
    public void RemoveImportQuarantineItem(ImportQuarantineItem item)
    {
        if (string.IsNullOrEmpty(item.QuarantinePath) || !File.Exists(item.QuarantinePath))
        {
            return;
        }

        var kept = new List<string>();
        var removed = false;
        foreach (var rawLine in File.ReadLines(item.QuarantinePath, Encoding.UTF8))
        {
            if (string.IsNullOrWhiteSpace(rawLine))
            {
                continue;
            }
            if (!removed && MatchesQuarantineEntry(rawLine, item))
            {
                removed = true; // drop exactly one matching entry
                continue;
            }
            kept.Add(rawLine);
        }

        if (kept.Count == 0)
        {
            File.Delete(item.QuarantinePath);
        }
        else
        {
            WriteAllTextAtomic(
                item.QuarantinePath,
                string.Join(Environment.NewLine, kept) + Environment.NewLine,
                encoding: Utf8NoBom);
        }
    }

    private static bool MatchesQuarantineEntry(string rawLine, ImportQuarantineItem item)
    {
        try
        {
            var parsed = JsonSerializer.Deserialize<ImportQuarantineItem>(rawLine, JsonOptions);
            return parsed is not null
                && parsed.RowNumber == item.RowNumber
                && parsed.SourcePath == item.SourcePath
                && parsed.Raw == item.Raw;
        }
        catch (JsonException)
        {
            return false; // keep unparseable lines
        }
    }

    private string _repositoryRoot = string.Empty;
    private string _engineDirectory = string.Empty;
    private string _pythonExecutable = "python";
    private IReadOnlyDictionary<string, string> _localEnvironment = new Dictionary<string, string>();

    /// <summary>Whether the Python engine tree was located. When false, engine calls are
    /// refused with a clear message instead of crashing, and the desktop shows a setup
    /// screen. Never throws from the constructor (a missing engine is an expected,
    /// recoverable state for a distributed build).</summary>
    public bool IsEngineAvailable { get; private set; }

    /// <summary>Why the engine is unavailable (null when available).</summary>
    public string? EngineUnavailableReason { get; private set; }

    public PythonEngineService()
    {
        TryReinitialize();
    }

    /// <summary>Re-run the default engine resolution (repo-root walk + CORPUS_STUDIO_ENGINE_DIR).
    /// Sets <see cref="IsEngineAvailable"/> and returns the result. Used at startup and by the
    /// setup screen's Retry.</summary>
    public bool TryReinitialize()
    {
        try
        {
            _repositoryRoot = FindRepositoryRoot();
            _localEnvironment = LoadLocalEnvironment(_repositoryRoot);
            _engineDirectory = ResolveEngineDirectory(_repositoryRoot, _localEnvironment);
            _pythonExecutable = ResolvePythonExecutable(_engineDirectory);
            IsEngineAvailable = true;
            EngineUnavailableReason = null;
            return true;
        }
        catch (Exception ex)
        {
            IsEngineAvailable = false;
            EngineUnavailableReason = ex.Message;
            return false;
        }
    }

    /// <summary>Point the service at a user-picked folder — either the engine directory itself
    /// (contains <c>corpus_studio/cli.py</c>) or a repo root containing <c>engine/</c>. Flips
    /// <see cref="IsEngineAvailable"/> to true on success.</summary>
    public bool TryLocateEngine(string candidateDirectory)
    {
        if (string.IsNullOrWhiteSpace(candidateDirectory) || !Directory.Exists(candidateDirectory))
        {
            return false;
        }

        string? engineDir = null;
        if (File.Exists(Path.Combine(candidateDirectory, "corpus_studio", "cli.py")))
        {
            engineDir = candidateDirectory;
        }
        else if (File.Exists(Path.Combine(candidateDirectory, "engine", "corpus_studio", "cli.py")))
        {
            engineDir = Path.Combine(candidateDirectory, "engine");
        }

        if (engineDir is null)
        {
            return false;
        }

        _engineDirectory = engineDir;
        _repositoryRoot = Directory.GetParent(engineDir)?.FullName ?? engineDir;
        _localEnvironment = LoadLocalEnvironment(_repositoryRoot);
        _pythonExecutable = ResolvePythonExecutable(_engineDirectory);
        IsEngineAvailable = true;
        EngineUnavailableReason = null;
        return true;
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
        WriteAllTextAtomic(
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
        WriteAllTextAtomic(
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
        WriteAllTextAtomic(
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
        WriteAllTextAtomic(
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
        WriteAllTextAtomic(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
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
        WriteAllTextAtomic(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
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
        WriteAllTextAtomic(queuePath, string.Join(Environment.NewLine, lines) + Environment.NewLine, encoding: Utf8NoBom);
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

    /// <summary>Inspect a public Hugging Face dataset (configs/splits, columns, license).
    /// Read-only; no auth or upload.</summary>
    public async Task<HfDatasetInspection> HfInspectAsync(string datasetId)
    {
        var output = await RunEngineCommandAsync("hf-inspect", datasetId);
        return JsonSerializer.Deserialize<HfDatasetInspection>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid HF inspection.");
    }

    /// <summary>Fetch + map a public Hugging Face dataset into a STAGING JSONL file (never
    /// examples.jsonl — the caller runs the staging file through the normal import-preview
    /// flow, so the desktop stays the single writer).</summary>
    public async Task<HfImportResult> HfImportAsync(
        string datasetId,
        string schemaId,
        string outPath,
        string config,
        string split,
        int limit,
        IReadOnlyDictionary<string, string> mapping
    )
    {
        var arguments = new List<string>
        {
            "hf-import",
            datasetId,
            "--schema",
            schemaId,
            "--out",
            outPath,
            "--config",
            config,
            "--split",
            split,
            "--limit",
            limit.ToString(CultureInfo.InvariantCulture),
        };
        foreach (var pair in mapping)
        {
            arguments.Add("--map");
            arguments.Add($"{pair.Key}={pair.Value}");
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<HfImportResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid HF import result.");
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

    public async Task<GateReport> RunDatasetGatesAsync(
        string projectPath,
        string schemaId,
        bool exportScope = false
    )
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var output = await RunEngineCommandAsync(
            "gate-run",
            examplesPath,
            schemaId,
            "--scope",
            exportScope ? "export" : "dataset",
            "--project-dir",
            projectPath
        );
        return JsonSerializer.Deserialize<GateReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid gate report.");
    }

    /// <summary>Run the chat-structure gates (chat_suite scope) over the project's dataset —
    /// conversation sequence checks the row validator can't see. Returns a GateReport shown
    /// through the same gate/Problems surface as dataset gates.</summary>
    public async Task<GateReport> RunChatGatesAsync(string projectPath)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var output = await RunEngineCommandAsync(
            "chat-gate", examplesPath, "--schema", "chat", "--project-dir", projectPath);
        return JsonSerializer.Deserialize<GateReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid gate report.");
    }

    // ---- Evaluation Suites (v1.3 M2) --------------------------------------------------

    /// <summary>List the registered evaluation suites under the project's evaluation_suites/.</summary>
    public async Task<IReadOnlyList<SuiteSummary>> ListSuitesAsync(string projectPath)
    {
        var output = await RunEngineCommandAsync("suite-list", "--project-dir", projectPath, "--json");
        return ParseSuiteSummaries(output);
    }

    /// <summary>Run a registered suite by name. Each case is a LIVE backend evaluation (slow).</summary>
    public async Task<SuiteReport> RunSuiteAsync(string projectPath, string suiteName)
    {
        var output = await RunEngineCommandAsync("suite-run", suiteName, "--project-dir", projectPath, "--json");
        return ParseSuiteReport(output);
    }

    /// <summary>Scaffold evaluation_suites/&lt;name&gt;.json for the user to edit (in the Files
    /// explorer). The engine validates the name and refuses to overwrite an existing suite.</summary>
    public Task NewSuiteAsync(string projectPath, string name) =>
        RunEngineCommandAsync("suite-init", name, "--project-dir", projectPath);

    /// <summary>Parse `suite-list --json` output. Pure/testable.</summary>
    public static IReadOnlyList<SuiteSummary> ParseSuiteSummaries(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return [];
        }
        return JsonSerializer.Deserialize<List<SuiteSummary>>(json, JsonOptions) ?? [];
    }

    /// <summary>Parse `suite-run --json` output. Pure/testable.</summary>
    public static SuiteReport ParseSuiteReport(string json)
    {
        return JsonSerializer.Deserialize<SuiteReport>(json, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid suite report.");
    }

    public async Task<ArenaReport> RunArenaAsync(
        string promptsText,
        IReadOnlyList<string> models,
        string? judgeModel = null,
        string? projectPath = null
    )
    {
        var suitePath = WritePromptSuiteToTempJsonl(promptsText);

        var arguments = new List<string> { "arena-run", suitePath };
        foreach (var model in models)
        {
            arguments.Add("--model");
            arguments.Add(model);
        }
        if (!string.IsNullOrWhiteSpace(judgeModel))
        {
            arguments.Add("--judge-model");
            arguments.Add(judgeModel);
        }
        if (!string.IsNullOrWhiteSpace(projectPath))
        {
            arguments.Add("--project-dir");
            arguments.Add(projectPath);
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<ArenaReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid arena report.");
    }

    /// <summary>Turn one-prompt-per-line text into a JSONL prompt suite temp file.</summary>
    private static string WritePromptSuiteToTempJsonl(string promptsText)
    {
        var lines = (promptsText ?? string.Empty)
            .Replace("\r\n", "\n")
            .Split('\n')
            .Select(line => line.Trim())
            .Where(line => line.Length > 0);

        var builder = new StringBuilder();
        foreach (var line in lines)
        {
            builder.Append(JsonSerializer.Serialize(new Dictionary<string, string> { ["prompt"] = line }));
            builder.Append('\n');
        }

        var directory = Path.Combine(Path.GetTempPath(), "CorpusStudio");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, $"arena-{Guid.NewGuid():N}.jsonl");
        WriteAllTextAtomic(path, builder.ToString(), encoding: Utf8NoBom);
        return path;
    }

    // --- Training run registry (desktop writes-direct; engine owns the schema) ---

    public static string MintTrainingRunId()
    {
        return $"{DateTime.UtcNow:yyyyMMddTHHmmss}-{Guid.NewGuid():N}"[..24];
    }

    public static string UtcNowIso()
    {
        return DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture);
    }

    private static readonly System.Text.RegularExpressions.Regex ValidRunId =
        new("^[A-Za-z0-9._-]+$", System.Text.RegularExpressions.RegexOptions.Compiled);

    public void SaveTrainingRunRecord(string projectPath, TrainingRunRecord record)
    {
        // run_id must be filename-safe so distinct ids can never collapse to the
        // same file and silently overwrite each other.
        if (string.IsNullOrEmpty(record.RunId) || !ValidRunId.IsMatch(record.RunId))
        {
            throw new ArgumentException($"Invalid run_id '{record.RunId}'.", nameof(record));
        }

        var directory = Path.Combine(projectPath, "training_runs");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, record.RunId + ".json");
        WriteAllTextAtomic(path, JsonSerializer.Serialize(record, JsonOptions) + "\n", encoding: Utf8NoBom);
    }

    /// <summary>Load run records (newest first). A run left in `running` whose
    /// process is gone is reconciled to `interrupted` and persisted (to the file
    /// it was loaded from), so a force-closed/crashed run does not stay `running`
    /// forever. Duplicate-run_id files are tolerated (first wins), so one stray
    /// copy cannot hide the whole history.</summary>
    public IReadOnlyList<TrainingRunRecord> LoadTrainingRunRecords(string projectPath)
    {
        var directory = Path.Combine(projectPath, "training_runs");
        if (!Directory.Exists(directory))
        {
            return [];
        }

        var loaded = new List<(string Path, TrainingRunRecord Record)>();
        foreach (var file in Directory.EnumerateFiles(directory, "*.json"))
        {
            try
            {
                var record = JsonSerializer.Deserialize<TrainingRunRecord>(File.ReadAllText(file), JsonOptions);
                if (record is not null)
                {
                    loaded.Add((file, record));
                }
            }
            catch
            {
                // Skip a corrupt record rather than failing the whole listing.
            }
        }

        // Tolerate duplicate run_ids (e.g. a hand-copied file): keep the first.
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var unique = loaded.Where(item => seen.Add(item.Record.RunId)).ToList();

        var now = UtcNowIso();
        foreach (var (path, record) in unique)
        {
            if (record.Status == "running" && !IsRecordProcessAlive(record))
            {
                record.Status = "interrupted";
                record.UpdatedAt = now;
                record.Notes = (string.IsNullOrEmpty(record.Notes) ? string.Empty : record.Notes + " ")
                    + "reconciled: process not alive on load";
                try
                {
                    // Persist to the file it was loaded from, not a recomputed name.
                    WriteAllTextAtomic(path, JsonSerializer.Serialize(record, JsonOptions) + "\n", encoding: Utf8NoBom);
                }
                catch
                {
                    // Best-effort persistence of the reconciliation.
                }
            }
        }

        var records = unique.Select(item => item.Record).ToList();
        records.Sort((a, b) => string.CompareOrdinal(b.RunId, a.RunId));
        return records;
    }

    // --- Model artifact registry (v0.9; writes-direct, reference paths only) ---

    private static readonly string[] ArtifactDescriptorFiles = ["adapter_config.json", "config.json"];

    /// <summary>Cheap fingerprint (never hashes weight bytes): a file's size+mtime,
    /// or a directory's key descriptor file. Null when nothing is readable.</summary>
    public static string? ComputeArtifactFingerprint(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                var info = new FileInfo(path);
                return $"{info.Length}:{info.LastWriteTimeUtc.Ticks}";
            }
            if (Directory.Exists(path))
            {
                var descriptor = ArtifactDescriptorFiles
                    .Select(name => Path.Combine(path, name))
                    .FirstOrDefault(File.Exists)
                    ?? Directory.EnumerateFiles(path).OrderBy(p => p, StringComparer.Ordinal).FirstOrDefault();
                if (descriptor is null)
                {
                    return null;
                }
                var info = new FileInfo(descriptor);
                return $"{Path.GetFileName(descriptor)}={info.Length}:{info.LastWriteTimeUtc.Ticks}";
            }
        }
        catch
        {
            return null;
        }
        return null;
    }

    /// <summary>Integrity vs current disk state (never persisted). Pure over an
    /// injected fingerprint function so it is unit-testable.</summary>
    public static string ComputeArtifactIntegrity(ModelArtifactRecord record, Func<string, string?> fingerprintOf)
    {
        if (!File.Exists(record.Path) && !Directory.Exists(record.Path))
        {
            return "missing";
        }
        if (string.IsNullOrEmpty(record.Fingerprint))
        {
            return "ok"; // could not verify at register time; don't cry wolf
        }
        var current = fingerprintOf(record.Path);
        if (current is null)
        {
            return "ok";
        }
        return current == record.Fingerprint ? "ok" : "modified";
    }

    private static string NormalizeArtifactPath(string path) => Path.GetFullPath(path);

    private static string MakeArtifactId(string runId, string normalizedPath)
    {
        var hash = Convert.ToHexString(
            System.Security.Cryptography.SHA1.HashData(Utf8NoBom.GetBytes(normalizedPath)))
            .ToLowerInvariant()[..8];
        return $"{runId}-{hash}";
    }

    private void SaveArtifactRecord(string projectPath, ModelArtifactRecord record)
    {
        if (string.IsNullOrEmpty(record.ArtifactId) || !ValidRunId.IsMatch(record.ArtifactId))
        {
            throw new ArgumentException($"Invalid artifact_id '{record.ArtifactId}'.", nameof(record));
        }
        var directory = Path.Combine(projectPath, "model_artifacts");
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, record.ArtifactId + ".json");
        WriteAllTextAtomic(path, JsonSerializer.Serialize(record, JsonOptions) + "\n", encoding: Utf8NoBom);
    }

    /// <summary>Register (idempotently) an artifact for a run. Re-registering the
    /// same run+path preserves created_at + status and refreshes fingerprint.</summary>
    public ModelArtifactRecord RegisterArtifact(
        string projectPath, string runId, string path, string kind = "adapter", string notes = "")
    {
        if (!File.Exists(Path.Combine(projectPath, "training_runs", runId + ".json")))
        {
            throw new InvalidOperationException($"No training run '{runId}' to attach an artifact to.");
        }

        var normalized = NormalizeArtifactPath(path);
        var artifactId = MakeArtifactId(runId, normalized);
        var artifactFile = Path.Combine(projectPath, "model_artifacts", artifactId + ".json");
        var now = UtcNowIso();

        var createdAt = now;
        var status = "candidate";
        var keepNotes = notes;
        if (File.Exists(artifactFile))
        {
            try
            {
                var existing = JsonSerializer.Deserialize<ModelArtifactRecord>(File.ReadAllText(artifactFile), JsonOptions);
                if (existing is not null)
                {
                    createdAt = existing.CreatedAt;
                    status = existing.Status;
                    keepNotes = string.IsNullOrEmpty(notes) ? existing.Notes : notes;
                }
            }
            catch
            {
                // A corrupt prior record is replaced.
            }
        }

        var record = new ModelArtifactRecord
        {
            ArtifactId = artifactId,
            RunId = runId,
            CreatedAt = createdAt,
            UpdatedAt = now,
            Path = normalized,
            Kind = string.IsNullOrWhiteSpace(kind) ? "adapter" : kind,
            Status = status,
            Fingerprint = ComputeArtifactFingerprint(normalized),
            Notes = keepNotes,
        };
        SaveArtifactRecord(projectPath, record);
        return record;
    }

    public IReadOnlyList<(ModelArtifactRecord Record, string Integrity)> LoadArtifacts(
        string projectPath, Func<ModelArtifactRecord, string>? integrityOf = null)
    {
        integrityOf ??= record => ComputeArtifactIntegrity(record, ComputeArtifactFingerprint);
        var directory = Path.Combine(projectPath, "model_artifacts");
        if (!Directory.Exists(directory))
        {
            return [];
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var records = new List<ModelArtifactRecord>();
        foreach (var file in Directory.EnumerateFiles(directory, "*.json"))
        {
            try
            {
                var record = JsonSerializer.Deserialize<ModelArtifactRecord>(File.ReadAllText(file), JsonOptions);
                if (record is not null && seen.Add(record.ArtifactId))
                {
                    records.Add(record);
                }
            }
            catch
            {
                // Skip a corrupt record.
            }
        }

        records.Sort((a, b) => string.CompareOrdinal(b.ArtifactId, a.ArtifactId));
        return records.Select(record => (record, integrityOf(record))).ToList();
    }

    /// <summary>Render the weight card markdown for an artifact (live; nothing stored).</summary>
    public async Task<string> GetWeightCardAsync(string projectPath, string artifactId)
    {
        return await RunEngineCommandAsync("artifact-card", projectPath, "--artifact-id", artifactId);
    }

    /// <summary>Run the promote gate for an artifact (integrity + source-run regression).</summary>
    public async Task<GateReport> GateArtifactAsync(string projectPath, string artifactId)
    {
        var output = await RunEngineCommandAsync("artifact-gate", projectPath, "--artifact-id", artifactId);
        return JsonSerializer.Deserialize<GateReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid gate report.");
    }

    /// <summary>Promote (keep) an artifact through the ENGINE, which enforces the promote gate
    /// on the status write — so a keep can't bypass integrity/regression checks. Throws (with the
    /// gate's message) when the engine blocks the promotion.</summary>
    public async Task<ModelArtifactRecord> PromoteArtifactAsync(string projectPath, string artifactId)
    {
        var output = await RunEngineCommandAsync(
            "artifact-update", projectPath, "--artifact-id", artifactId, "--status", "kept");
        return JsonSerializer.Deserialize<ModelArtifactRecord>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid artifact record.");
    }

    /// <summary>Directly set an artifact's status to <c>candidate</c> or <c>rejected</c> (both
    /// ungated by design). Promotion to <c>kept</c> is refused here: it must go through the
    /// gated engine path (<see cref="PromoteArtifactAsync"/> → <c>artifact-update --status kept</c>,
    /// which runs the promote gate), so this direct C# writer can never bypass that gate.</summary>
    public ModelArtifactRecord UpdateArtifactStatus(string projectPath, string artifactId, string status)
    {
        if (status == "kept")
        {
            throw new ArgumentException(
                "Promoting an artifact to 'kept' must go through the gated engine path "
                    + "(PromoteArtifactAsync), which runs the promote gate; this direct writer "
                    + "only sets 'candidate' or 'rejected'.",
                nameof(status));
        }
        if (status is not ("candidate" or "rejected"))
        {
            throw new ArgumentException($"Unknown artifact status '{status}'.", nameof(status));
        }
        var file = Path.Combine(projectPath, "model_artifacts", artifactId + ".json");
        if (!File.Exists(file))
        {
            throw new InvalidOperationException($"No artifact '{artifactId}'.");
        }
        var record = JsonSerializer.Deserialize<ModelArtifactRecord>(File.ReadAllText(file), JsonOptions)
            ?? throw new InvalidOperationException("Corrupt artifact record.");
        record.Status = status;
        record.UpdatedAt = UtcNowIso();
        SaveArtifactRecord(projectPath, record);
        return record;
    }

    /// <summary>Link an after-training evaluation (path + the model it targeted,
    /// for provenance) to the newest run record. Returns the linked run_id.</summary>
    public string? LinkAfterEvalToNewestRun(string projectPath, string afterEvalPath, string? afterEvalModel)
    {
        var newest = LoadTrainingRunRecords(projectPath).FirstOrDefault();
        if (newest is null)
        {
            return null;
        }

        newest.AfterEvalPath = afterEvalPath;
        newest.AfterEvalModel = afterEvalModel;
        newest.UpdatedAt = UtcNowIso();
        SaveTrainingRunRecord(projectPath, newest);
        return newest.RunId;
    }

    public async Task<GateReport> RunTrainingRunGateAsync(string projectPath, string runId)
    {
        var output = await RunEngineCommandAsync("training-run-gate", projectPath, "--run-id", runId);
        return JsonSerializer.Deserialize<GateReport>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid gate report.");
    }

    /// <summary>Flip any `running` record that is not alive to `interrupted`.
    /// Pure over an injected liveness check so it is unit-testable.</summary>
    public static IReadOnlyList<TrainingRunRecord> ReconcileRunningRecords(
        IReadOnlyList<TrainingRunRecord> records,
        Func<TrainingRunRecord, bool> isAlive,
        string updatedAt
    )
    {
        foreach (var record in records)
        {
            if (record.Status == "running" && !isAlive(record))
            {
                record.Status = "interrupted";
                record.UpdatedAt = updatedAt;
                record.Notes = (string.IsNullOrEmpty(record.Notes) ? string.Empty : record.Notes + " ")
                    + "reconciled: process not alive on load";
            }
        }

        return records;
    }

    /// <summary>Liveness by pid AND process identity: a recycled pid whose start
    /// time differs from the recorded one is treated as dead.</summary>
    private static bool IsRecordProcessAlive(TrainingRunRecord record)
    {
        if (record.Pid is not int pid)
        {
            return false;
        }

        try
        {
            using var process = Process.GetProcessById(pid);
            if (process.HasExited)
            {
                return false;
            }

            if (!string.IsNullOrEmpty(record.ProcessStartedAt)
                && DateTime.TryParse(
                    record.ProcessStartedAt,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.RoundtripKind,
                    out var recordedStart))
            {
                try
                {
                    if (Math.Abs((process.StartTime - recordedStart).TotalSeconds) > 2)
                    {
                        return false; // pid was recycled by an unrelated process
                    }
                }
                catch
                {
                    // Start time unreadable -> fall back to presence only.
                }
            }

            return true;
        }
        catch
        {
            return false; // Not running (or not queryable) -> treat as dead.
        }
    }

    public async Task<IReadOnlyList<ProviderPolicyItem>> GetProviderPoliciesAsync(string projectPath)
    {
        var output = await RunEngineCommandAsync("provider-policy", "--project-dir", projectPath);
        var result = JsonSerializer.Deserialize<ProviderPolicyListResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid provider policy list.");
        return result.Providers.Values
            .OrderBy(policy => policy.ProviderId, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public async Task ApproveProviderGenerationAsync(
        string projectPath,
        string providerId,
        string modelId,
        bool revoke = false
    )
    {
        var arguments = new List<string>
        {
            "provider-approve",
            "--provider",
            providerId,
            "--project-dir",
            projectPath,
            "--model",
            modelId,
        };
        if (revoke)
        {
            arguments.Add("--revoke");
        }

        // Throws (via RunEngineCommandAsync) if the engine rejects the request,
        // e.g. approving an evaluator-only provider (exit code 2).
        await RunEngineCommandAsync(arguments.ToArray());
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

    // --- Dataset version history (v1.0; engine-through, never recompute fingerprint) ---

    /// <summary>List dataset versions (newest first) with live integrity, via the engine
    /// (<c>dataset-version-list</c>). The engine owns the fingerprint, so integrity is
    /// verified against the current dataset rather than guessed in C#.</summary>
    public async Task<IReadOnlyList<DatasetVersionDisplayItem>> LoadDatasetVersionsAsync(string projectPath)
    {
        var output = await RunEngineCommandAsync("dataset-version-list", projectPath);
        return ParseDatasetVersionList(output);
    }

    /// <summary>Capture a dataset version through the engine (<c>dataset-version-create</c>),
    /// which computes the fingerprint + row count. Opt-in; call only when the dataset is
    /// quiescent (after an append/import commit), never automatically per-mutation.</summary>
    public async Task<DatasetVersionRecord> CreateDatasetVersionAsync(
        string projectPath, string label, string trigger)
    {
        var arguments = new List<string> { "dataset-version-create", projectPath };
        if (!string.IsNullOrWhiteSpace(trigger))
        {
            arguments.Add("--trigger");
            arguments.Add(trigger);
        }
        if (!string.IsNullOrWhiteSpace(label))
        {
            arguments.Add("--label");
            arguments.Add(label);
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return ParseDatasetVersionRecord(output);
    }

    /// <summary>Render a dataset version card (live projection; nothing stored).</summary>
    public async Task<string> GetDatasetVersionCardAsync(string projectPath, string versionId)
    {
        return await RunEngineCommandAsync("dataset-version-show", projectPath, "--version-id", versionId);
    }

    /// <summary>Render a diff (added/removed/common + sample rows) between two versions,
    /// via the engine (<c>dataset-version-diff</c>). Read-only; needs both versions to have
    /// stored rows — the engine refuses (throws) otherwise, which the caller surfaces.</summary>
    public async Task<string> GetDatasetVersionDiffAsync(string projectPath, string baseVersionId, string otherVersionId)
    {
        return await RunEngineCommandAsync(
            "dataset-version-diff", projectPath, "--version-id", baseVersionId, "--other", otherVersionId);
    }

    /// <summary>Assess the current dataset's debt via the engine (<c>dataset-debt --json</c>):
    /// the quality signals, normalized by dataset size, ranked, and graded. The engine owns
    /// all computation — the desktop only parses and colors.</summary>
    public async Task<DebtReport> GetDatasetDebtAsync(string projectPath)
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            // No dataset on disk yet: report "no rows to assess" (N/A), matching the
            // engine's empty-dataset contract, instead of surfacing a raw file error.
            return new DebtReport { HasData = false, Grade = "N/A", ExampleCount = 0, Items = [] };
        }
        var output = await RunEngineCommandAsync("dataset-debt", examplesPath, "--json");
        return ParseDebtReport(output);
    }

    /// <summary>Parse a <c>dataset-debt --json</c> DebtReport. Pure/static for testability.</summary>
    public static DebtReport ParseDebtReport(string json)
    {
        return JsonSerializer.Deserialize<DebtReport>(json, JsonOptions)
            ?? throw new InvalidOperationException("Engine returned no debt report.");
    }

    /// <summary>Parse a <c>dataset-version-list</c> payload into display rows. Pure/static
    /// so the JSON contract (including the live <c>current_integrity</c> annotation) is
    /// unit-testable without spawning the engine.</summary>
    public static IReadOnlyList<DatasetVersionDisplayItem> ParseDatasetVersionList(string json)
    {
        var result = JsonSerializer.Deserialize<DatasetVersionListResult>(json, JsonOptions);
        if (result?.Versions is null)
        {
            return [];
        }

        return result.Versions.Select(record => new DatasetVersionDisplayItem(record)).ToList();
    }

    /// <summary>Parse a bare <c>dataset-version-create</c> record (no integrity annotation).</summary>
    public static DatasetVersionRecord ParseDatasetVersionRecord(string json)
    {
        return JsonSerializer.Deserialize<DatasetVersionRecord>(json, JsonOptions)
            ?? throw new InvalidOperationException("Engine returned no dataset version record.");
    }

    /// <summary>Parse a <c>dataset-version-restore --json</c> RestoreResult.</summary>
    public static RestoreResult ParseRestoreResult(string json)
    {
        return JsonSerializer.Deserialize<RestoreResult>(json, JsonOptions)
            ?? throw new InvalidOperationException("Engine returned no restore result.");
    }

    /// <summary>Reconstruct a version to a temp file BESIDE examples.jsonl (same volume,
    /// so the later swap can be atomic), verified against the recorded fingerprint. The
    /// engine refuses examples.jsonl, so the temp is a distinct sibling name.</summary>
    public async Task<(string TempPath, RestoreResult Result)> RestoreVersionToTempAsync(
        string projectPath, string versionId)
    {
        var tempPath = Path.Combine(
            projectPath, "examples.jsonl.restore-" + Guid.NewGuid().ToString("N") + ".tmp");
        // VERIFY ON (never --no-verify): the engine writes the temp only if the
        // reconstruction matches the version's recorded fingerprint.
        var output = await RunEngineCommandAsync(
            "dataset-version-restore", projectPath, "--version-id", versionId,
            "--output", tempPath, "--json");
        return (tempPath, ParseRestoreResult(output));
    }

    /// <summary>Atomically move an EXISTING temp file onto a target on the same volume:
    /// <c>File.Replace</c> when the target exists, else <c>File.Move</c>. Distinct from
    /// <see cref="WriteAllTextAtomic"/>, which takes content rather than a source file.</summary>
    public static void AtomicReplaceFromFile(string sourceTempPath, string targetPath)
    {
        if (File.Exists(targetPath))
        {
            File.Replace(sourceTempPath, targetPath, destinationBackupFileName: null);
        }
        else
        {
            File.Move(sourceTempPath, targetPath);
        }
    }

    /// <summary>Whether a captured undo version is a safe recovery point before a restore.
    /// A "successful" capture (no exception) is NOT enough: the engine records a
    /// fingerprint-only version with <c>rows_stored=false</c> (exit 0) when the row store
    /// can't be written, and such a version cannot be restored. It is safe to proceed only
    /// when the undo actually stored its rows, OR there was genuinely nothing to preserve.
    /// <para>"Nothing to preserve" is NOT simply <c>RowCount == 0</c>: a present-but-UNREADABLE
    /// dataset also reports 0 rows (the engine returns a hollow capture with a NULL fingerprint),
    /// and restoring over it would overwrite recoverable bytes with no way back. So a 0-row undo
    /// is safe only when the current file is absent, or genuinely empty — which the engine
    /// records with a real content fingerprint (a null fingerprint means "couldn't read").</para></summary>
    public static bool IsUndoRestorable(DatasetVersionRecord undo, bool currentDatasetExists)
    {
        if (undo.RowsStored)
        {
            return true;
        }
        if (!currentDatasetExists)
        {
            return true; // nothing on disk to lose (missing / fresh project)
        }
        // File exists: allow only when it is GENUINELY empty (fingerprint present), never when
        // it is present-but-unreadable (0 rows + null fingerprint).
        return undo.RowCount == 0 && undo.ContentFingerprint is not null;
    }

    /// <summary>Restore a version in place — the app's highest-stakes write. The ordering
    /// is deliberately paranoid because it overwrites the user's dataset:
    /// <list type="number">
    /// <item>Capture the CURRENT dataset as an undo version AND confirm it is a genuine,
    /// restorable recovery point. If capture throws, or produced a hollow undo for a
    /// non-empty dataset, we abort so examples.jsonl is never overwritten unrecoverably.</item>
    /// <item>Reconstruct the selected version to a verified temp beside examples.jsonl.</item>
    /// <item>Atomically swap the temp onto examples.jsonl.</item>
    /// </list>
    /// Any failure before step 3 leaves examples.jsonl untouched; <c>File.Replace</c>
    /// makes step 3 all-or-nothing. The temp is always cleaned up.</summary>
    public async Task<RestoreResult> RestoreDatasetVersionInPlaceAsync(
        string projectPath, string versionId, string undoLabel)
    {
        // (1) Undo point FIRST. A throw here aborts before the dataset is touched.
        var undo = await CreateDatasetVersionAsync(projectPath, undoLabel, "before_restore");

        // A non-throwing capture is not proof of a usable undo: the engine records a
        // fingerprint-only version (rows_stored=false, exit 0) when the row store can't be
        // written OR when the current dataset was present-but-unreadable. Overwriting the
        // dataset after that would destroy recoverable bytes with no way back, so refuse
        // unless the undo is genuinely restorable (or there was nothing to preserve).
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!IsUndoRestorable(undo, File.Exists(examplesPath)))
        {
            throw new InvalidOperationException(
                "Refusing to restore: the current dataset could not be safely captured for undo — "
                + "it is present but unreadable/corrupted, or the row store could not be written "
                + "(low disk space or a locked project). Restoring now would overwrite it with no way "
                + "back. Fix or remove the current examples.jsonl, or free space/unlock the project, "
                + "and retry. Your dataset was not changed.");
        }

        string? tempPath = null;
        try
        {
            var (temp, result) = await RestoreVersionToTempAsync(projectPath, versionId);
            tempPath = temp;
            // (3) Atomic swap. File.Replace/Move consumes the temp; on success it is gone.
            AtomicReplaceFromFile(tempPath, examplesPath);
            tempPath = null;
            return result;
        }
        finally
        {
            // Robust cleanup: if the temp still exists (restore or swap failed), remove it.
            if (tempPath is not null)
            {
                try
                {
                    File.Delete(tempPath);
                }
                catch
                {
                    // Best-effort; a leftover .tmp is harmless.
                }
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
        AppendAllTextAtomic(examplesPath, jsonl, Utf8NoBom);
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

        // Seed the seen-set with the rows already in the dataset so a re-import (or a row
        // already present) doesn't append a duplicate — the commit is idempotent.
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        var seen = new HashSet<string>(StringComparer.Ordinal);
        if (File.Exists(examplesPath))
        {
            foreach (var existingLine in File.ReadLines(examplesPath, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(existingLine))
                {
                    continue;
                }
                var existingSignature = CanonicalRowSignature(existingLine);
                if (existingSignature is not null)
                {
                    seen.Add(existingSignature);
                }
            }
        }

        var rows = new List<string>();
        var skippedDuplicates = 0;
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
            var serialized = JsonSerializer.Serialize(document.RootElement);
            var signature = CanonicalRowSignature(serialized);
            // seen.Add returns false when the signature is already present (in the dataset or
            // earlier in this batch) — skip it and count it as a duplicate.
            if (signature is not null && !seen.Add(signature))
            {
                skippedDuplicates++;
                continue;
            }
            rows.Add(serialized);
        }

        var quarantinePath = report.FailedRows.Count == 0
            ? null
            : WriteImportQuarantine(projectPath, importPath, report);

        if (rows.Count > 0)
        {
            Directory.CreateDirectory(projectPath);
            var jsonl = string.Join(Environment.NewLine, rows) + Environment.NewLine;
            AppendAllTextAtomic(examplesPath, jsonl, Utf8NoBom);
        }

        return new ImportCommitResult(
            rows.Count,
            report.FailedRows.Count,
            quarantinePath,
            skippedDuplicates
        );
    }

    public async Task<ExportResult> ExportProjectExamplesAsync(
        string projectPath,
        string schemaId,
        bool removeDuplicates = false,
        bool removeLowInformation = false
    )
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputPath = Path.Combine(ResolveExportRoot(), projectId, "export.jsonl");

        var arguments = new List<string> { "export", examplesPath, outputPath, schemaId };
        if (removeDuplicates)
        {
            arguments.Add("--dedupe");
        }
        if (removeLowInformation)
        {
            arguments.Add("--drop-low-information");
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<ExportResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The engine returned an invalid export result.");
    }

    public async Task<PreferenceExportResult> ExportPreferenceForTrainingAsync(
        string projectPath,
        string format
    )
    {
        var examplesPath = Path.Combine(projectPath, "examples.jsonl");
        if (!File.Exists(examplesPath))
        {
            throw new FileNotFoundException("Project examples file was not found.", examplesPath);
        }

        var projectId = new DirectoryInfo(projectPath).Name;
        var outputDirectory = Path.Combine(ResolveExportRoot(), projectId, "preference_export");
        Directory.CreateDirectory(outputDirectory);
        var outputPath = Path.Combine(outputDirectory, $"preference_{format}.jsonl");

        var output = await RunEngineCommandAsync(
            "preference-export",
            examplesPath,
            "--output-path",
            outputPath,
            "--format",
            format
        );
        return JsonSerializer.Deserialize<PreferenceExportResult>(output, JsonOptions)
            ?? throw new InvalidOperationException(
                "The engine returned an invalid preference export result."
            );
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

        WriteAllTextAtomic(
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

    public async Task<BenchmarkReport> RunBenchmarkAsync(
        string projectPath,
        string schemaId,
        string backend,
        IReadOnlyList<string> models,
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
        var reportPath = Path.Combine(reportDirectory, $"{timestamp}_benchmark_report.json");

        var arguments = new List<string>
        {
            "benchmark",
            examplesPath,
            schemaId,
            "--backend",
            backend,
            "--output-path",
            reportPath,
            "--score-threshold",
            scoreThreshold.ToString(CultureInfo.InvariantCulture),
            "--timeout-seconds",
            timeoutSeconds.ToString(CultureInfo.InvariantCulture),
        };

        foreach (var model in models)
        {
            arguments.Add("--model");
            arguments.Add(model);
        }

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
        var parsed = JsonSerializer.Deserialize<BenchmarkRunOutput>(reportJson, JsonOptions);
        return parsed?.Benchmark
            ?? throw new InvalidOperationException("The Python engine returned an invalid benchmark report.");
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
        WriteAllTextAtomic(reportItem.ReportPath, updatedJson + Environment.NewLine, encoding: Utf8NoBom);

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

    public async Task<TrainingCompatibilityResult> CheckTrainingCompatibilityAsync(
        string schemaId,
        string datasetFormat,
        string target
    )
    {
        var arguments = new List<string>
        {
            "training-compat",
            "--schema",
            schemaId,
            "--target",
            target,
        };
        if (!string.IsNullOrWhiteSpace(datasetFormat))
        {
            arguments.Add("--format");
            arguments.Add(datasetFormat);
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<TrainingCompatibilityResult>(output, JsonOptions)
            ?? throw new InvalidOperationException(
                "The engine returned an invalid training compatibility result."
            );
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

    public async Task<TrainingCheckpointsResult> GetTrainingCheckpointsAsync(
        string outputDirectory,
        string target,
        string? configPath
    )
    {
        var arguments = new List<string>
        {
            "training-checkpoints",
            outputDirectory,
            "--target",
            target,
        };
        if (!string.IsNullOrWhiteSpace(configPath))
        {
            arguments.Add("--config-path");
            arguments.Add(configPath);
        }

        var output = await RunEngineCommandAsync(arguments.ToArray());
        return JsonSerializer.Deserialize<TrainingCheckpointsResult>(output, JsonOptions)
            ?? throw new InvalidOperationException("The Python engine returned an invalid checkpoints result.");
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

        WriteAllTextAtomic(
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

        WriteAllTextAtomic(
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
        if (!IsEngineAvailable)
        {
            throw new InvalidOperationException(
                "The Python engine is not available. "
                + (EngineUnavailableReason ?? "Locate the engine folder to continue."));
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = _pythonExecutable,
            WorkingDirectory = engineDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Utf8NoBom,
            StandardErrorEncoding = Utf8NoBom,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add("corpus_studio.cli");

        // Force UTF-8 across the desktop<->engine pipe so non-ASCII dataset text
        // round-trips instead of corrupting on the Windows console/pipe code page.
        startInfo.Environment["PYTHONUTF8"] = "1";
        startInfo.Environment["PYTHONIOENCODING"] = "utf-8";

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

        // No hard timeout: local eval / AI Assist runs can legitimately be long.
        // Cancellation is user-driven via CancelRunningEngineCommand().
        using var runCts = new CancellationTokenSource();
        _currentRunCts = runCts;

        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        var stopwatch = Stopwatch.StartNew();

        try
        {
            await process.WaitForExitAsync(runCts.Token);
        }
        catch (OperationCanceledException)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // Process may have exited between the HasExited check and Kill.
            }

            stopwatch.Stop();
            EmitCommandLog(arguments, exitCode: -1, stopwatch.ElapsedMilliseconds, stderr: null, cancelled: true);
            throw new OperationCanceledException("The engine command was cancelled.");
        }
        finally
        {
            if (ReferenceEquals(_currentRunCts, runCts))
            {
                _currentRunCts = null;
            }
        }

        var output = await outputTask;
        var error = await errorTask;
        stopwatch.Stop();
        EmitCommandLog(arguments, process.ExitCode, stopwatch.ElapsedMilliseconds, error, cancelled: false);

        return new EngineProcessResult(process.ExitCode, output, error);
    }

    /// <summary>Raise <see cref="CommandCompleted"/> for one engine invocation. Best-effort:
    /// a throwing subscriber must never break the engine call, so failures are swallowed.</summary>
    private void EmitCommandLog(string[] arguments, int exitCode, long durationMs, string? stderr, bool cancelled)
    {
        var handler = CommandCompleted;
        if (handler is null)
        {
            return;
        }

        try
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss", CultureInfo.InvariantCulture);
            var entry = EngineLogEntry.FromInvocation(arguments, exitCode, durationMs, stderr, timestamp, cancelled);
            handler(this, entry);
        }
        catch
        {
            // The Output panel is a diagnostic convenience; it must never affect engine calls.
        }
    }

    private static string WriteDraftToTempJsonl(string draftText)
    {
        var directory = Path.Combine(Path.GetTempPath(), "CorpusStudio");
        Directory.CreateDirectory(directory);

        var path = Path.Combine(directory, $"{Guid.NewGuid():N}.jsonl");
        WriteAllTextAtomic(path, NormalizeDraftToJsonl(draftText), encoding: Utf8NoBom);
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

        WriteAllTextAtomic(
            quarantinePath,
            string.Join(Environment.NewLine, entries) + Environment.NewLine,
            encoding: Utf8NoBom
        );
        PruneQuarantineFiles(quarantineDirectory, MaxRetainedQuarantineFiles);
        return quarantinePath;
    }

    /// <summary>How many <c>*_rejected.jsonl</c> quarantine files to retain per project.
    /// Each import with failures writes one; this bounds the directory while keeping a
    /// generous recovery window. Recovered/emptied files are still deleted eagerly by
    /// <see cref="RemoveImportQuarantineItem"/>; this only trims the oldest unrecovered ones.</summary>
    public const int MaxRetainedQuarantineFiles = 50;

    /// <summary>Delete the oldest quarantine files beyond <paramref name="keep"/> so the
    /// directory cannot grow without bound across many imports. Files carry a sortable
    /// timestamp prefix, so ordinal-descending name order is newest-first. Best-effort:
    /// a file that cannot be deleted is left in place.</summary>
    public static void PruneQuarantineFiles(string quarantineDirectory, int keep)
    {
        if (keep < 0 || !Directory.Exists(quarantineDirectory))
        {
            return;
        }

        var stale = Directory
            .EnumerateFiles(quarantineDirectory, "*_rejected.jsonl")
            .OrderByDescending(path => Path.GetFileName(path), StringComparer.Ordinal)
            .Skip(keep)
            .ToList();
        foreach (var path in stale)
        {
            try { File.Delete(path); }
            catch (IOException) { /* best-effort; a locked file stays */ }
            catch (UnauthorizedAccessException) { /* best-effort */ }
        }
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
