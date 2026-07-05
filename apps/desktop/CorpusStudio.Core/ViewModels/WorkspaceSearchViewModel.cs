using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Backs the workspace content-search ("find in files") panel. Holds the query,
/// runs <see cref="WorkspaceSearchService"/> off the UI thread, and exposes the flat match
/// list the view binds to. Read-only over the workspace; opening a result is the caller's job.</summary>
public sealed class WorkspaceSearchViewModel : INotifyPropertyChanged
{
    private readonly IWorkspaceSearchService _search;

    public WorkspaceSearchViewModel(IWorkspaceSearchService? search = null)
    {
        _search = search ?? new WorkspaceSearchService();
    }

    private string? _workspaceRoot;
    private string _query = string.Empty;
    private bool _caseSensitive;
    private bool _isSearching;
    private string _status = "Enter a search term to find it across this workspace's files.";

    public string Query
    {
        get => _query;
        set => SetField(ref _query, value);
    }

    public bool CaseSensitive
    {
        get => _caseSensitive;
        set => SetField(ref _caseSensitive, value);
    }

    public bool IsSearching
    {
        get => _isSearching;
        private set => SetField(ref _isSearching, value);
    }

    public string Status
    {
        get => _status;
        private set => SetField(ref _status, value);
    }

    public ObservableCollection<WorkspaceSearchMatch> Results { get; } = new();

    public bool HasResults => Results.Count > 0;

    /// <summary>Point the search at a workspace root. Switching roots clears stale results.</summary>
    public void SetWorkspaceRoot(string? root)
    {
        if (!SamePath(_workspaceRoot, root))
        {
            _workspaceRoot = root;
            Clear();
        }
    }

    /// <summary>Run the search off the UI thread and publish the matches. A blank query or no
    /// workspace clears the list rather than searching.</summary>
    public async Task RunAsync()
    {
        // Re-entrancy guard: ignore a new run while one is already in flight (rapid Enter presses
        // or a click during a search would otherwise race on Results and leave interleaved state).
        if (IsSearching)
        {
            return;
        }

        var query = _query?.Trim() ?? string.Empty;
        if (string.IsNullOrEmpty(query) || string.IsNullOrWhiteSpace(_workspaceRoot))
        {
            Clear();
            Status = "Enter a search term to find it across this workspace's files.";
            return;
        }

        IsSearching = true;
        Status = "Searching…";
        var root = _workspaceRoot;
        var caseSensitive = _caseSensitive;

        try
        {
            var result = await Task.Run(() => _search.Search(root, query, caseSensitive));

            Results.Clear();
            foreach (var match in result.Matches)
            {
                Results.Add(match);
            }
            Status = FormatStatus(result, query);
        }
        catch (System.Exception ex)
        {
            // The search is meant to be self-contained, but never let an unexpected failure leave
            // the panel stuck on "Searching…" (or crash an async-void handler that awaits this):
            // surface it and fall through to the finally that resets the busy flag.
            Results.Clear();
            Status = $"Search failed: {ex.Message}";
        }
        finally
        {
            IsSearching = false;
            OnChanged(nameof(HasResults));
        }
    }

    public void Clear()
    {
        Results.Clear();
        OnChanged(nameof(HasResults));
    }

    private static string FormatStatus(WorkspaceSearchResult result, string query)
    {
        if (result.Matches.Count == 0)
        {
            return $"No matches for \"{query}\" in {result.FilesScanned} file(s).";
        }

        var fileWord = result.FilesMatched == 1 ? "file" : "files";
        var matchWord = result.Matches.Count == 1 ? "match" : "matches";
        var summary = $"{result.Matches.Count} {matchWord} in {result.FilesMatched} {fileWord}";
        return result.Truncated ? $"{summary} (showing the first {result.Matches.Count})." : $"{summary}.";
    }

    private static bool SamePath(string? left, string? right)
    {
        if (string.IsNullOrWhiteSpace(left) && string.IsNullOrWhiteSpace(right))
        {
            return true;
        }
        if (string.IsNullOrWhiteSpace(left) || string.IsNullOrWhiteSpace(right))
        {
            return false;
        }
        var comparison = System.OperatingSystem.IsWindows()
            ? System.StringComparison.OrdinalIgnoreCase
            : System.StringComparison.Ordinal;
        return string.Equals(
            left.Replace('\\', '/').TrimEnd('/'),
            right.Replace('\\', '/').TrimEnd('/'),
            comparison);
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }
        field = value;
        OnChanged(name);
        return true;
    }

    private void OnChanged(string? name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
