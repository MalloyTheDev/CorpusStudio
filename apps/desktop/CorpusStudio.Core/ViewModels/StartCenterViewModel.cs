using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Backs the Start Center (v1.2.4 Workspace System, view layer). Owns the Recent
/// Workspaces list from <see cref="RecentWorkspaceService"/> — a missing folder is kept and
/// flagged, never dropped. Kept small and separate from MainWindowViewModel per the
/// workspace design.</summary>
public sealed class StartCenterViewModel : INotifyPropertyChanged
{
    private readonly RecentWorkspaceService _recents;

    public StartCenterViewModel(RecentWorkspaceService? recents = null)
    {
        _recents = recents ?? new RecentWorkspaceService();
        Refresh();
    }

    public ObservableCollection<RecentWorkspaceDisplayItem> Recents { get; } = new();

    public bool HasRecents => Recents.Count > 0;

    public string RecentCountLabel => Recents.Count switch
    {
        0 => "No recent workspaces",
        1 => "1 workspace",
        _ => $"{Recents.Count} workspaces",
    };

    /// <summary>Reload the registry from disk (refreshing live missing-path flags) and
    /// rebuild the display list.</summary>
    public void Refresh()
    {
        Recents.Clear();
        foreach (var record in _recents.Load())
        {
            Recents.Add(new RecentWorkspaceDisplayItem(record));
        }

        OnChanged(nameof(HasRecents));
        OnChanged(nameof(RecentCountLabel));
    }

    public void SetPinned(string path, bool pinned)
    {
        _recents.Save(RecentWorkspaceService.SetPinned(_recents.Load(), path, pinned));
        Refresh();
    }

    public void Remove(string path)
    {
        _recents.Save(RecentWorkspaceService.Remove(_recents.Load(), path));
        Refresh();
    }

    /// <summary>Record a just-opened or just-created workspace at the front of the registry
    /// (deduped by path, preserving a prior pin), then refresh.</summary>
    public void RecordOpened(string path, string name, string? schemaId, string nowIso)
    {
        var record = new RecentWorkspaceRecord
        {
            Path = path,
            Name = name,
            SchemaId = schemaId,
            LastOpenedAt = nowIso,
        };
        _recents.Save(RecentWorkspaceService.AddOrUpdate(_recents.Load(), record));
        Refresh();
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnChanged(string name) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>A Recent Workspaces card for the Start Center. Pure display projection of a
/// <see cref="RecentWorkspaceRecord"/> — computes the schema tag, a friendly timestamp, and
/// the pinned/missing badge state the prototype shows.</summary>
public sealed class RecentWorkspaceDisplayItem
{
    public RecentWorkspaceDisplayItem(RecentWorkspaceRecord record)
    {
        Path = record.Path;
        Name = record.DisplayName;
        SchemaId = record.SchemaId;
        IsPinned = record.IsPinned;
        IsMissing = record.MissingPath;
        SchemaTag = TagFor(record.SchemaId);
        WhenText = FormatWhen(record.LastOpenedAt);
    }

    public string Path { get; }
    public string Name { get; }
    public string? SchemaId { get; }
    public bool IsPinned { get; }
    public bool IsMissing { get; }

    /// <summary>Two-letter schema tag for the card's colored square (e.g. "IN", "CH").</summary>
    public string SchemaTag { get; }

    /// <summary>Friendly last-opened timestamp, or "" when unknown.</summary>
    public string WhenText { get; }

    public string PinTitle => IsPinned ? "Unpin" : "Pin to top";

    private static string TagFor(string? schemaId)
    {
        if (string.IsNullOrWhiteSpace(schemaId))
        {
            return "WS";
        }

        var letters = new string(schemaId.Where(char.IsLetter).ToArray());
        return (letters.Length >= 2 ? letters[..2] : letters.PadRight(2, ' ')).ToUpperInvariant();
    }

    private static string FormatWhen(string? lastOpenedAtIso)
    {
        if (string.IsNullOrWhiteSpace(lastOpenedAtIso))
        {
            return string.Empty;
        }

        return DateTimeOffset.TryParse(
            lastOpenedAtIso, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var when)
            ? when.LocalDateTime.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture)
            : string.Empty;
    }
}
