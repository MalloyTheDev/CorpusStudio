using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Workspace content search ("find in files"): the service matches lines in
/// text files under the root, skipping binaries, oversize files, and ignored dirs; the
/// view-model runs it off-thread and shapes the results.</summary>
public sealed class WorkspaceSearchTests : IDisposable
{
    private readonly string _root;

    public WorkspaceSearchTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "cs-search-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best-effort */ }
    }

    private void Write(string relativePath, string content)
    {
        var full = Path.Combine(_root, relativePath.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, content);
    }

    // --- service -------------------------------------------------------------

    [Fact]
    public void Search_FindsMatchesAcrossTextFiles_CaseInsensitiveByDefault()
    {
        Write("a.txt", "the Quick brown fox\nsecond line");
        Write("docs/b.md", "quick reference\nnothing here");

        var result = new WorkspaceSearchService().Search(_root, "quick");

        Assert.Equal(2, result.Matches.Count);       // "Quick" and "quick"
        Assert.Equal(2, result.FilesMatched);
        Assert.Contains(result.Matches, m => m.RelativePath.EndsWith("a.txt") && m.LineNumber == 1);
        Assert.Contains(result.Matches, m => m.RelativePath.Replace('\\', '/') == "docs/b.md");
    }

    [Fact]
    public void Search_CaseSensitive_RespectsCase()
    {
        Write("a.txt", "the Quick brown fox");
        Write("b.md", "quick reference");

        var service = new WorkspaceSearchService();
        Assert.Single(service.Search(_root, "Quick", caseSensitive: true).Matches);   // only a.txt
        Assert.Single(service.Search(_root, "quick", caseSensitive: true).Matches);   // only b.md
        Assert.Equal(2, service.Search(_root, "quick", caseSensitive: false).Matches.Count);
    }

    [Fact]
    public void Search_SkipsBinaryAndIgnoredDirectories()
    {
        Write("real.txt", "quick match");
        File.WriteAllBytes(Path.Combine(_root, "image.png"), System.Text.Encoding.ASCII.GetBytes("quick in a png"));
        Write(".git/config", "quick in git");
        Write("node_modules/pkg/index.js", "quick in a dep");

        var result = new WorkspaceSearchService().Search(_root, "quick");

        Assert.Single(result.Matches);                                  // only real.txt
        Assert.Equal("real.txt", result.Matches[0].RelativePath.Replace('\\', '/'));
    }

    [Fact]
    public void Search_HonorsResultCapAndReportsTruncation()
    {
        Write("many.txt", string.Join("\n", Enumerable.Range(0, 10).Select(i => $"match {i}")));

        var result = new WorkspaceSearchService { MaxResults = 4 }.Search(_root, "match");

        Assert.Equal(4, result.Matches.Count);
        Assert.True(result.Truncated);
    }

    [Fact]
    public void Search_SkipsOversizeFiles()
    {
        Write("big.txt", new string('x', 200) + " quick");   // contains the term
        var result = new WorkspaceSearchService { MaxFileBytes = 50 }.Search(_root, "quick");
        Assert.Empty(result.Matches);
    }

    [Fact]
    public void Search_EmptyQueryOrNoRoot_ReturnsEmpty()
    {
        Write("a.txt", "quick");
        var service = new WorkspaceSearchService();
        Assert.Empty(service.Search(_root, "").Matches);
        Assert.Empty(service.Search(null, "quick").Matches);
    }

    [Fact]
    public void Search_TruncatesLongMatchingLine()
    {
        Write("long.txt", "quick " + new string('y', 500));
        var result = new WorkspaceSearchService { MaxLineLength = 40 }.Search(_root, "quick");
        Assert.Single(result.Matches);
        Assert.True(result.Matches[0].LineText.Length <= 41); // 40 + the ellipsis
        Assert.EndsWith("…", result.Matches[0].LineText);
    }

    // --- view-model ----------------------------------------------------------

    [Fact]
    public async Task ViewModel_RunAsync_PublishesMatchesAndStatus()
    {
        Write("a.txt", "quick brown fox");
        var vm = new WorkspaceSearchViewModel();
        vm.SetWorkspaceRoot(_root);
        vm.Query = "quick";

        await vm.RunAsync();

        Assert.True(vm.HasResults);
        Assert.Single(vm.Results);
        Assert.False(vm.IsSearching);
        Assert.Contains("1 match", vm.Status);
    }

    [Fact]
    public async Task ViewModel_BlankQuery_ClearsResults()
    {
        Write("a.txt", "quick");
        var vm = new WorkspaceSearchViewModel();
        vm.SetWorkspaceRoot(_root);
        vm.Query = "quick";
        await vm.RunAsync();
        Assert.True(vm.HasResults);

        vm.Query = "   ";
        await vm.RunAsync();
        Assert.False(vm.HasResults);
        Assert.Contains("Enter a search term", vm.Status);
    }

    [Fact]
    public async Task ViewModel_SwitchingRoot_ClearsStaleResults()
    {
        Write("a.txt", "quick");
        var vm = new WorkspaceSearchViewModel();
        vm.SetWorkspaceRoot(_root);
        vm.Query = "quick";
        await vm.RunAsync();
        Assert.True(vm.HasResults);

        var other = Path.Combine(Path.GetTempPath(), "cs-search-other-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(other);
        try
        {
            vm.SetWorkspaceRoot(other);
            Assert.False(vm.HasResults);   // stale results dropped on root switch
        }
        finally
        {
            try { Directory.Delete(other, recursive: true); } catch (IOException) { }
        }
    }

    [Fact]
    public async Task ViewModel_NoMatch_ReportsZero()
    {
        Write("a.txt", "nothing relevant");
        var vm = new WorkspaceSearchViewModel();
        vm.SetWorkspaceRoot(_root);
        vm.Query = "zzz-not-present";
        await vm.RunAsync();
        Assert.False(vm.HasResults);
        Assert.Contains("No matches", vm.Status);
    }
}
