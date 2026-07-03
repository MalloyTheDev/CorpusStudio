using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>Workspace System foundation (v1.2.2, slice 1): manifest, recent-workspace
/// registry, path safety, and file-kind classification. Pure logic + tolerant I/O — no
/// existing behavior touched.</summary>
public sealed class WorkspaceFoundationTests : IDisposable
{
    private readonly string _tempRoot;

    public WorkspaceFoundationTests()
    {
        _tempRoot = Path.Combine(Path.GetTempPath(), "cs-ws-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempRoot);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempRoot))
            {
                Directory.Delete(_tempRoot, recursive: true);
            }
        }
        catch (IOException)
        {
            // Best-effort cleanup.
        }
    }

    private string NewDir(string name)
    {
        var dir = Path.Combine(_tempRoot, name);
        Directory.CreateDirectory(dir);
        return dir;
    }

    // ---- Manifest ----------------------------------------------------------------

    [Fact]
    public void Manifest_RoundTrips()
    {
        var root = NewDir("ws-round");
        var service = new WorkspaceManifestService();
        var manifest = new WorkspaceProjectManifest
        {
            ProjectId = "demo",
            Name = "Demo Dataset",
            SchemaId = "instruction",
            TemplateId = "standard",
            CreatedAt = "2026-07-03T00:00:00Z",
        };

        Assert.Null(service.Write(root, manifest));
        Assert.True(service.HasManifest(root));

        var read = service.Read(root);
        Assert.True(read.Ok);
        Assert.Equal("demo", read.Manifest!.ProjectId);
        Assert.Equal("instruction", read.Manifest.SchemaId);
        Assert.Equal("examples.jsonl", read.Manifest.ExamplesFile);
        Assert.True(read.Manifest.IsRecognized);
        Assert.False(read.Manifest.IsFutureVersion);
    }

    [Fact]
    public void Manifest_Missing_IsErrorNotCrash()
    {
        var root = NewDir("ws-missing");
        var read = new WorkspaceManifestService().Read(root);
        Assert.False(read.Ok);
        Assert.NotNull(read.Error);
    }

    [Fact]
    public void Manifest_Malformed_IsErrorNotCrash()
    {
        var root = NewDir("ws-bad");
        var service = new WorkspaceManifestService();
        Directory.CreateDirectory(service.MetadataDirectory(root));
        File.WriteAllText(service.ManifestPath(root), "{ not json ]");

        var read = service.Read(root);
        Assert.False(read.Ok);
        Assert.Contains("not valid JSON", read.Error);
    }

    [Fact]
    public void Manifest_FutureVersion_StillOpensButFlags()
    {
        var root = NewDir("ws-future");
        var service = new WorkspaceManifestService();
        Directory.CreateDirectory(service.MetadataDirectory(root));
        File.WriteAllText(service.ManifestPath(root),
            "{\"format\":\"corpus_studio_project\",\"format_version\":999,\"project_id\":\"x\"}");

        var read = service.Read(root);
        Assert.True(read.Ok);
        Assert.True(read.Manifest!.IsRecognized);
        Assert.True(read.Manifest.IsFutureVersion);
    }

    [Fact]
    public void Manifest_UnrecognizedFormat_ParsesButNotRecognized()
    {
        var root = NewDir("ws-unrec");
        var service = new WorkspaceManifestService();
        Directory.CreateDirectory(service.MetadataDirectory(root));
        File.WriteAllText(service.ManifestPath(root),
            "{\"format\":\"something_else\",\"project_id\":\"x\"}");

        var read = service.Read(root);
        Assert.True(read.Ok);
        Assert.False(read.Manifest!.IsRecognized);   // caller can refuse
    }

    // ---- Path safety -------------------------------------------------------------

    [Fact]
    public void PathSafety_AllowsDescent()
    {
        var root = NewDir("ws-safe");
        Assert.True(WorkspacePathSafety.TryResolveWithinRoot(root, "assets/img.png", out var resolved));
        Assert.True(WorkspacePathSafety.IsWithinRoot(root, resolved));
    }

    [Theory]
    [InlineData("../escape.txt")]
    [InlineData("sub/../../escape.txt")]
    [InlineData("a/b/../../../escape.txt")]
    public void PathSafety_RejectsTraversal(string relative)
    {
        var root = NewDir("ws-trav-" + Math.Abs(relative.GetHashCode()));
        Assert.False(WorkspacePathSafety.TryResolveWithinRoot(root, relative, out _));
    }

    [Fact]
    public void PathSafety_RejectsAbsoluteChild()
    {
        var root = NewDir("ws-abs");
        var absolute = OperatingSystem.IsWindows() ? @"C:\Windows\system32" : "/etc/passwd";
        Assert.False(WorkspacePathSafety.TryResolveWithinRoot(root, absolute, out _));
    }

    [Fact]
    public void PathSafety_SiblingPrefix_IsNotWithin()
    {
        // "C:\ws" must not be treated as a parent of "C:\ws-other".
        var root = NewDir("ws-prefix");
        Assert.False(WorkspacePathSafety.IsWithinRoot(root, root + "-other"));
    }

    [Theory]
    [InlineData("  My Dataset  ", "My Dataset")]
    [InlineData("bad/name", "bad_name")]
    [InlineData("a\\b", "a_b")]
    [InlineData("..", "")]
    [InlineData("   ", "")]
    public void PathSafety_SanitizesSegmentNames(string input, string expected)
    {
        Assert.Equal(expected, WorkspacePathSafety.SanitizeSegmentName(input));
    }

    // ---- File kind ---------------------------------------------------------------

    [Theory]
    [InlineData("data.jsonl", WorkspaceFileKind.Jsonl)]
    [InlineData("card.json", WorkspaceFileKind.Json)]
    [InlineData("README.md", WorkspaceFileKind.Markdown)]
    [InlineData("notes.txt", WorkspaceFileKind.Text)]
    [InlineData("config.yaml", WorkspaceFileKind.Yaml)]
    [InlineData("pyproject.toml", WorkspaceFileKind.Toml)]
    [InlineData("train.py", WorkspaceFileKind.Code)]
    [InlineData("Main.cs", WorkspaceFileKind.Code)]
    [InlineData("cat.PNG", WorkspaceFileKind.Image)]
    [InlineData("clip.mp4", WorkspaceFileKind.VideoFuture)]
    [InlineData("sound.wav", WorkspaceFileKind.AudioFuture)]
    [InlineData("model.safetensors", WorkspaceFileKind.Binary)]
    [InlineData("mystery.xyz", WorkspaceFileKind.Unknown)]
    [InlineData("NOEXTENSION", WorkspaceFileKind.Unknown)]
    public void FileKind_Classifies(string name, WorkspaceFileKind expected)
    {
        Assert.Equal(expected, WorkspaceFileKinds.Classify(name, isDirectory: false));
    }

    [Fact]
    public void FileKind_DirectoryIsFolder_EvenWithExtensionLikeName()
    {
        Assert.Equal(WorkspaceFileKind.Folder, WorkspaceFileKinds.Classify("assets.json", isDirectory: true));
    }

    [Fact]
    public void FileKind_TextEditable()
    {
        Assert.True(WorkspaceFileKinds.IsTextEditable(WorkspaceFileKind.Jsonl));
        Assert.False(WorkspaceFileKinds.IsTextEditable(WorkspaceFileKind.Image));
        Assert.False(WorkspaceFileKinds.IsTextEditable(WorkspaceFileKind.Binary));
    }

    // ---- Recent workspaces -------------------------------------------------------

    private RecentWorkspaceService RecentService(Func<string, bool>? exists = null) =>
        new(Path.Combine(_tempRoot, "appdata-" + Guid.NewGuid().ToString("N")), exists);

    [Fact]
    public void Recent_MissingRegistry_LoadsEmpty()
    {
        var service = RecentService();
        Assert.Empty(service.Load());
    }

    [Fact]
    public void Recent_CorruptRegistry_RecoversEmpty()
    {
        var service = RecentService();
        Directory.CreateDirectory(Path.GetDirectoryName(service.RegistryPath)!);
        File.WriteAllText(service.RegistryPath, "}{ this is not json");
        Assert.Empty(service.Load());   // no crash
    }

    [Fact]
    public void Recent_SaveThenLoad_RoundTrips()
    {
        var service = RecentService(_ => true);
        var record = new RecentWorkspaceRecord { Name = "A", Path = Path.Combine(_tempRoot, "A") };
        Assert.Null(service.Save(new[] { record }));

        var loaded = service.Load();
        Assert.Single(loaded);
        Assert.Equal("A", loaded[0].Name);
        Assert.False(loaded[0].MissingPath);
    }

    [Fact]
    public void Recent_AddOrUpdate_DedupesByPathAndMovesToFront()
    {
        var p1 = Path.Combine(_tempRoot, "one");
        var p2 = Path.Combine(_tempRoot, "two");
        var list = new List<RecentWorkspaceRecord>
        {
            new() { Name = "one", Path = p1 },
            new() { Name = "two", Path = p2 },
        };

        // Re-open "two" -> single entry, moved to front.
        list = RecentWorkspaceService.AddOrUpdate(list, new RecentWorkspaceRecord { Name = "two", Path = p2 });
        Assert.Equal(2, list.Count);
        Assert.Equal(p2, list[0].Path);
    }

    [Fact]
    public void Recent_AddOrUpdate_PreservesPriorPin()
    {
        var p = Path.Combine(_tempRoot, "pinme");
        var list = new List<RecentWorkspaceRecord> { new() { Path = p, IsPinned = true } };
        list = RecentWorkspaceService.AddOrUpdate(list, new RecentWorkspaceRecord { Path = p, IsPinned = false });
        Assert.True(list[0].IsPinned);   // pin carries over
    }

    [Fact]
    public void Recent_PinUnpinRemove()
    {
        var p = Path.Combine(_tempRoot, "x");
        var list = new List<RecentWorkspaceRecord> { new() { Path = p } };

        list = RecentWorkspaceService.SetPinned(list, p, true);
        Assert.True(list[0].IsPinned);

        list = RecentWorkspaceService.SetPinned(list, p, false);
        Assert.False(list[0].IsPinned);

        list = RecentWorkspaceService.Remove(list, p);
        Assert.Empty(list);
    }

    [Fact]
    public void Recent_MissingPathDetected_ButKept()
    {
        // Probe reports the folder gone; the entry stays, flagged missing.
        var gone = Path.Combine(_tempRoot, "gone");
        var service = RecentService(path => false);
        Assert.Null(service.Save(new[] { new RecentWorkspaceRecord { Name = "gone", Path = gone } }));

        var loaded = service.Load();
        Assert.Single(loaded);
        Assert.True(loaded[0].MissingPath);
    }

    [Fact]
    public void Recent_Cap_KeepsPinned_AndMostRecentUnpinned()
    {
        var list = new List<RecentWorkspaceRecord>();
        // Most-recent first: r0..r4 unpinned; a pinned one at the tail.
        for (int i = 0; i < 5; i++)
        {
            list.Add(new RecentWorkspaceRecord { Path = Path.Combine(_tempRoot, "u" + i) });
        }
        var pinnedOld = new RecentWorkspaceRecord { Path = Path.Combine(_tempRoot, "pinned"), IsPinned = true };
        list.Add(pinnedOld);

        var capped = RecentWorkspaceService.ApplyCap(list, 3);
        Assert.Equal(3, capped.Count);
        Assert.Contains(capped, r => r.IsPinned);                  // pinned kept despite being oldest
        Assert.Equal(Path.Combine(_tempRoot, "u0"), capped[0].Path); // most-recent unpinned kept
        Assert.DoesNotContain(capped, r => r.Path.EndsWith("u4")); // oldest unpinned dropped
    }
}
