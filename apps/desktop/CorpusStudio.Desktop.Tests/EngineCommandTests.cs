using System;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.Services;
using CorpusStudio.Desktop.ViewModels;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The first engine-run orchestration moved off the desktop code-behind into a shared async
/// command (RunDatasetDebtCommand → RunDatasetDebtAsync), behind the IEngineService seam so it is
/// testable with a fake engine (no Python shell-out).</summary>
public sealed class EngineCommandTests
{
    private sealed class FakeEngine : IEngineService
    {
        private readonly DebtReport _debt;
        private readonly GateReport _gate;
        public string? LastProjectPath { get; private set; }
        public FakeEngine(DebtReport debt, GateReport? gate = null)
        {
            _debt = debt;
            _gate = gate ?? new GateReport();
        }

        public Task<DebtReport> GetDatasetDebtAsync(string projectPath)
        {
            LastProjectPath = projectPath;
            return Task.FromResult(_debt);
        }

        public Task<GateReport> RunDatasetGatesAsync(string projectPath, string schemaId, bool exportScope = false)
        {
            LastProjectPath = projectPath;
            return Task.FromResult(_gate);
        }

        public Task<GateReport> RunChatGatesAsync(string projectPath) => Task.FromResult(_gate);
        public Task<ArenaReport> RunArenaAsync(string promptsText, System.Collections.Generic.IReadOnlyList<string> models,
            string? judgeModel = null, string? projectPath = null) => Task.FromResult(new ArenaReport());
        public Task<SuiteReport> RunSuiteAsync(string projectPath, string suiteName) => Task.FromResult(new SuiteReport());
        public Task<string> GetDatasetVersionCardAsync(string projectPath, string versionId) => Task.FromResult("# card");
        public Task<string> GetDatasetVersionDiffAsync(string projectPath, string baseVersionId, string otherVersionId) => Task.FromResult("# diff");
        public Task<string> GetWeightCardAsync(string projectPath, string artifactId) => Task.FromResult("# weights");
        public Task<DatasetCardResult> GenerateDatasetCardAsync(string projectPath, string schemaId) => Task.FromResult(new DatasetCardResult());
    }

    private static MainWindowViewModel VmWith(IEngineService engine) => new(
        new DebtViewModel(), new ArenaViewModel(), new SettingsViewModel(), new VersionsViewModel(),
        new ArtifactsViewModel(), new SuitesViewModel(), new SplitsViewModel(), new PreferenceReviewViewModel(),
        new QuarantineViewModel(), new ExamplesViewModel(), new WritingStudioViewModel(),
        new AiAssistRewriteBatchesViewModel(), new AiAssistConnectionViewModel(),
        new EvaluationConnectionViewModel(), new QualityViewModel(), engine);

    private static void SelectFakeProject(MainWindowViewModel vm) => vm.SelectProject(
        new DatasetProjectListItem(
            new DatasetProject("p", "P", "instruction", new DateTime(2026, 1, 1), new DateTime(2026, 1, 1)),
            @"C:\fake\project"),
        "instruction");

    [Fact]
    public async Task RunDatasetDebt_WithoutProject_SetsDebtError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "D", HasData = true }));

        await vm.RunDatasetDebtAsync();

        Assert.Contains("Create or select a dataset project", vm.Debt.DebtSummary);
    }

    [Fact]
    public async Task RunDatasetDebt_AppliesTheEngineReportToTheDebtTab()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "D", HasData = true, ExampleCount = 19 });
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.RunDatasetDebtAsync();

        Assert.Equal("D", vm.Debt.DebtGrade);
        Assert.Equal(@"C:\fake\project", engine.LastProjectPath);
    }

    [Fact]
    public async Task RunGates_WithProject_InvokesTheEngineForThatProject()
    {
        var engine = new FakeEngine(new DebtReport { Grade = "A" }, new GateReport());
        var vm = VmWith(engine);
        SelectFakeProject(vm);

        await vm.RunGatesAsync();

        Assert.Equal(@"C:\fake\project", engine.LastProjectPath);
    }

    [Fact]
    public async Task RunGates_WithoutProject_SetsGateError()
    {
        var vm = VmWith(new FakeEngine(new DebtReport { Grade = "A" }));
        await vm.RunGatesAsync();
        Assert.Contains("Create or select a dataset project", vm.GateSummary);
    }
}
