using System.Collections.Generic;
using System.ComponentModel;

using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;

using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The Dashboard's 7-node lifecycle strip (Author · Validate · Quality · Gate · Split ·
/// Evaluate · Train) exposes a per-node status token on the shared VM. These tests pin the project's
/// HONESTY invariant: a node lights up only from a REAL signal, and every node with no reliable signal
/// STAYS NEUTRAL (a completed step is never faked; neutral-where-unknown can never read as done/green).</summary>
public sealed class LifecycleStatusTests
{
    private static SavedExampleItem Example(int row)
        => new(row, $"preview {row}", $"{{\"instruction\":\"i{row}\",\"input\":\"\",\"output\":\"o\"}}");

    private static QualityReport QualityReport(
        int examples = 100, int empty = 0, int pii = 0) => new()
    {
        ExampleCount = examples,
        EmptyRowCount = empty,
        PiiFindingCount = pii,
        PiiFindings = pii > 0
            ? new[] { new PiiFinding { Kind = "api_key", Severity = "high", MatchCount = pii } }
            : [],
    };

    private static GateReport GateReport(string overall, params GateResult[] results)
        => new()
        {
            Scope = "dataset",
            OverallStatus = overall,
            PassCount = System.Array.FindAll(results, r => r.Status == "pass").Length,
            WarnCount = System.Array.FindAll(results, r => r.Status == "warn").Length,
            BlockCount = System.Array.FindAll(results, r => r.Status == "block").Length,
            Results = results,
        };

    // ---- The signal-less nodes stay NEUTRAL (the core honesty invariant) --------------------

    [Fact]
    public void FreshProject_AllNodesHonest_NeutralOrLocked()
    {
        var vm = new MainWindowViewModel();

        // No authored rows, no quality run, no gate run, no evaluation -> every derived node is neutral.
        Assert.Equal("neutral", vm.AuthorNodeStatus);
        Assert.Equal("neutral", vm.QualityNodeStatus);
        Assert.Equal("neutral", vm.GateNodeStatus);
        Assert.Equal("neutral", vm.EvaluateNodeStatus);

        // Validate + Split have no reliable "passed / generated" signal, so they are neutral by design.
        Assert.Equal("neutral", vm.ValidateNodeStatus);
        Assert.Equal("neutral", vm.SplitNodeStatus);

        // Train is the terminal gated phase: locked, never "done".
        Assert.Equal("locked", vm.TrainNodeStatus);
    }

    [Fact]
    public void ValidateAndSplit_StayNeutral_EvenAfterOtherPhasesRun()
    {
        // No matter what else happens, the two signal-less nodes never light up (no fake status).
        var vm = new MainWindowViewModel();
        vm.SetExamples([Example(1), Example(2)]);
        vm.Quality.ApplyQualityReport(QualityReport());
        vm.ApplyGateReport(GateReport("block",
            new GateResult { Name = "PII", Status = "block", Message = "x" }));

        Assert.Equal("neutral", vm.ValidateNodeStatus);
        Assert.Equal("neutral", vm.SplitNodeStatus);
    }

    // ---- Author: done once the dataset has rows ---------------------------------------------

    [Fact]
    public void Author_DoneWhenRowsExist_NeutralWhenEmpty()
    {
        var vm = new MainWindowViewModel();
        Assert.Equal("neutral", vm.AuthorNodeStatus);

        vm.SetExamples([Example(1)]);
        Assert.Equal("done", vm.AuthorNodeStatus);

        vm.SetExamples([]); // dataset emptied again -> honestly back to neutral
        Assert.Equal("neutral", vm.AuthorNodeStatus);
    }

    // ---- Quality: colour FOLLOWS QualityStatusColor -----------------------------------------

    [Theory]
    [InlineData(0, 0, "#15803D", "done")] // clean  -> green
    [InlineData(3, 0, "#B45309", "warn")] // issues -> amber
    [InlineData(0, 2, "#B91C1C", "bad")]  // PII    -> red
    public void Quality_FollowsQualityStatusColor(int empty, int pii, string expectedColor, string expectedNode)
    {
        var vm = new MainWindowViewModel();
        vm.Quality.ApplyQualityReport(QualityReport(empty: empty, pii: pii));

        // The node status is derived from the very colour the Quality panel publishes.
        Assert.True(vm.Quality.HasQualityMetrics);
        Assert.Equal(expectedColor, vm.Quality.QualityStatusColor);
        Assert.Equal(expectedNode, vm.QualityNodeStatus);
    }

    [Fact]
    public void Quality_NeutralBeforeAnyRun()
    {
        var vm = new MainWindowViewModel();
        Assert.False(vm.Quality.HasQualityMetrics);
        Assert.Equal("neutral", vm.QualityNodeStatus);
    }

    // ---- Gate: real problems only; a clean/absent gate is NEVER "done" ----------------------

    [Fact]
    public void Gate_BlockIsBad_WarnIsWarn()
    {
        var vm = new MainWindowViewModel();

        vm.ApplyGateReport(GateReport("block",
            new GateResult { Name = "PII", Status = "block", Message = "found api_key" }));
        Assert.Equal("bad", vm.GateNodeStatus);

        vm.ApplyGateReport(GateReport("warn",
            new GateResult { Name = "Quality", Status = "warn", Message = "thin" }));
        Assert.Equal("warn", vm.GateNodeStatus);
    }

    [Fact]
    public void Gate_CleanPassStaysNeutral_NeverDone()
    {
        // A clean gate produces no Problems badge; "ran-clean" and "not-run" are indistinguishable, so
        // the node stays neutral rather than faking a green pass (a clean gate is not approval).
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(GateReport("pass",
            new GateResult { Name = "Schema", Status = "pass", Message = "ok" }));

        Assert.False(vm.HasProblemsBadge);
        Assert.Equal("neutral", vm.GateNodeStatus);
    }

    [Fact]
    public void Gate_ResetProblems_ReturnsToNeutral()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyGateReport(GateReport("block",
            new GateResult { Name = "PII", Status = "block", Message = "x" }));
        Assert.Equal("bad", vm.GateNodeStatus);

        vm.ResetProblems();
        Assert.Equal("neutral", vm.GateNodeStatus);
    }

    // ---- Evaluate: done once results exist ---------------------------------------------------

    [Fact]
    public void Evaluate_DoneWhenResultsExist_NeutralWhenNone()
    {
        var vm = new MainWindowViewModel();
        Assert.Equal("neutral", vm.EvaluateNodeStatus);

        vm.Evaluation.EvaluationResults.Add(
            new EvaluationExampleResult { ExampleId = "1", Score = 85, Passed = true });
        Assert.Equal("done", vm.EvaluateNodeStatus);

        vm.Evaluation.EvaluationResults.Clear();
        Assert.Equal("neutral", vm.EvaluateNodeStatus);
    }

    // ---- Train: always locked ----------------------------------------------------------------

    [Fact]
    public void Train_IsAlwaysLocked()
    {
        var vm = new MainWindowViewModel();
        vm.SetExamples([Example(1)]);
        vm.Evaluation.EvaluationResults.Add(
            new EvaluationExampleResult { ExampleId = "1", Score = 90, Passed = true });

        Assert.Equal("locked", vm.TrainNodeStatus);
    }

    // ---- Change notification (so the strip repaints when the signal changes) ----------------

    [Fact]
    public void NodeStatus_RaisesPropertyChanged_WhenSignalsChange()
    {
        var vm = new MainWindowViewModel();
        var raised = new HashSet<string>();
        vm.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is not null)
            {
                raised.Add(e.PropertyName);
            }
        };

        vm.SetExamples([Example(1)]);
        vm.Quality.ApplyQualityReport(QualityReport(empty: 1));
        vm.ApplyGateReport(GateReport("block",
            new GateResult { Name = "PII", Status = "block", Message = "x" }));
        vm.Evaluation.EvaluationResults.Add(
            new EvaluationExampleResult { ExampleId = "1", Score = 10, Passed = false });

        Assert.Contains(nameof(vm.AuthorNodeStatus), raised);
        Assert.Contains(nameof(vm.QualityNodeStatus), raised);
        Assert.Contains(nameof(vm.GateNodeStatus), raised);
        Assert.Contains(nameof(vm.EvaluateNodeStatus), raised);
    }
}
