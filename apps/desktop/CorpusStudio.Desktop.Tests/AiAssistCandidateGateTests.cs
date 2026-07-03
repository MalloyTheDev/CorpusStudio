using System.Text.Json;
using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>v1.2.1 desktop surfacing of the AI Assist candidate gate. The gate is a
/// pre-review signal only: never auto-accepts (a clean gate is not approval) and never
/// auto-rejects (a block is confirm-then-allow); null is shown honestly, never as a
/// fake pass; and the run view and persisted queue item share one renderer.</summary>
public sealed class AiAssistCandidateGateTests
{
    private static GateReport Gate(string overall, params GateResult[] results)
    {
        return new GateReport
        {
            Scope = "dataset",
            Target = "ai_assist_candidates",
            OverallStatus = overall,
            PassCount = System.Array.FindAll(results, r => r.Status == "pass").Length,
            WarnCount = System.Array.FindAll(results, r => r.Status == "warn").Length,
            BlockCount = System.Array.FindAll(results, r => r.Status == "block").Length,
            Results = results,
        };
    }

    // ---- Shared renderer: three honest states ------------------------------------

    [Fact]
    public void RenderCandidateGate_Pass_IsSignalNotApproval()
    {
        var text = GateReport.RenderCandidateGate(
            Gate("pass", new GateResult { GateId = "schema", Name = "Schema", Status = "pass", Message = "ok" }),
            hasSuggestedContent: true);

        Assert.Contains("Candidate gate: PASS", text);
        Assert.Contains("not approval", text);          // a clean gate is never approval
        Assert.Contains("before your edits", text);      // stale-after-edit framing
        Assert.DoesNotContain("approved", text);
        Assert.DoesNotContain("safe to save", text);
    }

    [Fact]
    public void RenderCandidateGate_Block_IsBlockFirstWithRepair()
    {
        var text = GateReport.RenderCandidateGate(
            Gate("block",
                new GateResult { GateId = "schema", Name = "Schema", Status = "pass", Message = "ok" },
                new GateResult { GateId = "pii", Name = "PII / secret", Status = "block", Message = "Found api_key.", Repair = "Remove keys." }),
            hasSuggestedContent: true);

        Assert.Contains("Candidate gate: BLOCK", text);
        Assert.Contains("[BLOCK] PII / secret: Found api_key.", text);
        Assert.Contains("fix: Remove keys.", text);
        // Block-first ordering: the block result appears before the pass result.
        Assert.True(text.IndexOf("[BLOCK]", System.StringComparison.Ordinal)
                    < text.IndexOf("[PASS]", System.StringComparison.Ordinal));
    }

    [Fact]
    public void RenderCandidateGate_NullNoContent_SaysNoRowsToGate()
    {
        var text = GateReport.RenderCandidateGate(null, hasSuggestedContent: false);
        Assert.Contains("n/a", text);
        Assert.Contains("no candidate rows to gate", text);
    }

    [Fact]
    public void RenderCandidateGate_NullWithContent_SaysNotRunNeverPass()
    {
        var text = GateReport.RenderCandidateGate(null, hasSuggestedContent: true);
        Assert.Contains("not run", text);
        Assert.Contains("see warnings", text);
        Assert.DoesNotContain("PASS", text);   // ungateable content must never read as a pass
    }

    // ---- Status color: never green for null/unknown ------------------------------

    [Theory]
    [InlineData("pass", "#16A34A")]
    [InlineData("warn", "#D97706")]
    [InlineData("block", "#DC2626")]
    [InlineData("BLOCK", "#DC2626")]   // case-insensitive
    [InlineData("", "#64748B")]
    [InlineData("weird", "#64748B")]
    public void StatusColor_MapsStatus(string status, string expected)
    {
        Assert.Equal(expected, GateReport.StatusColor(status));
    }

    [Fact]
    public void StatusColor_NullIsNeutralGray_NeverGreen()
    {
        Assert.Equal("#64748B", GateReport.StatusColor(null));
        Assert.NotEqual("#16A34A", GateReport.StatusColor(null));
    }

    // ---- Serialization + back-compat ---------------------------------------------

    [Fact]
    public void AiAssistRunResult_DeserializesCandidateGate()
    {
        const string json = """
        {"schema_id":"instruction","action":"draft-example","model":"m",
         "suggested_jsonl":"{}",
         "candidate_gate":{"scope":"dataset","target":"ai_assist_candidates",
           "overall_status":"block","pass_count":3,"warn_count":0,"block_count":1,"results":[]}}
        """;

        var result = JsonSerializer.Deserialize<AiAssistRunResult>(json);

        Assert.NotNull(result);
        Assert.NotNull(result!.CandidateGate);
        Assert.Equal("block", result.CandidateGate!.OverallStatus);
        Assert.Equal(1, result.CandidateGate.BlockCount);
    }

    [Fact]
    public void AiAssistRunResult_OldPayloadWithoutGate_IsNull()
    {
        const string json = """{"schema_id":"instruction","action":"review","model":"m","suggested_jsonl":""}""";
        var result = JsonSerializer.Deserialize<AiAssistRunResult>(json);
        Assert.NotNull(result);
        Assert.Null(result!.CandidateGate);   // absent -> null, never a fake pass
    }

    [Fact]
    public void FromRunResult_CopiesCandidateGate()
    {
        var result = new AiAssistRunResult
        {
            SchemaId = "instruction",
            Action = "draft-example",
            Model = "m",
            SuggestedJsonl = "{}",
            CandidateGate = Gate("block", new GateResult { GateId = "pii", Name = "PII", Status = "block", Message = "x" }),
        };

        var item = AiAssistReviewQueueItem.FromRunResult("source", result);

        Assert.NotNull(item.CandidateGate);
        Assert.Equal("block", item.CandidateGate!.OverallStatus);
        // The persisted queue item renders the gate through the SAME shared renderer.
        Assert.Contains("Candidate gate: BLOCK", item.DetailText);
    }

    [Fact]
    public void QueueItem_OldPersistedItemWithoutGate_RendersNoGateNotFakePass()
    {
        const string json = """
        {"review_id":"r1","review_state":"review_required","action":"draft-example","model":"m",
         "source_draft":"s","suggested_jsonl":"{}"}
        """;

        var item = JsonSerializer.Deserialize<AiAssistReviewQueueItem>(json);

        Assert.NotNull(item);
        Assert.Null(item!.CandidateGate);
        // hasSuggestedContent true + null gate -> "not run", never a green PASS.
        Assert.Contains("not run", item.DetailText);
        Assert.DoesNotContain("Candidate gate: PASS", item.DetailText);
    }

    // ---- Confirm-on-block signal: never-auto-reject decision hook -----------------

    [Fact]
    public void SelectedBlocks_TrueOnlyWhenRunGateBlocks()
    {
        var vm = new MainWindowViewModel();

        vm.ApplyAiAssistRunResult(RunResult("pass"));
        Assert.False(vm.SelectedAiAssistCandidateGateBlocks);

        vm.ApplyAiAssistRunResult(RunResult("warn"));
        Assert.False(vm.SelectedAiAssistCandidateGateBlocks);

        vm.ApplyAiAssistRunResult(RunResult("block"));
        Assert.True(vm.SelectedAiAssistCandidateGateBlocks);
    }

    [Fact]
    public void SelectedBlocks_IsCaseInsensitive()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyAiAssistRunResult(RunResult("BLOCK"));
        Assert.True(vm.SelectedAiAssistCandidateGateBlocks);
    }

    [Fact]
    public void SelectedBlocks_SelectedQueueItemGateWinsOverRunGate()
    {
        var vm = new MainWindowViewModel();
        // Fresh run passed...
        vm.ApplyAiAssistRunResult(RunResult("pass"));
        Assert.False(vm.SelectedAiAssistCandidateGateBlocks);

        // ...but the selected queued item's gate blocks -> the item's gate wins.
        var blocked = new AiAssistReviewQueueItem
        {
            Action = "draft-example",
            Model = "m",
            SuggestedJsonl = "{}",
            CandidateGate = Gate("block", new GateResult { GateId = "pii", Name = "PII", Status = "block", Message = "x" }),
        };
        vm.SelectedAiAssistReviewQueueItem = blocked;
        Assert.True(vm.SelectedAiAssistCandidateGateBlocks);

        // Deselect -> falls back to the fresh-run (passing) gate.
        vm.SelectedAiAssistReviewQueueItem = null;
        Assert.False(vm.SelectedAiAssistCandidateGateBlocks);
    }

    // ---- VM header status/color honest across states -----------------------------

    [Fact]
    public void ApplyRunResult_SetsHeaderStatusAndColor_Block()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyAiAssistRunResult(RunResult("block"));

        Assert.Equal("BLOCK", vm.AiAssistCandidateGateStatus);
        Assert.Equal("#DC2626", vm.AiAssistCandidateGateColor);
        Assert.Contains("Candidate gate: BLOCK", vm.AiAssistReviewText);
    }

    [Fact]
    public void ApplyRunResult_NullGateNoContent_HeaderIsNaGray()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyAiAssistRunResult(new AiAssistRunResult { SuggestedJsonl = "", CandidateGate = null });

        Assert.Equal("n/a", vm.AiAssistCandidateGateStatus);
        Assert.Equal("#64748B", vm.AiAssistCandidateGateColor);
        Assert.NotEqual("#16A34A", vm.AiAssistCandidateGateColor);   // never green
    }

    [Fact]
    public void ApplyRunResult_NullGateWithContent_HeaderIsNotRunGray()
    {
        var vm = new MainWindowViewModel();
        vm.ApplyAiAssistRunResult(new AiAssistRunResult { SuggestedJsonl = "{}", CandidateGate = null });

        Assert.Equal("not run", vm.AiAssistCandidateGateStatus);
        Assert.Equal("#64748B", vm.AiAssistCandidateGateColor);
        Assert.Contains("not run", vm.AiAssistReviewText);
    }

    private static AiAssistRunResult RunResult(string overall)
    {
        return new AiAssistRunResult
        {
            SchemaId = "instruction",
            Action = "draft-example",
            Model = "m",
            SuggestedJsonl = "{}",
            CandidateGate = Gate(overall,
                new GateResult { GateId = "schema", Name = "Schema", Status = overall.ToLowerInvariant() == "pass" ? "pass" : overall.ToLowerInvariant(), Message = "x" }),
        };
    }
}
