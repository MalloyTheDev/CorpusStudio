using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>Deserialized result of the engine `gate-run` command.</summary>
public sealed class GateReport
{
    [JsonPropertyName("scope")]
    public string Scope { get; init; } = string.Empty;

    [JsonPropertyName("target")]
    public string Target { get; init; } = string.Empty;

    [JsonPropertyName("overall_status")]
    public string OverallStatus { get; init; } = string.Empty;

    [JsonPropertyName("pass_count")]
    public int PassCount { get; init; }

    [JsonPropertyName("warn_count")]
    public int WarnCount { get; init; }

    [JsonPropertyName("block_count")]
    public int BlockCount { get; init; }

    [JsonPropertyName("results")]
    public IReadOnlyList<GateResult> Results { get; init; } = [];

    /// <summary>Renders the AI Assist candidate gate for display. A pre-review signal
    /// only — never approval. Three honest states so a null gate is never a fake pass
    /// and never silent. Pure/testable, and used by BOTH the run-result view and the
    /// persisted review-queue item so the two views cannot diverge.</summary>
    public static string RenderCandidateGate(GateReport? gate, bool hasSuggestedContent)
    {
        if (gate is null)
        {
            return hasSuggestedContent
                // Content was proposed but no line was a gate-able JSON object.
                ? "Candidate gate: not run — the suggested content was not gate-able (see warnings)."
                // No candidate rows at all -> nothing to gate (never a fake pass).
                : "Candidate gate: n/a — no candidate rows to gate.";
        }

        var status = gate.OverallStatus ?? string.Empty;
        // Only an explicit "pass" earns the green check. An unknown/empty status — an
        // older persisted queue item, a future engine status, or a malformed payload —
        // gets a neutral marker so it can never read as a fake pass (mirrors StatusColor,
        // whose catch-all is neutral gray, never green).
        var icon = status.ToLowerInvariant() switch
        {
            "block" => "⛔",
            "warn" => "⚠",
            "pass" => "✅",
            _ => "•",
        };
        var displayStatus = string.IsNullOrWhiteSpace(status) ? "UNKNOWN" : status.ToUpperInvariant();

        var lines = new List<string>
        {
            $"{icon} Candidate gate: {displayStatus} "
            + $"({gate.PassCount} pass, {gate.WarnCount} warn, {gate.BlockCount} block)",
            "— a pre-review signal, not approval; you still review, validate, and save "
            + "(verdict is on the generated candidate, before your edits).",
        };

        foreach (var result in OrderBlockFirst(gate.Results))
        {
            var mark = (result.Status ?? string.Empty).ToLowerInvariant() switch
            {
                "block" => "[BLOCK]",
                "warn" => "[WARN]",
                "pass" => "[PASS]",
                _ => "[?]",
            };
            lines.Add($"{mark} {result.Name}: {result.Message}");
            if (!string.Equals(result.Status, "pass", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(result.Repair))
            {
                lines.Add($"    fix: {result.Repair}");
            }
        }

        return string.Join(Environment.NewLine, lines);
    }

    // Block first, then warn, then pass; stable within a rank.
    private static IEnumerable<GateResult> OrderBlockFirst(IReadOnlyList<GateResult> results)
    {
        return results.OrderBy(result => (result.Status ?? string.Empty).ToLowerInvariant() switch
        {
            "block" => 0,
            "warn" => 1,
            _ => 2,
        });
    }

    /// <summary>Foreground hex for a gate status. Null/unknown/empty is neutral GRAY —
    /// never green — so an absent or unrecognized verdict can never read as a pass.
    /// Case-insensitive; pure/testable (mirrors DebtReport.GradeColor).</summary>
    public static string StatusColor(string? status)
    {
        return (status ?? string.Empty).ToLowerInvariant() switch
        {
            "pass" => "#16A34A",   // green
            "warn" => "#D97706",   // amber
            "block" => "#DC2626",  // red
            _ => "#64748B",        // gray — null/empty/unknown: never green
        };
    }
}

public sealed class GateResult
{
    [JsonPropertyName("gate_id")]
    public string GateId { get; init; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; init; } = string.Empty;

    [JsonPropertyName("observed")]
    public string Observed { get; init; } = string.Empty;

    [JsonPropertyName("expected")]
    public string Expected { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; init; } = string.Empty;

    [JsonPropertyName("repair")]
    public string? Repair { get; init; }
}
