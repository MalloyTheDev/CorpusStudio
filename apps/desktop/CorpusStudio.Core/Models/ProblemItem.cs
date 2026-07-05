namespace CorpusStudio.Desktop.Models;

/// <summary>One row in the Workspace Problems panel (v1.2.6). A structured, display-ready
/// projection of a single gate finding (<see cref="GateResult"/>) so the panel can render a
/// scannable list instead of a text blob. Pure/testable — the icon, colour, and sort rank are
/// all derived from the gate status and reuse <see cref="GateReport.StatusColor"/> so the
/// panel can never disagree with the rest of the UI about what a status means.</summary>
public sealed class ProblemItem
{
    public string Severity { get; init; } = string.Empty;
    public string SeverityIcon { get; init; } = string.Empty;
    public string SeverityColor { get; init; } = GateReport.StatusColor(null);
    public string Name { get; init; } = string.Empty;
    public string Message { get; init; } = string.Empty;
    public string Fix { get; init; } = string.Empty;
    public bool HasFix => !string.IsNullOrWhiteSpace(Fix);

    /// <summary>Sort key: blocks before warns before everything else (stable within a rank).</summary>
    public int SeverityRank { get; init; } = 2;

    /// <summary>Project a gate result into a problem row. Only block/warn results are
    /// "problems"; a pass is not a problem (callers filter passes out before building the
    /// panel list). The status match is case-insensitive; an unknown status is treated as a
    /// neutral finding (never green), mirroring <see cref="GateReport.StatusColor"/>.</summary>
    public static ProblemItem FromGateResult(GateResult result)
    {
        var status = (result.Status ?? string.Empty).ToLowerInvariant();
        var (icon, rank) = status switch
        {
            "block" => ("⛔", 0),
            "warn" => ("⚠", 1),
            "pass" => ("✅", 2),
            _ => ("•", 2),
        };

        return new ProblemItem
        {
            Severity = string.IsNullOrWhiteSpace(status) ? "unknown" : status,
            SeverityIcon = icon,
            SeverityColor = GateReport.StatusColor(result.Status),
            Name = result.Name,
            Message = result.Message,
            Fix = result.Repair ?? string.Empty,
            SeverityRank = rank,
        };
    }

    /// <summary>True when a gate result represents an actionable problem (block or warn).
    /// A pass — or an unknown status — is not surfaced as a problem.</summary>
    public static bool IsProblem(GateResult result)
    {
        var status = (result.Status ?? string.Empty).ToLowerInvariant();
        return status is "block" or "warn";
    }
}
