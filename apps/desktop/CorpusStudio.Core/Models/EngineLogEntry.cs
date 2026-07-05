namespace CorpusStudio.Desktop.Models;

/// <summary>One line in the Workspace Output / Logs panel (v1.2.7): a record of a single
/// engine CLI invocation (verb, argument summary, outcome, duration, and any stderr tail on
/// failure). Ephemeral — held in an in-memory ring buffer, never persisted. The status icon
/// and colour are pure/testable and mirror the neutral-for-unknown convention used elsewhere
/// (see <see cref="GateReport.StatusColor"/>). API keys are passed to the engine via the
/// environment, not argv, so the argument summary carries no secrets.</summary>
public sealed class EngineLogEntry
{
    public string Timestamp { get; init; } = string.Empty;   // "HH:mm:ss", stamped by the service
    public string Command { get; init; } = string.Empty;     // the CLI verb, e.g. "gate-run"
    public string ArgsSummary { get; init; } = string.Empty; // truncated argv tail (flags/paths)
    public string Status { get; init; } = "ok";              // ok | error | cancelled
    public long DurationMs { get; init; }
    public string Detail { get; init; } = string.Empty;      // stderr tail, only on error

    public string StatusIcon => StatusIconFor(Status);
    public string StatusColor => StatusColorFor(Status);
    public bool HasDetail => !string.IsNullOrWhiteSpace(Detail);
    public string DurationLabel => $"{DurationMs} ms";

    public static string StatusIconFor(string status) => (status ?? string.Empty).ToLowerInvariant() switch
    {
        "ok" => "✓",
        "error" => "✕",
        "cancelled" => "⊘",
        _ => "•",
    };

    /// <summary>Foreground hex for a log status. Unknown/empty is neutral gray — never green —
    /// mirroring <see cref="GateReport.StatusColor"/>.</summary>
    public static string StatusColorFor(string status) => (status ?? string.Empty).ToLowerInvariant() switch
    {
        "ok" => "#16A34A",       // green
        "error" => "#DC2626",    // red
        "cancelled" => "#64748B", // gray
        _ => "#64748B",
    };

    private const int MaxArgsLength = 96;
    private const int MaxDetailLength = 400;

    /// <summary>Build a log entry from an engine invocation. Pure — the caller passes the
    /// timestamp string so this stays testable (no clock). <paramref name="argv"/> is the CLI
    /// argument vector as passed to the engine (argv[0] is the verb). The stderr tail is only
    /// kept when the command did not succeed.</summary>
    public static EngineLogEntry FromInvocation(
        string[] argv,
        int exitCode,
        long durationMs,
        string? stderr,
        string timestamp,
        bool cancelled = false)
    {
        argv ??= [];
        var command = argv.Length > 0 && !string.IsNullOrWhiteSpace(argv[0]) ? argv[0] : "(engine)";
        var argsSummary = argv.Length > 1 ? string.Join(" ", argv[1..]) : string.Empty;
        if (argsSummary.Length > MaxArgsLength)
        {
            argsSummary = argsSummary[..MaxArgsLength] + "…";
        }

        var status = cancelled ? "cancelled" : (exitCode == 0 ? "ok" : "error");

        var detail = string.Empty;
        if (status != "ok" && !string.IsNullOrWhiteSpace(stderr))
        {
            detail = stderr.Trim();
            if (detail.Length > MaxDetailLength)
            {
                // Keep the tail — the actionable error is usually at the end of a traceback.
                detail = "…" + detail[^MaxDetailLength..];
            }
        }

        return new EngineLogEntry
        {
            Timestamp = timestamp,
            Command = command,
            ArgsSummary = argsSummary,
            Status = status,
            DurationMs = durationMs,
            Detail = detail,
        };
    }
}
