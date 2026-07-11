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

    // ---- Dashboard "Recent Activity" projection ---------------------------------------------------
    // Pure, testable helpers that let the Dashboard feed render a real engine invocation without a
    // separate activity model. The glyph key is a Nocturne icon resource name (resolved to a Path
    // geometry by the view's ActivityGlyphConverter); the message and meta are human renderings of the
    // entry's own fields — nothing is synthesised.

    /// <summary>Nocturne icon resource key for this entry's leading glyph, chosen from the real verb
    /// (and status). See <see cref="GlyphKeyFor"/>.</summary>
    public string ActivityGlyphKey => GlyphKeyFor(Command, Status);

    /// <summary>One-line human summary of the invocation for the activity feed. See <see cref="MessageFor"/>.</summary>
    public string ActivityMessage => MessageFor(Command, Status);

    /// <summary>Right-aligned "when" for the activity row: the entry's real timestamp and (if measured)
    /// its duration — never a fabricated relative time.</summary>
    public string ActivityMeta
    {
        get
        {
            var hasTime = !string.IsNullOrWhiteSpace(Timestamp);
            if (hasTime && DurationMs > 0)
            {
                return $"{Timestamp} · {DurationLabel}";
            }

            if (hasTime)
            {
                return Timestamp;
            }

            return DurationMs > 0 ? DurationLabel : string.Empty;
        }
    }

    /// <summary>Map a real engine verb + outcome to a Phosphor glyph resource key: a failed command
    /// surfaces a warning; otherwise the verb picks a distinctive glyph (quality → broom, import →
    /// import, capture/version → git-commit, gate/validate → check-circle); anything else is the
    /// neutral list glyph. All keys exist in the Avalonia Icons.axaml resource set.</summary>
    public static string GlyphKeyFor(string? command, string? status)
    {
        if ((status ?? string.Empty).Equals("error", StringComparison.OrdinalIgnoreCase))
        {
            return "IcoWarningFill";
        }

        var verb = (command ?? string.Empty).ToLowerInvariant();
        if (verb.Contains("quality"))
        {
            return "IcoBroom";
        }

        if (verb.Contains("import"))
        {
            return "IcoImport";
        }

        if (verb.Contains("capture") || verb.Contains("version") || verb.Contains("commit") || verb.Contains("snapshot"))
        {
            return "IcoGitCommit";
        }

        if (verb.Contains("gate") || verb.Contains("validate"))
        {
            return "IcoCheckCircleFill";
        }

        return "IcoListDashes";
    }

    /// <summary>Human-readable activity line for a real verb, with a failure/cancellation outcome
    /// appended. An unrecognised verb is shown verbatim (never a fabricated label).</summary>
    public static string MessageFor(string? command, string? status)
    {
        var action = FriendlyAction(command);
        return (status ?? string.Empty).ToLowerInvariant() switch
        {
            "error" => action + " — failed",
            "cancelled" => action + " — cancelled",
            _ => action,
        };
    }

    private static string FriendlyAction(string? command)
    {
        var verb = (command ?? string.Empty).ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(verb))
        {
            return "Engine command";
        }

        if (verb.Contains("quality")) return "Ran quality check";
        if (verb.Contains("import")) return "Imported rows";
        if (verb.Contains("gate")) return "Ran gates";
        if (verb.Contains("validate")) return "Validated draft";
        if (verb.Contains("split")) return "Generated splits";
        if (verb.Contains("export")) return "Exported dataset";
        if (verb.Contains("capture") || verb.Contains("version") || verb.Contains("commit")) return "Captured version";
        if (verb.Contains("eval")) return "Ran evaluation";
        if (verb.Contains("assist")) return "Ran AI assist";
        if (verb.Contains("train")) return "Training run";
        return command!.Trim();
    }

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
