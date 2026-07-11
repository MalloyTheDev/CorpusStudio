using System.Globalization;

namespace CorpusStudio.Desktop.Models;

/// <summary>One row in the right-panel Quality metric grid (v1.2.12): a labeled quality count
/// with a value and a status colour/icon, so the panel reads as a scannable list instead of a
/// wall of text. Pure/testable — the icon and colour are derived from the count and mirror the
/// neutral-for-info / green-for-clean convention used elsewhere.</summary>
public sealed class QualityMetric
{
    public string Label { get; init; } = string.Empty;
    public string Value { get; init; } = string.Empty;
    public string Severity { get; init; } = "info";     // info | ok | warn | problem
    public string StatusColor { get; init; } = "#64748B";
    public string StatusIcon { get; init; } = "•";

    /// <summary>Short uppercase badge for the Quality-screen metric card (Avalonia fidelity slice B),
    /// derived from the severity so a nonzero PII/secret reads HIGH, other issues WARN, a clean count
    /// OK, and an informational count INFO — mirrors the neutral/green/amber/red convention.</summary>
    public string SeverityBadge => Severity switch
    {
        "problem" => "HIGH",
        "warn" => "WARN",
        "ok" => "OK",
        _ => "INFO",
    };

    /// <summary>A neutral informational metric (e.g. the example count) — never an issue.</summary>
    public static QualityMetric Info(string label, int value) => new()
    {
        Label = label,
        Value = value.ToString(CultureInfo.InvariantCulture),
        Severity = "info",
        StatusColor = "#64748B",   // gray
        StatusIcon = "•",
    };

    /// <summary>An issue count: 0 reads as clean (green ✓); a nonzero count reads as attention.
    /// A <paramref name="severe"/> metric (PII / secrets) escalates a nonzero count to a red
    /// problem instead of an amber warning.</summary>
    public static QualityMetric Issue(string label, int count, bool severe = false)
    {
        var (severity, color, icon) = count == 0
            ? ("ok", "#16A34A", "✓")                         // green
            : severe
                ? ("problem", "#DC2626", "⛔")                // red
                : ("warn", "#D97706", "⚠");                  // amber

        return new QualityMetric
        {
            Label = label,
            Value = count.ToString(CultureInfo.InvariantCulture),
            Severity = severity,
            StatusColor = color,
            StatusIcon = icon,
        };
    }
}
