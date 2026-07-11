using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Json;

namespace CorpusStudio.Desktop.Models;

/// <summary>A per-row data-quality signal derived HONESTLY from the row's real JSON content — never
/// invented. <see cref="Bad"/> = the row is not valid JSON; <see cref="Warn"/> = valid JSON but a
/// recognised instruction/output field is present-yet-empty (an incomplete pair); <see cref="Ok"/> =
/// otherwise (clean).</summary>
public enum ExampleStatus
{
    Ok,
    Warn,
    Bad,
}

/// <summary>One saved dataset row (Examples tab). Carries only what the engine loads —
/// <see cref="RowNumber"/>, a <see cref="Preview"/> line, and the pretty-printed <see cref="Json"/> —
/// so the 3-arg construction used by the engine and tests is unchanged.
///
/// <para>The structured Examples screen (status pills, the detail pane's instruction/output +
/// Tags/Source/Tokens/Added row, the list status icon) reads the derived members below. They are parsed
/// on demand from the real <see cref="Json"/>: absent fields read "—" (<see cref="MissingField"/>) and
/// the status comes from actual content, so nothing is fabricated. Parsing is kept off the record's
/// value identity (equality stays over RowNumber/Preview/Json).</para></summary>
public sealed record SavedExampleItem(int RowNumber, string Preview, string Json)
{
    /// <summary>Placeholder shown for any metadata field the row does not carry — never a fabricated value.</summary>
    public const string MissingField = "—";

    // Preview sentinel the engine writes when a row is not valid JSON (see PythonEngineService).
    private const string InvalidJsonPreview = "Invalid JSON row";

    private static readonly string[] InstructionKeys = ["instruction", "prompt", "question", "text"];
    private static readonly string[] OutputKeys = ["output", "response", "chosen", "completion", "answer"];
    private static readonly string[] TagsKeys = ["tags", "labels", "categories"];
    private static readonly string[] SourceKeys = ["source", "origin", "dataset"];
    private static readonly string[] TokensKeys = ["tokens", "token_count", "n_tokens", "num_tokens"];
    private static readonly string[] AddedKeys = ["added", "created_at", "created", "timestamp", "date"];

    public string Title => $"Example {RowNumber}";

    public string DisplayName => $"{Title}: {Preview}";

    /// <summary>Real per-row quality status derived from <see cref="Json"/>. Parsed on demand.</summary>
    public ExampleStatus Status => ParseStatus();

    public bool IsOk => Status == ExampleStatus.Ok;

    public bool IsWarn => Status == ExampleStatus.Warn;

    public bool IsBad => Status == ExampleStatus.Bad;

    /// <summary>Clean = a valid, complete row. Feeds the "Clean" filter pill + detail badge.</summary>
    public bool IsClean => Status == ExampleStatus.Ok;

    /// <summary>Flagged = anything not clean (empty field or broken JSON). Feeds the "Flagged" pill.</summary>
    public bool IsFlagged => Status != ExampleStatus.Ok;

    /// <summary>Short status word for the detail-pane badge.</summary>
    public string StatusLabel => Status switch
    {
        ExampleStatus.Ok => "Clean",
        ExampleStatus.Warn => "Flagged",
        _ => "Broken",
    };

    /// <summary>The instruction/prompt text for the detail pane, or "—" when the row has none.</summary>
    public string Instruction => FirstNonEmptyString(InstructionKeys) ?? MissingField;

    /// <summary>The output/response text for the detail pane, or "—" when the row has none.</summary>
    public string Output => FirstNonEmptyString(OutputKeys) ?? MissingField;

    public string Tags => ReadTags() ?? MissingField;

    public string Source => FirstNonEmptyString(SourceKeys) ?? MissingField;

    /// <summary>A real token count field only — never an estimate. "—" when the row carries none.</summary>
    public string Tokens => ReadNumber(TokensKeys) ?? MissingField;

    public string Added => FirstNonEmptyString(AddedKeys) ?? MissingField;

    private JsonElement? Root()
    {
        if (Preview == InvalidJsonPreview)
        {
            return null;
        }

        try
        {
            using var document = JsonDocument.Parse(Json);
            // Clone so the value survives the using-scope disposal.
            return document.RootElement.Clone();
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private ExampleStatus ParseStatus()
    {
        if (Preview == InvalidJsonPreview)
        {
            return ExampleStatus.Bad;
        }

        JsonElement obj;
        try
        {
            using var document = JsonDocument.Parse(Json);
            obj = document.RootElement.Clone();
        }
        catch (JsonException)
        {
            // The row is not valid JSON — broken, regardless of the preview text.
            return ExampleStatus.Bad;
        }

        if (obj.ValueKind != JsonValueKind.Object)
        {
            // A valid non-object payload we can't quality-check → treat as clean.
            return ExampleStatus.Ok;
        }

        // Flag rows with a present-but-empty recognised instruction/output field (an incomplete pair),
        // or a messages array that has a user turn but no non-empty assistant reply.
        if (HasPresentButEmpty(obj, OutputKeys)
            || HasPresentButEmpty(obj, InstructionKeys)
            || HasEmptyAssistantTurn(obj))
        {
            return ExampleStatus.Warn;
        }

        return ExampleStatus.Ok;
    }

    private static bool HasPresentButEmpty(JsonElement obj, string[] keys)
    {
        foreach (var key in keys)
        {
            if (obj.TryGetProperty(key, out var value)
                && value.ValueKind == JsonValueKind.String
                && string.IsNullOrWhiteSpace(value.GetString()))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasEmptyAssistantTurn(JsonElement obj)
    {
        if (!obj.TryGetProperty("messages", out var messages) || messages.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        var sawUser = false;
        var sawAssistantContent = false;
        foreach (var message in messages.EnumerateArray())
        {
            if (message.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            var role = message.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String
                ? r.GetString()
                : null;
            var content = message.TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;

            if (role == "user")
            {
                sawUser = true;
            }
            else if (role == "assistant" && !string.IsNullOrWhiteSpace(content))
            {
                sawAssistantContent = true;
            }
        }

        return sawUser && !sawAssistantContent;
    }

    private string? FirstNonEmptyString(string[] keys)
    {
        if (Root() is not { ValueKind: JsonValueKind.Object } obj)
        {
            return null;
        }

        foreach (var key in keys)
        {
            if (obj.TryGetProperty(key, out var value)
                && value.ValueKind == JsonValueKind.String
                && !string.IsNullOrWhiteSpace(value.GetString()))
            {
                return value.GetString();
            }
        }

        // Chat schema: fall back to the first matching-role message content.
        if (obj.TryGetProperty("messages", out var messages) && messages.ValueKind == JsonValueKind.Array)
        {
            var wantAssistant = ReferenceEquals(keys, OutputKeys);
            foreach (var message in messages.EnumerateArray())
            {
                if (message.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }

                var role = message.TryGetProperty("role", out var r) && r.ValueKind == JsonValueKind.String
                    ? r.GetString()
                    : null;
                var matches = wantAssistant ? role == "assistant" : role is "user" or "system";
                if (matches
                    && message.TryGetProperty("content", out var content)
                    && content.ValueKind == JsonValueKind.String
                    && !string.IsNullOrWhiteSpace(content.GetString()))
                {
                    return content.GetString();
                }
            }
        }

        return null;
    }

    private string? ReadTags()
    {
        if (Root() is not { ValueKind: JsonValueKind.Object } obj)
        {
            return null;
        }

        foreach (var key in TagsKeys)
        {
            if (!obj.TryGetProperty(key, out var value))
            {
                continue;
            }

            if (value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString()))
            {
                return value.GetString();
            }

            if (value.ValueKind == JsonValueKind.Array)
            {
                var parts = value.EnumerateArray()
                    .Where(item => item.ValueKind == JsonValueKind.String)
                    .Select(item => item.GetString())
                    .Where(text => !string.IsNullOrWhiteSpace(text))
                    .ToList();
                if (parts.Count > 0)
                {
                    return string.Join(" · ", parts);
                }
            }
        }

        return null;
    }

    private string? ReadNumber(string[] keys)
    {
        if (Root() is not { ValueKind: JsonValueKind.Object } obj)
        {
            return null;
        }

        foreach (var key in keys)
        {
            if (obj.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number)
            {
                return value.TryGetInt64(out var whole)
                    ? whole.ToString(CultureInfo.InvariantCulture)
                    : value.GetDouble().ToString(CultureInfo.InvariantCulture);
            }
        }

        return null;
    }
}
