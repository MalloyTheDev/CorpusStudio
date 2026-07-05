using System.IO;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class ImportQuarantineItem
{
    [JsonPropertyName("source_path")]
    public string SourcePath { get; init; } = string.Empty;

    [JsonPropertyName("row_number")]
    public int RowNumber { get; init; }

    [JsonPropertyName("raw")]
    public string Raw { get; init; } = string.Empty;

    [JsonPropertyName("errors")]
    public List<ValidationIssue> Errors { get; init; } = [];

    [JsonIgnore]
    public string QuarantinePath { get; set; } = string.Empty;

    [JsonIgnore]
    public string DisplayName
    {
        get
        {
            var fileName = string.IsNullOrWhiteSpace(QuarantinePath)
                ? "quarantine"
                : Path.GetFileName(QuarantinePath);
            var message = Errors.FirstOrDefault()?.Message ?? "Rejected row";
            return $"{fileName} row {RowNumber}: {message}";
        }
    }

    [JsonIgnore]
    public string DetailText
    {
        get
        {
            var errors = Errors.Count == 0
                ? "Unknown error"
                : string.Join(Environment.NewLine, Errors.Select(error =>
                    string.IsNullOrWhiteSpace(error.Field)
                        ? $"- {error.Message}"
                        : $"- {error.Message} [{error.Field}]"));

            return string.Join(
                Environment.NewLine,
                [
                    $"Quarantine: {QuarantinePath}",
                    $"Source: {SourcePath}",
                    $"Row: {RowNumber}",
                    "",
                    "Errors:",
                    errors,
                    "",
                    "Raw:",
                    Raw,
                ]
            );
        }
    }
}
