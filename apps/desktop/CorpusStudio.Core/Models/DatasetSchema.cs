using System.Text.Json;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed record DatasetSchema(
    [property: JsonPropertyName("id")]
    string Id,
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("version")]
    string Version,
    [property: JsonPropertyName("fields")]
    IReadOnlyList<DatasetField> Fields,
    [property: JsonPropertyName("description")]
    string? Description = null,
    [property: JsonPropertyName("example")]
    JsonElement? Example = null
)
{
    private static readonly JsonSerializerOptions ExampleSerializerOptions = new()
    {
        WriteIndented = true
    };

    /// <summary>The schema's canonical example row, pretty-printed for the editor.</summary>
    [JsonIgnore]
    public string ExampleText => Example is { } element
        ? JsonSerializer.Serialize(element, ExampleSerializerOptions)
        : string.Empty;
}

public sealed record DatasetField(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("type")]
    string Type,
    [property: JsonPropertyName("required")]
    bool Required
);
