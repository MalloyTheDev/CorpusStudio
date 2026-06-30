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
    IReadOnlyList<DatasetField> Fields
);

public sealed record DatasetField(
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("type")]
    string Type,
    [property: JsonPropertyName("required")]
    bool Required
);
