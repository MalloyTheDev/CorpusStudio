using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed record DatasetProject(
    [property: JsonPropertyName("id")]
    string Id,
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("schema_id")]
    string SchemaId,
    [property: JsonPropertyName("created_at")]
    DateTime CreatedAt,
    [property: JsonPropertyName("updated_at")]
    DateTime UpdatedAt
);
