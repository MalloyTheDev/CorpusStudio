namespace CorpusStudio.Desktop.Models;

public sealed record DatasetProject(
    string Id,
    string Name,
    string SchemaId,
    DateTime CreatedAt,
    DateTime UpdatedAt
);
