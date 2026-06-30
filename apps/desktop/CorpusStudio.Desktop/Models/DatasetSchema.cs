namespace CorpusStudio.Desktop.Models;

public sealed record DatasetSchema(
    string Id,
    string Name,
    string Version,
    IReadOnlyList<DatasetField> Fields
);

public sealed record DatasetField(
    string Name,
    string Type,
    bool Required
);
