namespace CorpusStudio.Desktop.Models;

public sealed record NewProjectRequest(
    string ProjectId,
    string Name,
    string SchemaId,
    string SchemaName
);
