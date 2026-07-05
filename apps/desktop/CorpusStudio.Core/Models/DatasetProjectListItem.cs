namespace CorpusStudio.Desktop.Models;

public sealed record DatasetProjectListItem(DatasetProject Project, string ProjectPath)
{
    public string Id => Project.Id;

    public string Name => Project.Name;

    public string SchemaId => Project.SchemaId;

    public string DisplayName => $"{Name} ({Id})";
}
