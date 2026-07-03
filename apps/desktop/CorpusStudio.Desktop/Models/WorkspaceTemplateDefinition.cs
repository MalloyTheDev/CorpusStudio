namespace CorpusStudio.Desktop.Models;

/// <summary>The kind of project scaffold a template produces (v1.2.3 Workspace System,
/// slice 3). Kept separate from the dataset <em>schema</em> — one template applies to any
/// schema (one universal workspace, many schemas).</summary>
public enum WorkspaceTemplateKind
{
    Empty,
    Minimal,
    Standard,
    Full,
    SchemaSpecific,
}

/// <summary>A New Dataset Project template: a named, described scaffold recipe. The
/// concrete folders/files it produces are computed by
/// <see cref="Services.ProjectTemplateService.BuildPlan"/> so the wizard can preview them
/// before anything is written to disk.</summary>
public sealed class WorkspaceTemplateDefinition
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public string Description { get; init; } = string.Empty;
    public WorkspaceTemplateKind Kind { get; init; }
}
