using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>The outcome of merging a trained LoRA adapter into its base — the desktop mirror of the
/// engine's <c>train-merge</c> result (<c>corpus_studio/training/merge.py</c>). <see cref="Strategy"/>
/// is the one that actually ran (gpu | cpu | adapter-only); when it is <c>adapter-only</c>,
/// <see cref="Merged"/> is false and <see cref="OutputPath"/> is the adapter dir (serve base+adapter
/// unmerged). The <c>auto</c> strategy walks gpu → cpu → adapter-only, so this always resolves.</summary>
public sealed class MergeResult
{
    [JsonPropertyName("strategy")]
    public string Strategy { get; init; } = string.Empty;

    [JsonPropertyName("merged")]
    public bool Merged { get; init; }

    [JsonPropertyName("output_path")]
    public string OutputPath { get; init; } = string.Empty;

    [JsonPropertyName("base_model")]
    public string BaseModel { get; init; } = string.Empty;

    [JsonPropertyName("notes")]
    public List<string> Notes { get; init; } = [];
}
