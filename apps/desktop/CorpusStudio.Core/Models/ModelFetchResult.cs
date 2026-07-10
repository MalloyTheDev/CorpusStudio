using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>The outcome of downloading a base model from the Hugging Face Hub — the desktop mirror of
/// the engine's <c>model-fetch</c> result (<c>corpus_studio/training/model_fetch.py</c>). The download
/// is resumable; the important field is <see cref="License"/> / <see cref="LicensePermissive"/> — the
/// *base model's* license governs what you may do with anything trained from it, so a non-permissive or
/// unknown license lands in <see cref="Warnings"/> (fail-closed). A pickle-only (`.bin`) model also
/// warns (loading it can run arbitrary code).</summary>
public sealed class ModelFetchResult
{
    [JsonPropertyName("repo_id")]
    public string RepoId { get; init; } = string.Empty;

    [JsonPropertyName("revision")]
    public string? Revision { get; init; }

    [JsonPropertyName("local_path")]
    public string LocalPath { get; init; } = string.Empty;

    [JsonPropertyName("license")]
    public string? License { get; init; }

    [JsonPropertyName("license_permissive")]
    public bool LicensePermissive { get; init; }

    [JsonPropertyName("weight_files")]
    public List<string> WeightFiles { get; init; } = [];

    [JsonPropertyName("total_size_mb")]
    public double TotalSizeMb { get; init; }

    [JsonPropertyName("warnings")]
    public List<string> Warnings { get; init; } = [];
}
