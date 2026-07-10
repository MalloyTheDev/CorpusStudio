using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>What the local machine can do for the FIRST-PARTY trainer — the desktop mirror of the
/// engine's <c>train-check --json</c> report (<c>corpus_studio/training/environment.py</c>). Counts and
/// flags only, no secrets. Two readiness levels: <see cref="CpuToyReady"/> (the tiny CPU smoke path can
/// run) and <see cref="Ready"/> (a real 4-bit QLoRA GPU run is possible). Honest by construction — a
/// missing dep or absent GPU degrades the verdict, never crashes the probe.</summary>
public sealed class TrainingRuntimeReport
{
    [JsonPropertyName("installed")]
    public Dictionary<string, string?> Installed { get; init; } = new();

    [JsonPropertyName("missing")]
    public List<string> Missing { get; init; } = [];

    [JsonPropertyName("gpu")]
    public TrainingGpuInfo Gpu { get; init; } = new();

    [JsonPropertyName("bitsandbytes_ok")]
    public bool BitsAndBytesOk { get; init; }

    [JsonPropertyName("cpu_toy_ready")]
    public bool CpuToyReady { get; init; }

    [JsonPropertyName("ready")]
    public bool Ready { get; init; }

    [JsonPropertyName("notes")]
    public List<string> Notes { get; init; } = [];

    [JsonPropertyName("install_hint")]
    public string InstallHint { get; init; } = "pip install corpus-studio-engine[train]";
}

public sealed class TrainingGpuInfo
{
    [JsonPropertyName("available")]
    public bool Available { get; init; }

    [JsonPropertyName("device_count")]
    public int DeviceCount { get; init; }

    [JsonPropertyName("name")]
    public string Name { get; init; } = string.Empty;

    [JsonPropertyName("total_memory_gb")]
    public double TotalMemoryGb { get; init; }
}
