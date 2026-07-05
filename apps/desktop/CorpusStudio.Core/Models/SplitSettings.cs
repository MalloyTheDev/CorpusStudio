using System.Globalization;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

public sealed class SplitSettings
{
    public const double DefaultTrainRatio = 0.9;
    public const double DefaultValidationRatio = 0.05;
    public const int DefaultSeed = 42;

    [JsonPropertyName("train_ratio")]
    public double TrainRatio { get; init; } = DefaultTrainRatio;

    [JsonPropertyName("validation_ratio")]
    public double ValidationRatio { get; init; } = DefaultValidationRatio;

    [JsonPropertyName("seed")]
    public int Seed { get; init; } = DefaultSeed;

    public static SplitSettings Default => new();

    public string TrainPercentText => FormatPercentValue(TrainRatio);

    public string ValidationPercentText => FormatPercentValue(ValidationRatio);

    private static string FormatPercentValue(double ratio)
    {
        return (ratio * 100).ToString("0.##", CultureInfo.InvariantCulture);
    }
}
