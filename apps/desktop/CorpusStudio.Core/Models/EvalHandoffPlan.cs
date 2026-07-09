using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CorpusStudio.Desktop.Models;

/// <summary>The ordered "evaluate the model you just trained" plan for a finished
/// training run (engine <c>training-eval-plan</c>). Closes the train→eval loop: the
/// gate + linkage already exist, but producing the after-eval means serving the
/// model — an external, format/stack-specific step the engine does not automate.
/// <see cref="Ready"/> is true only for a succeeded run.</summary>
public sealed class EvalHandoffPlan
{
    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("ready")]
    public bool Ready { get; set; }

    [JsonPropertyName("output_dir")]
    public string OutputDir { get; set; } = string.Empty;

    [JsonPropertyName("base_model")]
    public string BaseModel { get; set; } = string.Empty;

    [JsonPropertyName("served_model")]
    public string ServedModel { get; set; } = string.Empty;

    [JsonPropertyName("after_eval_path")]
    public string AfterEvalPath { get; set; } = string.Empty;

    [JsonPropertyName("note")]
    public string Note { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<HandoffStep> Steps { get; set; } = new();
}

/// <summary>One ordered step in an <see cref="EvalHandoffPlan"/>. <see cref="Command"/>
/// is empty for a manual/external step (serving) and a concrete, copy-pasteable
/// <c>corpus-studio</c> invocation for the automated steps.</summary>
public sealed class HandoffStep
{
    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("detail")]
    public string Detail { get; set; } = string.Empty;

    [JsonPropertyName("command")]
    public string Command { get; set; } = string.Empty;
}
