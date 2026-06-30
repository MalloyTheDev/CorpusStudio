using System.Collections.ObjectModel;

namespace CorpusStudio.Desktop.ViewModels;

public sealed class MainWindowViewModel
{
    public ObservableCollection<string> Projects { get; } = new()
    {
        "coding_tutor_v0.1",
        "raw_game_engine_corpus",
        "preference_pairs_demo"
    };

    public string ActiveProjectTitle { get; } = "New Dataset Project";

    public string ActiveSchemaDescription { get; } =
        "Choose a schema, write examples, validate rows, and export model-ready JSONL.";

    public string DraftText { get; set; } =
        "{\n  \"instruction\": \"Explain what a variable is.\",\n  \"input\": \"\",\n  \"output\": \"A variable stores a value.\"\n}";

    public string ValidationSummary { get; } =
        "No validation has run yet. v0.1 will call the Python dataset engine.";

    public string QualitySummary { get; } =
        "Quality dashboard placeholder: duplicates, token length, missing fields, and split leakage.";
}
