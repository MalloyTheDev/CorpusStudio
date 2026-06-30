namespace CorpusStudio.Desktop.Models;

public sealed record SavedExampleItem(int RowNumber, string Preview, string Json)
{
    public string Title => $"Example {RowNumber}";

    public string DisplayName => $"{Title}: {Preview}";
}
