using System.Windows;
using CorpusStudio.Desktop.Services;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

/// <summary>The pure WPF mapping in the Phase-0 dialog seam. MessageBox.Show can't run headless,
/// but the enum→WPF mapping + default-button + affirmative interpretation are pure and must
/// preserve the semantics of the confirm calls they replace.</summary>
public sealed class MessageBoxDialogServiceTests
{
    [Theory]
    [InlineData(DialogButtons.YesNo, MessageBoxButton.YesNo)]
    [InlineData(DialogButtons.OkCancel, MessageBoxButton.OKCancel)]
    public void ToButton_Maps(DialogButtons input, MessageBoxButton expected) =>
        Assert.Equal(expected, MessageBoxDialogService.ToButton(input));

    [Theory]
    [InlineData(DialogSeverity.Information, MessageBoxImage.Information)]
    [InlineData(DialogSeverity.Warning, MessageBoxImage.Warning)]
    [InlineData(DialogSeverity.Error, MessageBoxImage.Error)]
    [InlineData(DialogSeverity.Question, MessageBoxImage.Question)]
    public void ToImage_Maps(DialogSeverity input, MessageBoxImage expected) =>
        Assert.Equal(expected, MessageBoxDialogService.ToImage(input));

    [Theory]
    [InlineData(MessageBoxButton.YesNo, false, MessageBoxResult.No)]       // safe default
    [InlineData(MessageBoxButton.YesNo, true, MessageBoxResult.Yes)]       // affirmative default
    [InlineData(MessageBoxButton.OKCancel, false, MessageBoxResult.Cancel)]
    [InlineData(MessageBoxButton.OKCancel, true, MessageBoxResult.OK)]
    public void DefaultResult_PicksSafeOrAffirmative(
        MessageBoxButton button, bool affirmative, MessageBoxResult expected) =>
        Assert.Equal(expected, MessageBoxDialogService.DefaultResult(button, affirmative));

    [Theory]
    [InlineData(MessageBoxResult.Yes, true)]
    [InlineData(MessageBoxResult.OK, true)]
    [InlineData(MessageBoxResult.No, false)]
    [InlineData(MessageBoxResult.Cancel, false)]
    [InlineData(MessageBoxResult.None, false)]
    public void IsAffirmative_OnlyYesOrOk(MessageBoxResult result, bool expected) =>
        Assert.Equal(expected, MessageBoxDialogService.IsAffirmative(result));
}
