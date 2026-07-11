using System;
using System.Windows.Input;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>A minimal parameterized <see cref="ICommand"/> — the generic sibling of
/// <see cref="RelayCommand"/> — so a shared view-model can expose a single command that both heads
/// bind with a <c>CommandParameter</c> (e.g. the grouped-IA nav rows all bind one
/// <c>SelectStudioTabCommand</c> with the target tab name) instead of one command per target.
/// <see cref="ICommand"/> lives in the BCL, so this compiles in the net8.0 CorpusStudio.Core.</summary>
public sealed class RelayCommand<T> : ICommand
{
    private readonly Action<T?> _execute;
    private readonly Func<T?, bool>? _canExecute;

    public RelayCommand(Action<T?> execute, Func<T?, bool>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => _canExecute?.Invoke(Cast(parameter)) ?? true;

    public void Execute(object? parameter) => _execute(Cast(parameter));

    /// <summary>Re-query the guard (call when the underlying state changes).</summary>
    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);

    private static T? Cast(object? parameter) => parameter is T typed ? typed : default;
}
