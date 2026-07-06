using System;
using System.Windows.Input;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>A minimal <see cref="ICommand"/> over an <see cref="Action"/> (+ optional guard). Lets the
/// shared view-models expose commands so both heads bind <c>Command="{Binding X}"</c> instead of
/// per-head code-behind <c>Click</c> handlers — the cross-platform-friendly pattern the Avalonia port
/// moves toward. <see cref="ICommand"/> lives in the BCL (not WPF), so this compiles in the net8.0
/// CorpusStudio.Core.</summary>
public sealed class RelayCommand : ICommand
{
    private readonly Action _execute;
    private readonly Func<bool>? _canExecute;

    public RelayCommand(Action execute, Func<bool>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => _canExecute?.Invoke() ?? true;

    public void Execute(object? parameter) => _execute();

    /// <summary>Re-query the guard (call when the underlying state changes).</summary>
    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
