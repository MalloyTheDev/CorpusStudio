using System;
using System.Threading.Tasks;
using System.Windows.Input;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Parameterized async <see cref="ICommand"/> — the generic sibling of
/// <see cref="AsyncRelayCommand"/> for commands that act on a bound item (e.g. the explorer
/// context menu's rename/delete, which receive the clicked <c>WorkspaceTreeNode</c> via
/// <c>CommandParameter</c>). Guards against re-entry the same way, and coerces the incoming
/// <see cref="object"/> parameter to <typeparamref name="T"/> (a mismatch is treated as null so a
/// stray binding can't throw on the UI thread).</summary>
public sealed class AsyncRelayCommand<T> : ICommand
{
    private readonly Func<T?, Task> _execute;
    private readonly Func<T?, bool>? _canExecute;
    private bool _isRunning;

    public AsyncRelayCommand(Func<T?, Task> execute, Func<T?, bool>? canExecute = null)
    {
        _execute = execute ?? throw new ArgumentNullException(nameof(execute));
        _canExecute = canExecute;
    }

    public event EventHandler? CanExecuteChanged;

    public bool IsRunning
    {
        get => _isRunning;
        private set
        {
            if (_isRunning == value)
            {
                return;
            }

            _isRunning = value;
            CanExecuteChanged?.Invoke(this, EventArgs.Empty);
        }
    }

    public bool CanExecute(object? parameter) =>
        !_isRunning && (_canExecute?.Invoke(Coerce(parameter)) ?? true);

    public async void Execute(object? parameter)
    {
        if (!CanExecute(parameter))
        {
            return;
        }

        IsRunning = true;
        try
        {
            await _execute(Coerce(parameter));
        }
        finally
        {
            IsRunning = false;
        }
    }

    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);

    private static T? Coerce(object? parameter) => parameter is T typed ? typed : default;
}
