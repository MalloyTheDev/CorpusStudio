using System;
using System.Threading.Tasks;
using System.Windows.Input;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>An async <see cref="ICommand"/> over a <see cref="Func{Task}"/>. Guards against re-entry
/// (CanExecute is false while a run is in flight, so a second click can't start an overlapping run) and
/// swallows nothing — the wrapped task owns its own try/catch. Lets the shared view-models expose
/// engine-run orchestration as bindable commands instead of per-head code-behind async handlers.</summary>
public sealed class AsyncRelayCommand : ICommand
{
    private readonly Func<Task> _execute;
    private readonly Func<bool>? _canExecute;
    private bool _isRunning;

    public AsyncRelayCommand(Func<Task> execute, Func<bool>? canExecute = null)
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

    public bool CanExecute(object? parameter) => !_isRunning && (_canExecute?.Invoke() ?? true);

    public async void Execute(object? parameter)
    {
        if (!CanExecute(parameter))
        {
            return;
        }

        IsRunning = true;
        try
        {
            await _execute();
        }
        finally
        {
            IsRunning = false;
        }
    }

    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
