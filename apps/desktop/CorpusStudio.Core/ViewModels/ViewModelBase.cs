using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>Shared INotifyPropertyChanged base for the per-tab view-models extracted during the
/// god-object decomposition (backlog #4), so each doesn't re-implement SetField/OnPropertyChanged.</summary>
public abstract class ViewModelBase : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    protected void OnPropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}
