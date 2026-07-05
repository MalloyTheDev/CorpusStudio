using System.Collections.Generic;
using System.ComponentModel;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>The Settings tab's own view-model (backlog #4 decomposition; the first Phase-2 tab of
/// the Avalonia migration). Owns the SELF-CONTAINED Settings panels: general app settings (paths)
/// and the provider generation policy summary. The lab-backend settings deliberately stay on the
/// shell for now — <c>ApplyLabSettings</c> populates the Evaluation and AI Assist backend fields,
/// so it moves once those tabs are extracted. Behind an interface so the shell/tests/DI depend on
/// the contract, not the concrete class.</summary>
public interface ISettingsViewModel : INotifyPropertyChanged
{
    string SettingsSummary { get; }
    string ProviderPolicySummary { get; }

    /// <summary>Show the resolved app paths (repository/engine/python/projects/exports).</summary>
    void SetSettings(DesktopSettings settings);

    /// <summary>Format the provider generation policy (who may create trainable rows).</summary>
    void ApplyProviderPolicies(IReadOnlyList<ProviderPolicyItem> policies);

    /// <summary>Report a failed provider-policy action without a confident stale summary.</summary>
    void SetProviderPolicyError(string message);
}
