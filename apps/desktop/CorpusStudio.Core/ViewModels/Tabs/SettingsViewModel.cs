using System;
using System.Collections.Generic;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels.Tabs;

/// <summary>Concrete Settings tab view-model. Behaviour moved verbatim from the shell
/// (<c>MainWindowViewModel</c>): general app-path settings + the provider generation policy
/// summary. Provider-policy honesty is unchanged — ✅ only when generation is genuinely allowed;
/// an approved-but-blocked frontier provider is shown as approval-ignored.</summary>
public sealed class SettingsViewModel : ViewModelBase, ISettingsViewModel
{
    private string _settingsSummary = "Settings load when the app starts.";
    private string _providerPolicySummary =
        "Provider generation policy: refresh to see which providers may create trainable rows.";

    public string SettingsSummary
    {
        get => _settingsSummary;
        private set => SetField(ref _settingsSummary, value);
    }

    public string ProviderPolicySummary
    {
        get => _providerPolicySummary;
        private set => SetField(ref _providerPolicySummary, value);
    }

    public void SetSettings(DesktopSettings settings)
    {
        SettingsSummary = string.Join(
            Environment.NewLine,
            [
                $"Repository: {settings.RepositoryRoot}",
                $"Engine: {settings.EngineDirectory}",
                $"Python: {settings.PythonExecutable}",
                $"Projects: {settings.ProjectDirectory}",
                $"Exports: {settings.ExportDirectory}",
            ]
        );
    }

    public void SetProviderPolicyError(string message)
    {
        ProviderPolicySummary = $"Provider policy action failed.{Environment.NewLine}{message}";
    }

    public void ApplyProviderPolicies(IReadOnlyList<ProviderPolicyItem> policies)
    {
        var lines = new List<string>
        {
            "Provider generation policy (who may create trainable rows):",
            string.Empty,
        };

        foreach (var policy in policies)
        {
            var name = string.IsNullOrWhiteSpace(policy.DisplayName) ? policy.ProviderId : policy.DisplayName;
            if (policy.GenerationAllowed)
            {
                lines.Add($"✅ {name} ({policy.ProviderKind}) — generation ALLOWED (approved)");
            }
            else if (policy.UserApprovedGeneration)
            {
                // Approved but still blocked (frontier provider): make that explicit.
                lines.Add($"⛔ {name} ({policy.ProviderKind}) — approval ignored; evaluator-only");
            }
            else
            {
                lines.Add($"⛔ {name} ({policy.ProviderKind}) — generation blocked (evaluator-only or unapproved)");
            }
        }

        lines.Add(string.Empty);
        lines.Add("Approve a local model below to let AI Assist use it for generation (still review-required).");
        ProviderPolicySummary = string.Join(Environment.NewLine, lines);
    }
}
