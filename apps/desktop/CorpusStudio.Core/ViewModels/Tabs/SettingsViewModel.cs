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
    private GateThresholds _gateThresholds = new();
    private string _gateThresholdsSummary =
        "Gate thresholds load with the project. Edit and save to override how gates block/warn.";
    private string _providerApprovalProvider = "ollama";
    private string _providerApprovalModel = string.Empty;

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

    /// <summary>The provider chosen in the "approve generation" combo (bound two-way via SelectedValue).</summary>
    public string ProviderApprovalProvider
    {
        get => _providerApprovalProvider;
        set => SetField(ref _providerApprovalProvider, value);
    }

    /// <summary>The model name typed into the "approve generation" box (bound two-way).</summary>
    public string ProviderApprovalModel
    {
        get => _providerApprovalModel;
        set => SetField(ref _providerApprovalModel, value);
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

    /// <summary>The editable per-project gate thresholds (the Settings form two-way-binds its fields).
    /// Replaced wholesale on load so the form rebinds.</summary>
    public GateThresholds GateThresholds
    {
        get => _gateThresholds;
        private set => SetField(ref _gateThresholds, value);
    }

    public string GateThresholdsSummary
    {
        get => _gateThresholdsSummary;
        private set => SetField(ref _gateThresholdsSummary, value);
    }

    public void ApplyGateThresholds(GateThresholds thresholds)
    {
        GateThresholds = thresholds;
        GateThresholdsSummary = "Loaded effective gate thresholds (defaults merged with this project's "
            + "gate_thresholds.json). Edit and Save to override.";
    }

    public void SetGateThresholdsSaved()
    {
        GateThresholdsSummary = "Saved. The new thresholds apply the next time gates run for this project.";
    }

    public void SetGateThresholdsError(string message)
    {
        GateThresholdsSummary = $"Gate thresholds could not be saved.{Environment.NewLine}{message}";
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
