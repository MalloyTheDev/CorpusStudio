using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels.Tabs;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

// Provider-policy behaviour now lives in the extracted SettingsViewModel (Phase 2 decomposition).
public sealed class ProviderPolicyViewModelTests
{
    [Fact]
    public void ApplyProviderPolicies_ShowsAllowedForApprovedLocalModel()
    {
        var vm = new SettingsViewModel();
        vm.ApplyProviderPolicies(
        [
            new ProviderPolicyItem
            {
                ProviderId = "ollama",
                DisplayName = "Ollama (local)",
                ProviderKind = "local",
                GenerationAllowed = true,
                UserApprovedGeneration = true,
            },
        ]);

        Assert.Contains("✅ Ollama (local) (local) — generation ALLOWED", vm.ProviderPolicySummary);
    }

    [Fact]
    public void ApplyProviderPolicies_ShowsBlockedForEvaluatorOnly()
    {
        var vm = new SettingsViewModel();
        vm.ApplyProviderPolicies(
        [
            new ProviderPolicyItem
            {
                ProviderId = "openai",
                DisplayName = "OpenAI",
                ProviderKind = "hosted",
                GenerationAllowed = false,
                UserApprovedGeneration = false,
            },
        ]);

        Assert.Contains("⛔ OpenAI (hosted) — generation blocked", vm.ProviderPolicySummary);
    }

    [Fact]
    public void ApplyProviderPolicies_ShowsApprovalIgnoredForBlockedButApproved()
    {
        var vm = new SettingsViewModel();
        vm.ApplyProviderPolicies(
        [
            new ProviderPolicyItem
            {
                ProviderId = "openai",
                DisplayName = "OpenAI",
                ProviderKind = "hosted",
                GenerationAllowed = false,
                UserApprovedGeneration = true, // tried to approve, still blocked
            },
        ]);

        Assert.Contains("approval ignored; evaluator-only", vm.ProviderPolicySummary);
    }

    [Fact]
    public void SetProviderPolicyError_ShowsMessage()
    {
        var vm = new SettingsViewModel();
        vm.SetProviderPolicyError("openai is evaluator-only");
        Assert.Contains("Provider policy action failed", vm.ProviderPolicySummary);
        Assert.Contains("openai is evaluator-only", vm.ProviderPolicySummary);
    }

    [Fact]
    public void SetSettings_ShowsResolvedPaths()
    {
        var vm = new SettingsViewModel();
        vm.SetSettings(new DesktopSettings(
            "C:/repo", "C:/repo/engine", "C:/repo/engine/.venv/Scripts/python.exe",
            "C:/repo/data/projects", "C:/repo/exports"));

        Assert.Contains("Repository: C:/repo", vm.SettingsSummary);
        Assert.Contains("Python: C:/repo/engine/.venv/Scripts/python.exe", vm.SettingsSummary);
        Assert.Contains("Exports: C:/repo/exports", vm.SettingsSummary);
    }
}
