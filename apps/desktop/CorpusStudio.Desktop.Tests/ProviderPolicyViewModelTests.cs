using CorpusStudio.Desktop.Models;
using CorpusStudio.Desktop.ViewModels;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public sealed class ProviderPolicyViewModelTests
{
    [Fact]
    public void ApplyProviderPolicies_ShowsAllowedForApprovedLocalModel()
    {
        var vm = new MainWindowViewModel();
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
        var vm = new MainWindowViewModel();
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
        var vm = new MainWindowViewModel();
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
        var vm = new MainWindowViewModel();
        vm.SetProviderPolicyError("openai is evaluator-only");
        Assert.Contains("Provider policy action failed", vm.ProviderPolicySummary);
        Assert.Contains("openai is evaluator-only", vm.ProviderPolicySummary);
    }
}
