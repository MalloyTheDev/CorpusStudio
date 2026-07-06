using System.Collections.Generic;
using System.Threading.Tasks;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Services;

/// <summary>The engine operations a shared view-model needs, behind an interface so run-orchestration
/// can move out of the per-head code-behind into the VMs (as bindable async commands) and be faked in
/// tests. Implemented by <see cref="PythonEngineService"/>. Grown one method at a time as each engine
/// code-behind handler is converted — this is the first (dataset debt).</summary>
public interface IEngineService
{
    /// <summary>Compute the graded dataset-debt ledger for a project's examples.jsonl.</summary>
    Task<DebtReport> GetDatasetDebtAsync(string projectPath);

    /// <summary>Run the dataset gate suite (schema/quality/leakage/PII/eval) over a project.</summary>
    Task<GateReport> RunDatasetGatesAsync(string projectPath, string schemaId, bool exportScope = false);

    /// <summary>Run the chat conversation-structure gate over a chat project.</summary>
    Task<GateReport> RunChatGatesAsync(string projectPath);

    /// <summary>Run a prompt suite across several models (comparison artifacts, not trainable rows).</summary>
    Task<ArenaReport> RunArenaAsync(string promptsText, IReadOnlyList<string> models,
        string? judgeModel = null, string? projectPath = null);

    /// <summary>Run a named evaluation suite (live backend evaluations) and return its report.</summary>
    Task<SuiteReport> RunSuiteAsync(string projectPath, string suiteName);
}
