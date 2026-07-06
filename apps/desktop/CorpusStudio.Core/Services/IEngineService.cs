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
}
