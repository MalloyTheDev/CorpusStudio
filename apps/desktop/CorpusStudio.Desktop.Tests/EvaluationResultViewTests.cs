using CorpusStudio.Desktop.Models;
using Xunit;

namespace CorpusStudio.Desktop.Tests;

public class EvaluationResultViewTests
{
    [Fact]
    public void StatusChip_ReflectsPassFailError()
    {
        // slice-6 fidelity: the per-example results list chips pass/fail/error.
        var pass = new EvaluationExampleResult { ExampleId = "1", Score = 85, Passed = true };
        Assert.Equal("PASS", pass.StatusLabel);
        Assert.Equal("#6bbf9a", pass.StatusColor);

        var fail = new EvaluationExampleResult { ExampleId = "2", Score = 40, Passed = false };
        Assert.Equal("FAIL", fail.StatusLabel);
        Assert.Equal("#d76d6d", fail.StatusColor);

        // A backend error takes precedence over the pass/fail flag (recorded as a scored-0 failure).
        var error = new EvaluationExampleResult { ExampleId = "3", Score = 0, Passed = false, Error = "backend timeout" };
        Assert.Equal("ERROR", error.StatusLabel);
        Assert.Equal("#d9a35f", error.StatusColor);
    }
}
