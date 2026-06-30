# Corpus Studio Desktop

C# WPF desktop app for Corpus Studio.

The current v0.1 workflow supports:

- dashboard
- local dataset project creation
- built-in schema selection
- JSON example authoring
- Python-engine validation
- saving examples to the active project
- basic quality checks
- saved example inspection
- project reopening from the project list
- JSONL export
- local settings inspection

Build and launch from the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

Split management is still represented as an early UI surface while the engine command matures into a full desktop workflow.
