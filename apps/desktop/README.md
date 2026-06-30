# Corpus Studio Desktop

C# WPF desktop app for Corpus Studio.

The current v0.1 workflow supports:

- dashboard
- local dataset project creation
- built-in schema selection
- JSON example authoring
- Python-engine validation
- saving examples to the active project
- JSONL import preview with failed-row reporting
- basic quality checks
- saved example inspection
- project reopening from the project list
- train/validation/test split generation
- JSONL export
- local settings inspection

Build and launch from the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

JSONL imports are previewed against the active schema and only fully valid files
are appended to the active project's `examples.jsonl`.

Split generation uses the engine default ratios and writes files under the configured export directory.
