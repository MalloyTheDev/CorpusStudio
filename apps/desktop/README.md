# Corpus Studio Desktop

C# WPF desktop app for Corpus Studio.

The current v0.1 workflow supports:

- dashboard
- local dataset project creation
- built-in schema selection
- JSON example authoring
- Python-engine validation
- saving examples to the active project
- JSONL export
- local settings inspection

Build and launch from the repository root:

```powershell
dotnet build apps\desktop\CorpusStudio.Desktop.sln
.\apps\desktop\CorpusStudio.Desktop\bin\Debug\net8.0-windows\CorpusStudio.Desktop.exe
```

Quality and split management are still represented as early UI surfaces while the engine commands mature into full desktop workflows.
