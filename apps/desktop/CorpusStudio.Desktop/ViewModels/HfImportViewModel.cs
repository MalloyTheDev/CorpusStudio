using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.ViewModels;

/// <summary>One schema field's column-mapping row in the Hugging Face import dialog:
/// which HF column feeds this schema field (empty = leave the field unmapped).</summary>
public sealed class HfFieldMapping : ViewModelBase
{
    private string _selectedColumn;

    public HfFieldMapping(string fieldName, bool required, IReadOnlyList<string> availableColumns, string selectedColumn)
    {
        FieldName = fieldName;
        Required = required;
        AvailableColumns = availableColumns;
        _selectedColumn = selectedColumn;
    }

    public string FieldName { get; }
    public bool Required { get; }

    /// <summary>The dataset's columns, with a leading empty entry meaning "(none)".</summary>
    public IReadOnlyList<string> AvailableColumns { get; }

    public string SelectedColumn
    {
        get => _selectedColumn;
        set => SetField(ref _selectedColumn, value);
    }
}

/// <summary>Backs the "Import from Hugging Face" dialog (#7 slice 2). Inspect a public
/// dataset, surface its license, and map its columns to the ACTIVE project's schema; the
/// import itself writes a staging file that the caller runs through the normal
/// import-preview / quarantine flow (the desktop stays the single writer of examples.jsonl).
/// Read-only and public-only — gated datasets are blocked.</summary>
public sealed class HfImportViewModel : ViewModelBase
{
    private readonly IReadOnlyList<DatasetField> _schemaFields;

    private string _datasetId = string.Empty;
    private bool _isBusy;
    private bool _hasInspection;
    private bool _gated;
    private string _licenseNote = string.Empty;
    private HfConfigSplit? _selectedConfigSplit;
    private int _rowLimit = 200;
    private string _statusMessage =
        "Enter a public Hugging Face dataset id (e.g. tatsu-lab/alpaca), then Inspect.";

    public HfImportViewModel(string schemaId, string schemaName, IReadOnlyList<DatasetField> schemaFields)
    {
        SchemaId = schemaId;
        SchemaName = schemaName;
        _schemaFields = schemaFields;
    }

    public string SchemaId { get; }
    public string SchemaName { get; }

    public ObservableCollection<HfConfigSplit> ConfigsSplits { get; } = [];
    public ObservableCollection<HfFieldMapping> FieldMappings { get; } = [];

    public string DatasetId
    {
        get => _datasetId;
        set
        {
            if (SetField(ref _datasetId, value))
            {
                OnPropertyChanged(nameof(CanInspect));
            }
        }
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetField(ref _isBusy, value))
            {
                OnPropertyChanged(nameof(CanInspect));
                OnPropertyChanged(nameof(CanImport));
            }
        }
    }

    public bool HasInspection
    {
        get => _hasInspection;
        private set
        {
            if (SetField(ref _hasInspection, value))
            {
                OnPropertyChanged(nameof(CanImport));
            }
        }
    }

    public bool Gated
    {
        get => _gated;
        private set
        {
            if (SetField(ref _gated, value))
            {
                OnPropertyChanged(nameof(CanImport));
            }
        }
    }

    public string LicenseNote
    {
        get => _licenseNote;
        private set => SetField(ref _licenseNote, value);
    }

    public HfConfigSplit? SelectedConfigSplit
    {
        get => _selectedConfigSplit;
        set
        {
            if (SetField(ref _selectedConfigSplit, value))
            {
                OnPropertyChanged(nameof(CanImport));
            }
        }
    }

    public int RowLimit
    {
        get => _rowLimit;
        set => SetField(ref _rowLimit, value < 1 ? 1 : value);
    }

    public string StatusMessage
    {
        get => _statusMessage;
        private set => SetField(ref _statusMessage, value);
    }

    public bool CanInspect => !IsBusy && !string.IsNullOrWhiteSpace(DatasetId);

    public bool CanImport =>
        !IsBusy
        && HasInspection
        && !Gated
        && SelectedConfigSplit is not null
        && FieldMappings.Any(mapping => !string.IsNullOrEmpty(mapping.SelectedColumn));

    public void SetBusy(string message)
    {
        IsBusy = true;
        StatusMessage = message;
    }

    public void SetError(string message)
    {
        IsBusy = false;
        StatusMessage = message;
    }

    /// <summary>Populate the splits, license note, and column-mapping grid from an
    /// inspection. Each schema field is pre-mapped to a same-named column when one exists.</summary>
    public void ApplyInspection(HfDatasetInspection inspection)
    {
        IsBusy = false;
        Gated = inspection.Gated;
        LicenseNote = inspection.LicenseNote;

        ConfigsSplits.Clear();
        foreach (var configSplit in inspection.ConfigsSplits)
        {
            ConfigsSplits.Add(configSplit);
        }
        SelectedConfigSplit = ConfigsSplits.FirstOrDefault();

        // Column dropdown options: a leading "(none)" plus the dataset's columns.
        var options = new List<string> { string.Empty };
        options.AddRange(inspection.SampleColumns);

        var byLower = new Dictionary<string, string>();
        foreach (var column in inspection.SampleColumns)
        {
            byLower[column.ToLowerInvariant()] = column;
        }

        FieldMappings.Clear();
        foreach (var field in _schemaFields)
        {
            byLower.TryGetValue(field.Name.ToLowerInvariant(), out var suggested);
            var row = new HfFieldMapping(field.Name, field.Required, options, suggested ?? string.Empty);
            // Editing a mapping can enable/disable the Import button.
            row.PropertyChanged += (_, _) => OnPropertyChanged(nameof(CanImport));
            FieldMappings.Add(row);
        }

        HasInspection = true;
        StatusMessage = inspection.Gated
            ? "This dataset is gated (requires access approval); public import only."
            : $"Map columns to the {SchemaName} schema, then Import.";
        OnPropertyChanged(nameof(CanImport));
    }

    /// <summary>The explicit field→column mapping the user chose (unmapped fields omitted).</summary>
    public IReadOnlyDictionary<string, string> BuildMapping() =>
        FieldMappings
            .Where(mapping => !string.IsNullOrEmpty(mapping.SelectedColumn))
            .ToDictionary(mapping => mapping.FieldName, mapping => mapping.SelectedColumn);
}
