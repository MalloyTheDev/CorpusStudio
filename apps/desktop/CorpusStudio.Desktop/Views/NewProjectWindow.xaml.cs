using System.Collections.ObjectModel;
using System.Text.RegularExpressions;
using System.Windows;

using CorpusStudio.Desktop.Models;

namespace CorpusStudio.Desktop.Views;

public partial class NewProjectWindow : Window
{
    private static readonly Regex ProjectIdPattern = new("^[a-z0-9][a-z0-9_-]*$");
    private bool _projectIdTouched;

    public NewProjectWindow(IReadOnlyList<DatasetSchema> schemas)
    {
        InitializeComponent();
        Schemas = new ObservableCollection<DatasetSchema>(schemas);
        SchemaComboBox.ItemsSource = Schemas;
        SchemaComboBox.SelectedItem = Schemas.FirstOrDefault(schema => schema.Id == "instruction")
            ?? Schemas.FirstOrDefault();
        UpdateSchemaPreview();
    }

    public ObservableCollection<DatasetSchema> Schemas { get; }

    private void SchemaComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        UpdateSchemaPreview();
    }

    private void UpdateSchemaPreview()
    {
        if (SchemaComboBox.SelectedItem is not DatasetSchema schema)
        {
            SchemaDescriptionTextBlock.Text = string.Empty;
            SchemaExampleTextBox.Text = string.Empty;
            return;
        }

        SchemaDescriptionTextBlock.Text = schema.Description ?? string.Empty;
        SchemaExampleTextBox.Text = schema.ExampleText;
    }

    public NewProjectRequest? ProjectRequest { get; private set; }

    private void ProjectNameTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (_projectIdTouched)
        {
            return;
        }

        ProjectIdTextBox.Text = Slugify(ProjectNameTextBox.Text);
        ProjectIdTextBox.CaretIndex = ProjectIdTextBox.Text.Length;
    }

    private void ProjectIdTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
    {
        if (ProjectIdTextBox.IsKeyboardFocusWithin)
        {
            _projectIdTouched = true;
        }
    }

    private void CreateButton_Click(object sender, RoutedEventArgs e)
    {
        ErrorTextBlock.Text = string.Empty;

        var name = ProjectNameTextBox.Text.Trim();
        var projectId = ProjectIdTextBox.Text.Trim();
        var schema = SchemaComboBox.SelectedItem as DatasetSchema;

        if (string.IsNullOrWhiteSpace(name))
        {
            ErrorTextBlock.Text = "Enter a project name.";
            ProjectNameTextBox.Focus();
            return;
        }

        if (string.IsNullOrWhiteSpace(projectId) || !ProjectIdPattern.IsMatch(projectId))
        {
            ErrorTextBlock.Text = "Use lowercase letters, numbers, underscores, or hyphens for the project ID.";
            ProjectIdTextBox.Focus();
            return;
        }

        if (schema is null)
        {
            ErrorTextBlock.Text = "Choose a dataset schema.";
            SchemaComboBox.Focus();
            return;
        }

        ProjectRequest = new NewProjectRequest(projectId, name, schema.Id, schema.Name);
        DialogResult = true;
    }

    private static string Slugify(string value)
    {
        var lower = value.Trim().ToLowerInvariant();
        var characters = lower.Select(character =>
            char.IsLetterOrDigit(character) ? character : '_'
        ).ToArray();
        var slug = Regex.Replace(new string(characters), "_+", "_").Trim('_');
        return slug.Length == 0 ? "new_dataset_project" : slug;
    }
}
