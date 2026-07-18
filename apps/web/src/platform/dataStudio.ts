// Data Studio API — typed wrappers over the Tauri commands that shell to the engine CLI.
// The engine is the single writer of examples.jsonl (via `examples-append`); the shell only
// authors rows and dispatches the sanctioned command. Only callable inside Tauri.

export interface SchemaField {
  name: string;
  type: string;
  required: boolean;
  description?: string | null;
}

export interface DatasetSchema {
  id: string;
  name: string;
  version: string;
  description: string;
  fields: SchemaField[];
  example?: unknown;
}

export interface ProjectSummary {
  id: string;
  name: string;
  schema_id: string;
  created_at: string;
  updated_at: string;
}

export interface ProjectList {
  projects_root: string;
  count: number;
  projects: ProjectSummary[];
}

export interface PreviewIssue {
  level: string;
  message: string;
  row_number: number | null;
}

export interface FailedRow {
  row_number: number;
  raw_preview: string;
  errors: PreviewIssue[];
}

export interface PreviewReport {
  valid: boolean;
  schema_id: string;
  total_rows: number;
  accepted_rows: number;
  rejected_rows: number;
  failed_rows: FailedRow[];
}

export interface AppendResult {
  examples_path: string;
  appended: number;
  skipped_invalid: number;
  schema_id: string;
}

export interface DebtReport {
  example_count: number;
  has_data: boolean;
  grade: string;
  items: unknown[];
}

async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

export const listSchemas = (): Promise<DatasetSchema[]> => call<DatasetSchema[]>("data_schemas");

export const listProjects = (): Promise<ProjectList> => call<ProjectList>("data_projects");

export const createProject = (
  projectId: string,
  name: string,
  schema: string,
): Promise<{ project_dir: string }> =>
  call<{ project_dir: string }>("data_new_project", { projectId, name, schema });

export const previewRows = (schema: string, rowsJsonl: string): Promise<PreviewReport> =>
  call<PreviewReport>("data_preview", { schema, rowsJsonl });

export const appendRows = (projectDir: string, rowsJsonl: string): Promise<AppendResult> =>
  call<AppendResult>("data_append", { projectDir, rowsJsonl });

export const datasetDebt = (projectDir: string): Promise<DebtReport> =>
  call<DebtReport>("data_debt", { projectDir });

export interface ImportCommitResult {
  examples_path: string;
  committed: number;
  rejected: number;
  version_id: string | null;
  schema_id: string;
}

/** Validate/preview a source file (JSONL/CSV/TSV/Parquet) against the schema, without committing. */
export const importPreview = (schema: string, sourcePath: string): Promise<PreviewReport> =>
  call<PreviewReport>("data_import_preview", { schema, sourcePath });

/** Commit a source file's schema-valid rows into examples.jsonl + capture a version. */
export const importCommit = (projectDir: string, sourcePath: string): Promise<ImportCommitResult> =>
  call<ImportCommitResult>("data_import_commit", { projectDir, sourcePath });

/** The project's on-disk directory (the engine's default projects root + id). */
export const projectDir = (list: ProjectList, project: ProjectSummary): string =>
  `${list.projects_root}/${project.id}`;
