import { useCallback, useEffect, useState } from "react";
import {
  appendRows,
  createProject,
  datasetDebt,
  listProjects,
  listSchemas,
  previewRows,
  projectDir,
  type AppendResult,
  type DatasetSchema,
  type DebtReport,
  type PreviewReport,
  type ProjectList,
  type ProjectSummary,
} from "../platform/dataStudio";
import { Card, Chip, type Tone } from "./ui";

const GRADE_TONE: Record<string, Tone> = { A: "ok", B: "ok", C: "warn", D: "warn", F: "bad" };

/** Data Studio: create a project, author rows, validate them, and commit through the engine's
 *  sanctioned single writer (examples-append). Real engine data — no sample. */
export function DataStudio({ live }: { live: boolean }) {
  const [schemas, setSchemas] = useState<DatasetSchema[]>([]);
  const [list, setList] = useState<ProjectList | null>(null);
  const [selected, setSelected] = useState<ProjectSummary | null>(null);
  const [debt, setDebt] = useState<DebtReport | null>(null);
  const [rows, setRows] = useState("");
  const [preview, setPreview] = useState<PreviewReport | null>(null);
  const [appended, setAppended] = useState<AppendResult | null>(null);
  const [newId, setNewId] = useState("");
  const [newName, setNewName] = useState("");
  const [newSchema, setNewSchema] = useState("instruction");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!live) return;
    listSchemas().then(setSchemas).catch((e) => setError(String(e)));
    listProjects().then(setList).catch((e) => setError(String(e)));
  }, [live]);

  const loadDebt = useCallback(async (project: ProjectSummary, current: ProjectList) => {
    setDebt(await datasetDebt(projectDir(current, project)));
  }, []);

  const select = async (project: ProjectSummary): Promise<void> => {
    setSelected(project);
    setPreview(null);
    setAppended(null);
    setError(null);
    if (list) {
      try {
        await loadDebt(project, list);
      } catch (e) {
        setError(String(e));
      }
    }
  };

  const onCreate = async (): Promise<void> => {
    setBusy(true);
    setError(null);
    const id = newId.trim();
    try {
      await createProject(id, newName.trim() || id, newSchema);
      const fresh = await listProjects();
      setList(fresh);
      setNewId("");
      setNewName("");
      const created = fresh.projects.find((p) => p.id === id) ?? null;
      if (created) {
        setSelected(created);
        await loadDebt(created, fresh);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPreview = async (): Promise<void> => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setAppended(null);
    try {
      setPreview(await previewRows(selected.schema_id, rows));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onCommit = async (): Promise<void> => {
    if (!selected || !list) return;
    setBusy(true);
    setError(null);
    try {
      setAppended(await appendRows(projectDir(list, selected), rows));
      setPreview(null);
      setRows("");
      await loadDebt(selected, list);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!live) {
    return (
      <Card title="Data Studio">
        <p className="cs-note">
          Data Studio talks to the local engine, so it runs inside the Corpus Studio app, not the
          browser preview.
        </p>
      </Card>
    );
  }

  return (
    <div className="cs-datastudio">
      {error ? (
        <div className="cs-error" role="alert">
          {error}
        </div>
      ) : null}

      <Card title="Projects">
        <div className="cs-project-list">
          {list && list.projects.length > 0 ? (
            list.projects.map((p) => (
              <button
                key={p.id}
                className={`cs-project${selected?.id === p.id ? " on" : ""}`}
                onClick={() => void select(p)}
              >
                <span className="cs-project-name">{p.name}</span>
                <Chip tone="neutral">{p.schema_id}</Chip>
              </button>
            ))
          ) : (
            <p className="cs-note">No projects yet — create one below.</p>
          )}
        </div>
        <div className="cs-create">
          <input
            className="cs-input"
            placeholder="project-id"
            value={newId}
            onChange={(e) => setNewId(e.target.value)}
          />
          <input
            className="cs-input"
            placeholder="Name (optional)"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <select className="cs-input" value={newSchema} onChange={(e) => setNewSchema(e.target.value)}>
            {schemas.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
          <button className="cs-btn primary" disabled={busy || !newId.trim()} onClick={() => void onCreate()}>
            Create
          </button>
        </div>
      </Card>

      {selected ? (
        <Card title={`${selected.name} · ${selected.schema_id}`}>
          {debt ? (
            <div className="cs-debt">
              <Chip tone={GRADE_TONE[debt.grade] ?? "neutral"}>Grade {debt.grade}</Chip>
              <span className="cs-note">{debt.example_count} row(s)</span>
            </div>
          ) : null}

          <label className="cs-field">
            <span>Author rows (one JSON object per line)</span>
            <textarea
              className="cs-textarea"
              rows={6}
              value={rows}
              placeholder={'{"instruction": "...", "output": "..."}'}
              onChange={(e) => setRows(e.target.value)}
            />
          </label>
          <div className="cs-actions">
            <button className="cs-btn" disabled={busy || !rows.trim()} onClick={() => void onPreview()}>
              Preview
            </button>
            <button
              className="cs-btn primary"
              disabled={busy || !rows.trim()}
              onClick={() => void onCommit()}
            >
              Commit valid rows
            </button>
          </div>

          {preview ? (
            <div className="cs-preview">
              <Chip tone={preview.rejected_rows > 0 ? "warn" : "ok"}>{preview.accepted_rows} valid</Chip>
              {preview.rejected_rows > 0 ? <Chip tone="bad">{preview.rejected_rows} rejected</Chip> : null}
              {preview.failed_rows.map((f) => (
                <div key={f.row_number} className="cs-note">
                  row {f.row_number}: {f.errors.map((x) => x.message).join("; ")}
                </div>
              ))}
            </div>
          ) : null}
          {appended ? (
            <p className="cs-note">
              Committed {appended.appended} row(s)
              {appended.skipped_invalid > 0 ? `, skipped ${appended.skipped_invalid} invalid` : ""}.
            </p>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
