import { useCallback, useEffect, useState } from "react";
import {
  appendRows,
  createProject,
  datasetDebt,
  gateRun,
  importCommit,
  importPreview,
  listProjects,
  listSchemas,
  previewRows,
  projectDir,
  quality,
  type AppendResult,
  type DatasetSchema,
  type DebtReport,
  type GateReport,
  type ImportCommitResult,
  type PreviewReport,
  type ProjectList,
  type ProjectSummary,
  type QualityReport,
} from "../platform/dataStudio";
import { Card, Chip, type Tone } from "./ui";

const GRADE_TONE: Record<string, Tone> = { A: "ok", B: "ok", C: "warn", D: "warn", F: "bad" };
const SEV_TONE: Record<string, Tone> = { critical: "bad", high: "bad", medium: "warn", low: "neutral" };
const GATE_TONE: Record<string, Tone> = { pass: "ok", warn: "warn", block: "bad" };

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
  const [tab, setTab] = useState<"author" | "import" | "quality">("author");
  const [importPath, setImportPath] = useState("");
  const [importPrev, setImportPrev] = useState<PreviewReport | null>(null);
  const [importResult, setImportResult] = useState<ImportCommitResult | null>(null);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [gateReport, setGateReport] = useState<GateReport | null>(null);

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
    setImportPrev(null);
    setImportResult(null);
    setQualityReport(null);
    setGateReport(null);
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

  const onImportPreview = async (): Promise<void> => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setImportResult(null);
    try {
      setImportPrev(await importPreview(selected.schema_id, importPath.trim()));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onImportCommit = async (): Promise<void> => {
    if (!selected || !list) return;
    setBusy(true);
    setError(null);
    try {
      setImportResult(await importCommit(projectDir(list, selected), importPath.trim()));
      setImportPrev(null);
      await loadDebt(selected, list);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onRunQuality = async (): Promise<void> => {
    if (!selected || !list) return;
    setBusy(true);
    setError(null);
    try {
      setQualityReport(await quality(projectDir(list, selected)));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const onRunGates = async (): Promise<void> => {
    if (!selected || !list) return;
    setBusy(true);
    setError(null);
    try {
      setGateReport(await gateRun(projectDir(list, selected), selected.schema_id));
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

          <div className="cs-segment cs-tabs" role="tablist" aria-label="Dataset action">
            <button
              className={`cs-seg${tab === "author" ? " on" : ""}`}
              role="tab"
              aria-selected={tab === "author"}
              onClick={() => setTab("author")}
            >
              Author
            </button>
            <button
              className={`cs-seg${tab === "import" ? " on" : ""}`}
              role="tab"
              aria-selected={tab === "import"}
              onClick={() => setTab("import")}
            >
              Import file
            </button>
            <button
              className={`cs-seg${tab === "quality" ? " on" : ""}`}
              role="tab"
              aria-selected={tab === "quality"}
              onClick={() => setTab("quality")}
            >
              Quality
            </button>
          </div>

          {tab === "author" ? (
            <>
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
                  <Chip tone={preview.rejected_rows > 0 ? "warn" : "ok"}>
                    {preview.accepted_rows} valid
                  </Chip>
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
            </>
          ) : null}

          {tab === "import" ? (
            <>
              <label className="cs-field">
                <span>Source file path (JSONL, CSV, TSV, or Parquet)</span>
                <input
                  className="cs-input"
                  value={importPath}
                  placeholder="/path/to/data.csv"
                  onChange={(e) => setImportPath(e.target.value)}
                />
              </label>
              <div className="cs-actions">
                <button
                  className="cs-btn"
                  disabled={busy || !importPath.trim()}
                  onClick={() => void onImportPreview()}
                >
                  Preview
                </button>
                <button
                  className="cs-btn primary"
                  disabled={busy || !importPath.trim()}
                  onClick={() => void onImportCommit()}
                >
                  Commit valid rows
                </button>
              </div>
              {importPrev ? (
                <div className="cs-preview">
                  <Chip tone={importPrev.rejected_rows > 0 ? "warn" : "ok"}>
                    {importPrev.accepted_rows} valid
                  </Chip>
                  {importPrev.rejected_rows > 0 ? (
                    <Chip tone="bad">{importPrev.rejected_rows} rejected</Chip>
                  ) : null}
                  {importPrev.failed_rows.map((f) => (
                    <div key={f.row_number} className="cs-note">
                      row {f.row_number}: {f.errors.map((x) => x.message).join("; ")}
                    </div>
                  ))}
                </div>
              ) : null}
              {importResult ? (
                <p className="cs-note">
                  Committed {importResult.committed} row(s)
                  {importResult.rejected > 0 ? `, ${importResult.rejected} rejected` : ""}
                  {importResult.version_id ? ` · version ${importResult.version_id}` : ""}.
                </p>
              ) : null}
            </>
          ) : null}

          {tab === "quality" ? (
            <>
              <div className="cs-actions">
                <button className="cs-btn" disabled={busy} onClick={() => void onRunQuality()}>
                  Run quality report
                </button>
                <button className="cs-btn" disabled={busy} onClick={() => void onRunGates()}>
                  Run gates
                </button>
              </div>

              {debt && debt.items.length > 0 ? (
                <div className="cs-ledger">
                  {debt.items.map((it) => (
                    <div key={it.category} className="cs-ledger-item">
                      <Chip tone={SEV_TONE[it.severity] ?? "neutral"}>{it.severity}</Chip>
                      <span className="cs-ledger-msg">{it.message}</span>
                      <span className="cs-note">{it.remediation}</span>
                    </div>
                  ))}
                </div>
              ) : debt ? (
                <p className="cs-note">No outstanding debt items.</p>
              ) : null}

              {gateReport ? (
                <div className="cs-gates">
                  <Chip tone={GATE_TONE[gateReport.overall_status] ?? "neutral"}>
                    Gates: {gateReport.overall_status}
                  </Chip>
                  <span className="cs-note">
                    {gateReport.pass_count} pass · {gateReport.warn_count} warn · {gateReport.block_count} block
                  </span>
                  {gateReport.results
                    .filter((r) => r.status !== "pass")
                    .map((r) => (
                      <div key={r.gate_id} className="cs-note">
                        <Chip tone={GATE_TONE[r.status] ?? "neutral"}>{r.status}</Chip> {r.name}: {r.message}
                      </div>
                    ))}
                </div>
              ) : null}

              {qualityReport ? (
                <div className="cs-quality">
                  <div className="cs-note">
                    {qualityReport.duplicate_exact_count} exact-dup · {qualityReport.duplicate_normalized_count} near-dup ·{" "}
                    {qualityReport.low_information_count} low-info · {qualityReport.empty_row_count} empty
                  </div>
                  {qualityReport.pii_findings.map((p) => (
                    <div key={`${p.kind}-${p.sample}`} className="cs-note">
                      <Chip tone={SEV_TONE[p.severity] ?? "warn"}>{p.kind}</Chip> {p.match_count} match(es), row(s){" "}
                      {p.row_numbers.join(", ")} — {p.sample}
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
