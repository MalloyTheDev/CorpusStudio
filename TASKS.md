# Initial Build Tasks

## Repo setup

- [x] Initialize Git repository.
- [x] Confirm MIT license owner/year.
- [x] Create GitHub repository at `MalloyTheDev/CorpusStudio`.
- [x] Set `origin` to `https://github.com/MalloyTheDev/CorpusStudio.git`.
- [x] Push initial `main` branch after choosing public or private visibility.
- [x] Create Visual Studio solution.
- [x] Create Python virtual environment.
- [x] Run engine tests.
- [x] Decide WPF vs WinUI 3 final desktop target.

## v0.1 engine

- [x] Load built-in schemas.
- [x] Validate JSONL against schema.
- [x] Add project creation command.
- [x] Add JSONL export command.
- [x] Add random train/validation/test split.
- [x] Add basic quality report.

## v0.1 desktop

- [x] Dashboard view.
- [x] New project dialog.
- [x] Schema picker.
- [x] Example editor.
- [x] Validation panel.
- [x] Export center.
- [x] Settings page.

## Documentation

- [x] Expand product spec.
- [x] Add screenshots when UI exists.
- [x] Add schema examples.
- [x] Add development setup guide.
- [x] Add Evaluation Lab, AI Assist Lab, and Training Lab planning docs.
- [x] Add model backend and training config architecture docs.

## v0.2 desktop

- [x] Show saved example details in the Examples tab.
- [x] Expose quality checks in the desktop app.
- [x] Add split generation to the desktop app.
- [x] Let users configure split ratios and seed from the desktop app.
- [x] Make validation stricter for schema field types.
- [x] Make validation errors easier to jump to from the editor.
- [x] Add import preview and failed-row reporting.
- [x] Add partial-import recovery/quarantine controls for rejected rows.
- [x] Add richer quality checks for duplicate and low-information rows.
- [x] Add split preview warnings for tiny or empty validation/test files.
- [x] Persist split settings per project.
- [x] Add project-level quality history.
- [x] Add import quarantine review and retry UI.

## Lab foundations

- [x] Add Evaluation Lab report/scoring skeletons without model calls.
- [x] Add model backend config skeletons without network calls.
- [x] Add Training Lab config/estimator skeletons without trainer dependencies.
- [x] Wire Evaluation Lab MVP CLI to Ollama and OpenAI-compatible local endpoints.
- [x] Add Evaluation Lab desktop UI.
- [x] Add AI Assist Lab reviewed draft workflow.
- [x] Add Training Lab config export UI.

## Lab hardening

- [x] Add model backend health check CLI and desktop buttons.
- [x] Add Ollama/local backend model discovery CLI and desktop pickers.
- [x] Persist Evaluation and AI Assist backend/model settings per project.
- [x] Add Evaluation Lab report history and reload UI.
- [x] Add manual per-example scoring and notes for Evaluation Lab.
- [x] Add AI Assist persistent review queue with accept/reject states.
- [x] Add AI Assist side-by-side original/suggested diff.
- [x] Add schema-aware AI Assist action presets.
- [x] Add repetitive synthetic pattern checks.
- [x] Add preference-pair strength review.
- [x] Add AI Assist review queue filters and bulk triage controls.
- [x] Add dataset-wide synthetic pattern checks to quality reports.
- [x] Add AI Assist review queue search, sorting, and bulk triage undo.
- [x] Add multi-step AI Assist bulk triage undo.
- [x] Add saved AI Assist queue views.
- [x] Add Evaluation pre-run backend health checks.
- [x] Add Evaluation failed-example review queue filter.
- [x] Add synthetic warning severity levels and repair suggestions.
- [x] Add synthetic issue triage-to-rewrite handoff.
- [x] Add preference-pair review UI and AI Assist judge handoff.
- [x] Add multi-pair preference ranking and contrast filters.
- [x] Add Evaluation failed-example AI Assist triage preparation.
- [x] Add batch synthetic rewrite preparation for affected rows.
- [x] Add preference ranking export and visible batch judge preparation.

## Next priority board

- [x] Add Evaluation run comparison for two saved reports.
- [x] Add Evaluation regression rerun flow using the same dataset, backend, model, and score threshold.
- [x] Add weak-example edit/re-test loop from failed evaluation rows back into Writing Studio.
- [x] Add richer Evaluation report summaries by tag, failure reason, and score band.
- [ ] Add persistent AI Assist rewrite batches so prepared synthetic rewrites can be resumed after app restart.
- [x] Add target-specific Training config compatibility warnings for schema/format mismatches.
- [x] Add dataset card export using project metadata, schema, splits, quality history, and evaluation summary.
- [ ] Add optional SQLite-backed project index while keeping JSON/JSONL files inspectable.
- [ ] Add opt-in local integration tests for Ollama discovery, health, Evaluation, and AI Assist.
- [ ] Add a release checklist for public repo hygiene, screenshots, smoke evidence, and known non-features.
