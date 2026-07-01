# Release Checklist

A pre-release gate for public-repo hygiene, reproducible evidence, and honest
scope. Corpus Studio is local-first and does not launch training or call hosted
services on its own, so "release" here means a clean, inspectable, buildable
snapshot — not a deployment.

Work top to bottom. Every box should be checked or explicitly waived with a note
in the release PR.

## 1. Versioning

- [ ] Bump `version` in `engine/pyproject.toml`.
- [ ] Update `engine/corpus_studio/__init__.py` if it carries a version string.
- [ ] Note user-facing changes since the last tag (new Labs surfaces, new CLI
      commands, new project-local JSON files).

## 2. Automated verification

- [ ] Engine tests pass: `cd engine && .\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp`.
- [ ] Desktop builds clean (0 warnings, 0 errors):
      `dotnet build apps/desktop/CorpusStudio.Desktop.sln`.
- [ ] `ruff check engine` is clean (or documented).
- [ ] CI is green on the release commit (`.github/workflows/engine-tests.yml`).

## 3. Local backend smoke evidence (opt-in)

Requires a running Ollama/OpenAI-compatible backend with at least one model.
These are the paths a first-time user exercises; capture output as evidence.

- [ ] `corpus_studio.cli model-list --backend ollama` lists models.
- [ ] `corpus_studio.cli backend-health --backend ollama --model <model>` reports reachable.
- [ ] Opt-in integration tests pass or self-skip:
      `CORPUS_STUDIO_OLLAMA_INTEGRATION=1 pytest -m integration`.
- [ ] Desktop example smoke test runs: `scripts/smoke_desktop_examples.ps1`.

## 4. Screenshots

- [ ] Refresh `docs/screenshots/desktop-v0.1.png` (or add a new versioned image)
      so the README preview matches the current UI.
- [ ] Capture the Evaluation tab showing drilldown filters, reviewed fixes, and
      report history if they changed.

## 5. Repo hygiene

- [ ] No secrets or API keys committed; `.env` is git-ignored and only
      `.env.example` is tracked.
- [ ] No local datasets, `exports/`, `data/projects/`, or `index.sqlite3`
      committed (all derived/user data stays out of the repo).
- [ ] `LICENSE` owner and year are correct; MIT terms intact.
- [ ] `MANIFEST.md` lists every tracked source file (add new models, engine
      modules, and tests).
- [ ] Docs match reality: `TASKS.md`, `ROADMAP.md`, `DATA_MODEL.md`, and the Lab
      docs reflect shipped features and project-local files.

## 6. Known non-features (state them plainly)

Corpus Studio deliberately does **not** do these yet. Keep this list in the
release notes so expectations are set:

- No training launcher — Training Lab exports inspectable configs only.
- No cloud/hosted-provider orchestration or credential management; local
  backends (Ollama, OpenAI-compatible) only.
- No multi-model benchmark suites.
- AI Assist is review-first: suggestions require human accept/reject and are
  never auto-applied to a dataset.
- The SQLite project index is an optional cache; JSON/JSONL remain the source of
  truth and the app works without it.
- Evaluation runs are single-project, single-run; no streaming progress.

## 7. Final steps

- [ ] Squash/organize the release commit(s) with a clear summary.
- [ ] Tag the release and push.
- [ ] Attach smoke evidence and screenshots to the release notes.
