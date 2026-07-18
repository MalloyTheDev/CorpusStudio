# Release Checklist

A pre-release gate for public-repo hygiene, reproducible evidence, and honest
scope. Corpus Studio is local-first: it can dispatch a sealed plan to its opt-in
first-party worker or launch a user's installed external trainer (with explicit
confirmation of the exact command), but the dependency-light control plane / distributable bundles no CUDA/PyTorch
and calls no hosted services on its own, so "release" here means a clean,
inspectable, buildable snapshot — not a deployment.

Work top to bottom. Every box should be checked or explicitly waived with a note
in the release PR.

## 1. Versioning

- [ ] Bump `version` in `engine/pyproject.toml`.
- [ ] Update `engine/corpus_studio/__init__.py` if it carries a version string.
- [ ] Note user-facing changes since the last tag (new Studio surfaces, new CLI
      commands, new project-local JSON files).
- [ ] **Reconcile [`CURRENT_STATE.md`](CURRENT_STATE.md)** to what actually shipped —
      it is the single source of truth other docs defer to, so a shipped milestone must
      move out of "Not built yet" and stale limitations (here and in feature docs) must
      be cleared. Do this every milestone, not just at release, so the docs never drift a
      full version behind again.

## 2. Automated verification

- [ ] Engine tests pass on Linux: `cd engine` then
      `.venv/bin/python -m pytest -q --no-header --basetemp=.pytest_tmp`.
- [ ] Ruff and mypy pass on Linux: from `engine`, run
      `.venv/bin/python -m ruff check corpus_studio tests` and
      `.venv/bin/python -m mypy corpus_studio`.
- [ ] Avalonia builds clean on Linux:
      `dotnet build apps/desktop/CorpusStudio.Avalonia/CorpusStudio.Avalonia.csproj`.
- [ ] WPF builds and desktop unit tests pass on Windows only:
      `dotnet build apps/desktop/CorpusStudio.Desktop.sln` and
      `dotnet test apps/desktop/CorpusStudio.Desktop.Tests/CorpusStudio.Desktop.Tests.csproj`.
- [ ] Web/Tauri client builds on Linux: `cd apps/web`, then `npm ci` and `npm run build`.
- [ ] CI is green on the release commit (`.github/workflows/engine-tests.yml`
      and `.github/workflows/desktop-tests.yml`).

## 3. Local backend smoke evidence (opt-in)

Requires a running Ollama/OpenAI-compatible backend with at least one model.
These are the paths a first-time user exercises; capture output as evidence.

- [ ] `corpus_studio.cli model-list --backend ollama` lists models.
- [ ] `corpus_studio.cli backend-health --backend ollama --model <model>` reports reachable.
- [ ] Opt-in integration tests pass or self-skip:
      `CORPUS_STUDIO_OLLAMA_INTEGRATION=1 pytest -m integration`.
- [ ] Windows-only WPF example smoke test runs: `scripts/smoke_desktop_examples.ps1`.

## 4. Screenshots

- [ ] **Deferred until the production UI is settled.** The desktop is currently a
      high-fidelity **prototype**, so the README no longer ships a screenshot
      gallery. Re-introduce the workflow screenshots (Start Center, wizard,
      Explorer, Studio dashboard, Problems, Output, Debt) and the Evaluation-tab
      drilldown only once the settled product UI lands.

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

Corpus Studio deliberately does **not** do these. Keep this list in the release
notes so expectations are set:

- No training deps in the dependency-light core / distributable — it bundles no
  CUDA/PyTorch/Transformers. Training is opt-in via the `[train]` extra (the
  first-party QLoRA trainer, which delegates to TRL/peft), or launch your own
  installed trainer; either way the exact argv is shown and confirmed first.
- No cloud/hosted-provider *generation* or credential management: local backends
  (Ollama, OpenAI-compatible) do the generating; OpenAI/Anthropic are
  evaluator-only and only when the user configures them. Nothing is called on its
  own, and no dataset is uploaded/published.
- AI Assist is review-first: suggestions require human accept/reject and are
  never auto-applied to a dataset. Generated candidates are gated before review,
  but the gate only informs — it never auto-accepts or auto-rejects.
- The SQLite project index is an optional cache; JSON/JSONL remain the source of
  truth and the app works without it.
- Evaluation runs are single-project, single-run; no streaming progress.

## 7. Final steps

- [ ] Squash/organize the release commit(s) with a clear summary.
- [ ] Tag the release (`v*`) and push. This triggers the **Release** workflow
      (`.github/workflows/release.yml`): it re-runs the engine + desktop gates,
      publishes the self-contained single-file Windows build (`win-x64` profile,
      WPF head), and — via the `avalonia-dist` job — self-contained macOS
      (`osx-arm64` / `osx-x64`) and Linux (`linux-x64`) builds of the Avalonia
      head, attaching them all to the GitHub Release. (Run it via
      **workflow_dispatch** first for a dry-run build — it publishes nothing,
      only uploads each platform build as a workflow artifact.)
- [ ] Attach smoke evidence and screenshots to the release notes.
