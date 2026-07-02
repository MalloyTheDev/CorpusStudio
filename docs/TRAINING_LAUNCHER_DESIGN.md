# Training Launcher Design (v0.5)

Scope and architecture for the Local Training Launcher. This is the biggest
shift in the app's life: from *generating inspectable training configs* to
*launching and observing real training runs*. The roadmap gated it behind the
hardening that is now done.

## Principles

- **Local-first.** No cloud orchestration by default.
- **Never hide the trainer command.** The user always sees exactly what runs.
- **No forced ML dependencies in the engine.** Corpus Studio orchestrates the
  user's *installed* trainer (axolotl / TRL / Unsloth / Hugging Face /
  LLaMA-Factory). It never imports torch/CUDA. If the trainer is not installed,
  the app shows the command to run rather than failing silently.
- **Launching is a big, machine-consuming action.** It requires explicit
  confirmation showing the exact command before every launch.

## The one architectural decision

Orchestrate the user's installed trainer CLI. The **engine** produces the exact
command (and resume variant) per target and discovers checkpoints — pure
string/path work, fully testable. The **desktop** spawns and manages the
process. This keeps the engine dependency-free and puts OS process management
where it belongs.

## The hard part: live log streaming

The desktop↔engine bridge (`RunEngineProcessAsync`) is request/response — it
runs a command to completion and reads all output with `ReadToEndAsync`. A
training run lasts minutes to hours and needs **live, line-by-line output**.
That is net-new infrastructure and where the risk concentrates:

1. **Streaming process runner** — spawn the trainer directly (not through the
   engine), read stdout/stderr incrementally, append to a bounded log buffer.
2. **Background job state machine** — a run must survive tab switches, update
   status without blocking the UI, and support cancel (process-tree kill,
   reusing the P2 cancellation work).
3. **App-close behavior** — the trainer is a child process. Closing the app
   kills it unless detached; the MVP kills on close and warns (detach/reattach
   is a large follow-up).

## Staged plan

- **v0.5.0 — guided command (done).** Engine emits the exact launch command per
  target plus the resume variant and dependencies (`launch` in the
  `training-config` output, copyable from the desktop); `training-checkpoints`
  lists checkpoints and builds a resume command for the latest. Nothing is
  executed. Dependency-free.
- **v0.5.1 — in-app launch + live log viewer + stop.** The streaming runner and
  job state machine. Engine adds an `argv` form of the command so the desktop
  spawns without shell parsing.
- **v0.5.2 — live checkpoint tracking + in-app resume (done).** Configs carry
  `output_dir`; the desktop refreshes the checkpoint list via the
  `training-checkpoints` CLI (slow poll during a run + on end/stop/error +
  manual), and "Resume latest" relaunches through the same confirmation for
  targets with a CLI resume flag. The directory is the persistence — no state
  file, latest-only resume.
- **v0.5.3 — before/after eval comparison (done).** The newest saved evaluation
  report is captured as the "before" baseline at launch. The app cannot serve
  the trained adapter itself, so the user loads it into their local backend and
  runs an evaluation; "Compare vs baseline" then reuses the existing two-report
  comparison (after − before deltas), with honest guidance for the
  no-baseline / no-after-eval cases.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| No GPU / trainer not installed | Detect best-effort; degrade to the guided command; never fail silently. |
| App closed mid-run | MVP kills the child and warns; detach/reattach deferred. |
| Cross-platform (Windows-first; trainers often need Linux/WSL) | Command preview works everywhere; real launch may need WSL — detect and guide. |
| Runaway resource use | Explicit confirmation showing the exact command before every launch. |
| Corrupt/partial checkpoints | Checkpoint tracking is read-only/advisory; resume is opt-in. |
| Spawning arbitrary commands | Spawn a structured `argv` (no shell), from a command the engine generated and the user confirmed. |

## Testability

- Engine command/checkpoint logic is pure — unit-tested with crafted inputs.
- The streaming runner is tested against trivial deterministic processes
  (echo for streaming/exit, a long-lived process for stop/kill).
- Job-state and log-buffer logic live in the view-model and are tested without
  a UI thread; only the thin Dispatcher marshaling lives in code-behind.
