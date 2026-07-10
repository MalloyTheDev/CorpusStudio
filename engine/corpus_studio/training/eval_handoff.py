"""Close the train→eval loop: a grounded, honest "evaluate the model you just
trained" plan for a finished training run.

The *verification* machinery already exists — a run carries linked before/after
evaluation reports and the regression gate compares them
(``gates/runner.run_training_run_gate``). What is missing is the OPERATOR PATH
from "the run succeeded" to a linked after-eval, because producing that after-eval
means *serving* the trained model, which is an external, trainer/format-specific
step this local-first engine deliberately does not automate. This module turns a
run's own recorded fields into the concrete, ordered steps that close the loop:
serve the model, run the Eval Lab against it, link the report to the run, and gate
the result.

Honesty boundary: the serve step is a REMINDER with a clearly-labelled example,
not a guarantee — the exact command depends on the adapter/checkpoint format and
your serving stack (Ollama, vLLM, TGI, ...), which this engine does not inspect.
The eval / link / gate commands are exact and copy-pasteable once the model is
served and named. Nothing here runs a model, contacts a backend, or serves
anything; it only composes commands from the run record. A plan is only "ready"
for a run that actually SUCCEEDED — you cannot evaluate a model a run never
produced.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from corpus_studio.training.run_registry import SUCCEEDED

# The after-eval report lands at a stable, per-run relative path so the link +
# gate steps can reference it deterministically.
_AFTER_EVAL_REL = "eval_reports/after-{run_id}.json"

# Clearly-labelled placeholders — a plan stays copy-pasteable but is honest that
# these must be filled in (the served name is chosen by the user at serve time;
# the held-out set + schema come from the baseline eval).
_SERVED_PLACEHOLDER = "<your-served-model>"
_DATASET_PLACEHOLDER = "<held-out-dataset.jsonl>"
_SCHEMA_PLACEHOLDER = "<schema-id>"


class HandoffStep(BaseModel):
    """One ordered step in the close-the-loop plan.

    ``command`` is empty for a manual/external step (serving) and a concrete,
    copy-pasteable ``corpus-studio`` invocation for the automated steps.
    """

    title: str
    detail: str
    command: str = ""


class EvalHandoffPlan(BaseModel):
    """The ordered plan to evaluate the model a training run produced.

    ``ready`` is True only for a SUCCEEDED run; otherwise ``steps`` is empty and
    ``note`` explains why there is nothing to evaluate yet.
    """

    run_id: str
    status: str
    ready: bool = False
    output_dir: str = ""
    base_model: str = ""
    served_model: str = ""
    after_eval_path: str = ""
    note: str = ""
    steps: list[HandoffStep] = []


def _quote(value: str) -> str:
    """Double-quote a path for the shell (paths routinely contain spaces)."""

    return f'"{value}"'


def build_eval_handoff(
    run: Any,
    *,
    project_dir: str,
    eval_dataset_path: str = "",
    schema_id: str = "",
    backend: str = "ollama",
    base_url: str | None = None,
    served_model: str = "",
) -> EvalHandoffPlan:
    """Build the ordered serve → eval → link → gate plan for a finished run.

    ``run`` is duck-typed (a ``TrainingRunRecord``: ``status``, ``run_id``,
    ``output_dir``, ``base_model``). ``eval_dataset_path`` / ``schema_id`` should
    be the SAME held-out set + schema used for the run's baseline (before-eval) so
    the regression gate can compare like with like; when unknown they render as
    labelled placeholders. ``served_model`` is the name the model is served under
    (chosen at serve time); empty renders as a placeholder. Pure: no I/O, never
    raises on a well-formed record.
    """

    status = getattr(run, "status", "") or ""
    run_id = getattr(run, "run_id", "") or ""
    output_dir = getattr(run, "output_dir", "") or ""
    base_model = getattr(run, "base_model", "") or ""

    after_eval_rel = _AFTER_EVAL_REL.format(run_id=run_id)
    served = served_model or _SERVED_PLACEHOLDER
    dataset = eval_dataset_path or _DATASET_PLACEHOLDER
    schema = schema_id or _SCHEMA_PLACEHOLDER

    if status != SUCCEEDED:
        return EvalHandoffPlan(
            run_id=run_id,
            status=status,
            ready=False,
            output_dir=output_dir,
            base_model=base_model,
            served_model=served,
            after_eval_path=after_eval_rel,
            note=(
                f"This run's status is '{status or 'unknown'}', not '{SUCCEEDED}'. There is no "
                "produced model to evaluate yet — close the loop once the run succeeds."
            ),
            steps=[],
        )

    # First-party runs (corpus_studio) produce a LoRA ADAPTER, so the serve step is different from the
    # generic external-trainer one: merge it (train-merge / the Merge-adapter button) then serve, or
    # serve base+adapter unmerged. Read the target off the run record — no signature change needed.
    target_normalized = (getattr(run, "target", "") or "").strip().lower().replace("-", "_")
    is_first_party = target_normalized in {"corpus_studio", "corpusstudio", "corpus", "first_party", "firstparty"}
    if is_first_party:
        serve_detail = (
            f"This run produced a LoRA adapter in {output_dir or '<output_dir>'}. To serve it, either "
            "(a) MERGE it into a standalone model first — "
            f"`corpus-studio train-merge {_quote(output_dir) if output_dir else '<adapter-dir>'}` (or the "
            "Training tab's 'Merge adapter' button, which falls back to CPU-offload / adapter-only on a "
            "small GPU) — then serve the merged model, or (b) serve the base + adapter unmerged (vLLM "
            "`--enable-lora`, or `peft.PeftModel.from_pretrained` at load). CorpusStudio does NOT serve "
            "models; pick the path your stack supports, then name the served model for the next step."
        )
    else:
        serve_detail = (
            f"Serve the model this run produced (in {output_dir or '<output_dir>'}) so the "
            "Eval Lab can reach it. CorpusStudio does NOT serve models — this step is external "
            "and depends on the checkpoint/adapter format and your stack. Example (Ollama, GGUF): "
            f"`ollama create {served} -f Modelfile` with a Modelfile FROM the produced weights; or "
            "serve with vLLM / TGI at an OpenAI-compatible endpoint and use `--backend "
            "openai-compatible --base-url <url>` in the next step."
        )

    base_url_flag = f" --base-url {_quote(base_url)}" if base_url else ""

    eval_command = (
        f"corpus-studio eval-run {_quote(dataset)} {schema} "
        f"--model {served} --backend {backend}{base_url_flag} "
        f"--output-path {_quote(after_eval_rel)}"
    )
    link_command = (
        f"corpus-studio training-run-update {_quote(project_dir)} --run-id {run_id} "
        f"--after-eval-path {_quote(after_eval_rel)} --after-eval-model {served}"
    )
    gate_command = (
        f"corpus-studio training-run-gate {_quote(project_dir)} --run-id {run_id}"
    )

    steps = [
        HandoffStep(
            title="Serve the trained model",
            detail=serve_detail,
        ),
        HandoffStep(
            title="Evaluate the trained model (writes the after-eval report)",
            detail=(
                "Run the Eval Lab against the SAME held-out set and schema you used for the baseline "
                "(before-eval), or the regression gate cannot compare them. Keep the metric the same "
                "too: add the baseline's `--judge-model ...` here if the baseline used the LLM judge, "
                "otherwise both stay on keyword-overlap. Replace the served-model name if you named it "
                "differently when serving."
            ),
            command=eval_command,
        ),
        HandoffStep(
            title="Link the after-eval to this run",
            detail=(
                "Record the after-eval report and the model it targeted on the run so the regression "
                "gate can find them. The after-eval-model must be the TRAINED model, not the base "
                f"model ({base_model or '<base_model>'}) — provenance is only trusted when they differ."
            ),
            command=link_command,
        ),
        HandoffStep(
            title="Gate the run (before vs after)",
            detail=(
                "Regression-gate the run: it compares the linked before/after reports and BLOCKS a "
                "promote if the trained model regressed. Honest scope: a PASS means the structured "
                "score did not drop past the threshold on this held-out set — not proof of quality; "
                "and it WARNs (rather than trusting the delta) if the before/after metrics differ."
            ),
            command=gate_command,
        ),
    ]

    return EvalHandoffPlan(
        run_id=run_id,
        status=status,
        ready=True,
        output_dir=output_dir,
        base_model=base_model,
        served_model=served,
        after_eval_path=after_eval_rel,
        note=(
            "Serving the trained model is external (Ollama/vLLM/TGI); the eval, link, and gate "
            "commands below are exact once it is served and named."
        ),
        steps=steps,
    )
