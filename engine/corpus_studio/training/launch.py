"""Launch-command generation for the Training Lab (guided, not executed).

Corpus Studio does not run trainers itself. Given a rendered config and a
target, this emits the exact command the user would run with their own
installed trainer (axolotl / TRL / Unsloth / Hugging Face / LLaMA-Factory),
the resume variant, and the dependencies to install. It also lists checkpoints
found in an output directory so a resume command can point at the latest one.
Everything here is pure string/path work — no ML frameworks are imported.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class LaunchPlan(BaseModel):
    target: str
    command: str
    resume_command: str
    resume_supported: bool
    # Structured command so a launcher can spawn without shell parsing.
    argv: list[str] = Field(default_factory=list)
    resume_argv: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


_REVIEW_NOTE = (
    "Review this command before running; Corpus Studio does not launch training."
)

# Per-target launch metadata. ``resume_flag`` is a format string with a
# ``{checkpoint}`` placeholder when the trainer CLI supports a resume flag, else
# ``None`` (resume is config-driven for that target).
_TARGET_LAUNCH: dict[str, dict] = {
    "axolotl_yaml": {
        "template": 'accelerate launch -m axolotl.cli.train "{config}"',
        "argv_prefix": ["accelerate", "launch", "-m", "axolotl.cli.train"],
        "resume_flag": '--resume_from_checkpoint="{checkpoint}"',
        "resume_argv_flag": ["--resume_from_checkpoint"],
        "dependencies": ["axolotl", "accelerate"],
    },
    "llama_factory": {
        "template": 'llamafactory-cli train "{config}"',
        "argv_prefix": ["llamafactory-cli", "train"],
        "resume_flag": None,
        "resume_argv_flag": None,
        "dependencies": ["llama-factory"],
    },
    "trl_config": {
        "template": 'python "{config}"',
        "argv_prefix": ["python"],
        "resume_flag": None,
        "resume_argv_flag": None,
        "dependencies": ["trl", "transformers", "torch"],
    },
    "unsloth_script": {
        "template": 'python "{config}"',
        "argv_prefix": ["python"],
        "resume_flag": None,
        "resume_argv_flag": None,
        "dependencies": ["unsloth", "trl", "torch"],
    },
    "huggingface_trainer": {
        "template": 'python "{config}"',
        "argv_prefix": ["python"],
        "resume_flag": None,
        "resume_argv_flag": None,
        "dependencies": ["transformers", "torch"],
    },
}

RESUME_CHECKPOINT_PLACEHOLDER = "<checkpoint-dir>"


def build_launch_plan(
    target: str,
    config_path: str,
    resume_checkpoint: str | None = None,
) -> LaunchPlan:
    """Build the launch/resume commands for a target and rendered config."""

    meta = _TARGET_LAUNCH.get(target)
    if meta is None:
        supported = ", ".join(sorted(_TARGET_LAUNCH))
        raise ValueError(f"Unknown training target '{target}'. Use one of: {supported}.")

    command = meta["template"].format(config=config_path)
    argv = [*meta["argv_prefix"], config_path]
    notes = [_REVIEW_NOTE, f"Requires: {', '.join(meta['dependencies'])}."]

    resume_flag = meta["resume_flag"]
    if resume_flag is not None:
        checkpoint = resume_checkpoint or RESUME_CHECKPOINT_PLACEHOLDER
        resume_command = f"{command} {resume_flag.format(checkpoint=checkpoint)}"
        resume_argv = [*argv, *meta["resume_argv_flag"], checkpoint]
        resume_supported = True
    else:
        resume_command = command
        resume_argv = list(argv)
        resume_supported = False
        notes.append(
            "Resume is config-driven for this target; set the checkpoint path in the config."
        )

    return LaunchPlan(
        target=target,
        command=command,
        resume_command=resume_command,
        resume_supported=resume_supported,
        argv=argv,
        resume_argv=resume_argv,
        dependencies=list(meta["dependencies"]),
        notes=notes,
    )


def _checkpoint_step(name: str) -> int:
    match = re.search(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else -1


def find_checkpoints(output_dir: Path) -> list[str]:
    """Return ``checkpoint-N`` directory names in ``output_dir`` sorted by step."""

    if not output_dir.exists() or not output_dir.is_dir():
        return []

    checkpoints = [
        entry.name
        for entry in output_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("checkpoint-")
    ]
    return sorted(checkpoints, key=_checkpoint_step)


def latest_checkpoint(output_dir: Path) -> str | None:
    checkpoints = find_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None
