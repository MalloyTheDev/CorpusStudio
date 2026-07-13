"""The Environment Manager substrate — recipes + the install-PREVIEW resolver (Phase 2, slice 1).

"Dependency-light" describes the CONTROL PLANE, not the whole product. This module models the
three-layer dependency architecture (see ``docs/IMPLEMENTATION_PLAN.md``):

* the **control plane** — the always-installable core (this engine), no CUDA / ML framework;
* **capability profiles** — opt-in feature stacks added to the core process with graceful fallback;
* **backend worker environments** — ISOLATED per-framework runtimes (heavy frameworks pin conflicting
  torch/CUDA/xformers builds and cannot coexist in one env).

This slice ships the DECLARATIVE half: a registry of built-in :class:`EnvironmentRecipe` instances
(grounded in the engine's real optional extras) and a PURE :func:`resolve_dependencies` that renders
the exact, argv-structured (never shell) install plan + a CUDA-aware wheel-index choice + rough
disk/network estimates, for explicit confirmation BEFORE anything is installed. Actually creating the
venv + running the installers (the side-effectful half) is a later slice — this one is torch-free and
mutates nothing.
"""

from __future__ import annotations

from .contracts import (
    DependencyConflict,
    DependencyRequirement,
    DependencyResolution,
    EnvironmentRecipe,
)
from .common import Ref
from .enums import DependencyLayer, OperatingSystem, RecipeVerification

# The PyTorch wheel indices, by accelerator tag (grounded in the download.pytorch.org layout). torch
# is installed from its OWN index; the rest resolve from PyPI — so a CUDA build is selected without
# breaking every other package (the recommended pytorch.org install pattern).
PYTORCH_INDEX_URLS: dict[str, str] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "rocm6.2": "https://download.pytorch.org/whl/rocm6.2",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

# The distributions PyTorch ships from its own index (so they must NOT be requested from PyPI when a
# CUDA build is wanted).
_TORCH_DISTRIBUTIONS: frozenset[str] = frozenset({"torch", "torchvision", "torchaudio"})

# Rough, EXPLICITLY-HEURISTIC download sizes in MB. torch dominates and depends on the accelerator, so
# it is special-cased below; everything else uses this table (unknown → a conservative default).
_DOWNLOAD_MB: dict[str, int] = {
    "transformers": 50,
    "peft": 5,
    "trl": 12,
    "accelerate": 15,
    "datasets": 45,
    "bitsandbytes": 120,
    "unsloth": 70,
    "tiktoken": 6,
    "tokenizers": 10,
    "pyarrow": 45,
    "pydantic": 8,
    "typer": 4,
    "deepspeed": 60,
    "xformers": 120,
}
_DEFAULT_PACKAGE_MB = 20
_TORCH_CUDA_MB = 2600  # a CUDA torch wheel is ~2.5-2.8 GB
_TORCH_CPU_MB = 220
_MB = 1_000_000
# Installed on-disk footprint is larger than the compressed download (wheels + unpacked).
_DISK_FOOTPRINT_MULTIPLIER = 2.3


def builtin_recipes() -> list[EnvironmentRecipe]:
    """The built-in environment recipes. Only recipes grounded in the engine's REAL optional extras
    (``[train]`` / ``[parquet]`` / ``[tokenizer]`` / ``[model-tokenizer]``) are declared here; heavier
    backends (DeepSpeed/FSDP/Axolotl/LLaMA-Factory/MoE) arrive with their own backend slices so a
    recipe is never claimed before it can be built + probed."""
    all_os = [OperatingSystem.windows, OperatingSystem.wsl, OperatingSystem.linux, OperatingSystem.macos]
    return [
        EnvironmentRecipe(
            recipe_id="control-plane",
            display_name="Control plane (CorpusStudio core)",
            layer=DependencyLayer.control_plane,
            description="The always-installable dependency-light core — no CUDA, no ML framework. "
            "Opening CorpusStudio requires only this.",
            target="corpus_studio_engine",
            python_requires=">=3.10",
            dependency_requirements=[
                DependencyRequirement(name="pydantic", specifier=">=2", reason="contracts"),
                DependencyRequirement(name="typer", specifier=">=0.12", reason="CLI"),
            ],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
            notes=["Installed as `pip install -e .` (the engine); usually already present."],
        ),
        EnvironmentRecipe(
            recipe_id="capability-tokenization",
            display_name="Exact tokenization (tiktoken)",
            layer=DependencyLayer.capability,
            description="Adds exact GPT-family BPE token counts to the core (else a Unicode heuristic).",
            target="tokenization",
            python_requires=">=3.10",
            dependency_requirements=[DependencyRequirement(name="tiktoken", specifier=">=0.7")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="capability-model-tokenizer",
            display_name="Model tokenizers (tokenizers)",
            layer=DependencyLayer.capability,
            description="Loads a training target model's own tokenizer for exact per-model budgets.",
            target="model",
            python_requires=">=3.10",
            dependency_requirements=[DependencyRequirement(name="tokenizers", specifier=">=0.15")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="capability-data",
            display_name="Columnar data (pyarrow)",
            layer=DependencyLayer.capability,
            description="Parquet import/export (else a clear install hint).",
            target="data",
            python_requires=">=3.10",
            dependency_requirements=[DependencyRequirement(name="pyarrow", specifier=">=15")],
            supported_os=all_os,
            verification=RecipeVerification.hardware_verified,
        ),
        EnvironmentRecipe(
            recipe_id="backend-corpus-studio",
            display_name="Backend: CorpusStudio first-party trainer (TRL/PEFT QLoRA)",
            layer=DependencyLayer.backend_worker,
            description="The reference training backend — TRL + PEFT + bitsandbytes 4-bit QLoRA. This "
            "is the current `[train]` extra, promoted to an isolated backend environment.",
            target="corpus_studio",
            python_requires=">=3.10",
            dependency_requirements=[
                DependencyRequirement(name="torch", specifier=">=2.1"),
                DependencyRequirement(name="transformers", specifier=">=4.44"),
                DependencyRequirement(name="peft", specifier=">=0.11"),
                DependencyRequirement(name="trl", specifier=">=0.9"),
                DependencyRequirement(name="accelerate", specifier=">=0.30"),
                DependencyRequirement(name="datasets", specifier=">=2.19"),
                DependencyRequirement(
                    name="bitsandbytes",
                    specifier=">=0.43; platform_system != 'Darwin'",
                    reason="4-bit QLoRA (CUDA-only; skipped on macOS)",
                ),
            ],
            cuda_index_urls={k: v for k, v in PYTORCH_INDEX_URLS.items()},
            requires_cuda=False,  # the CPU-toy path runs without CUDA; a real GPU run needs a CUDA build
            supported_os=all_os,
            capability_probes=["cuda_available", "bf16_matmul", "bnb_4bit_load", "checkpoint_reload"],
            verification=RecipeVerification.hardware_verified,  # ran on a real RTX 5070 (Blackwell)
            notes=[
                "A default `pip` pulls the CPU torch build; a real GPU run needs the CUDA wheel index.",
                "bitsandbytes is CUDA-only — skipped on macOS (CPU/MPS have no 4-bit path).",
            ],
        ),
        EnvironmentRecipe(
            recipe_id="backend-unsloth",
            display_name="Backend: Unsloth (accelerated QLoRA)",
            layer=DependencyLayer.backend_worker,
            description="Unsloth's fused QLoRA kernels. CUDA-only; pins its own torch/xformers, so it "
            "MUST be an isolated environment (cannot share the corpus_studio env).",
            target="unsloth",
            python_requires=">=3.10",
            dependency_requirements=[
                DependencyRequirement(name="unsloth", reason="brings its own pinned torch/trl/xformers")
            ],
            cuda_index_urls={
                k: v for k, v in PYTORCH_INDEX_URLS.items() if k.startswith("cu")
            },
            requires_cuda=True,
            min_compute_capability="7.5",
            supported_os=[OperatingSystem.linux, OperatingSystem.wsl, OperatingSystem.windows],
            known_conflicts=[
                DependencyConflict(
                    packages=["unsloth", "torch"],
                    condition="pins specific torch/xformers builds; do not co-install with backend-corpus-studio",
                    severity="block",
                )
            ],
            capability_probes=["cuda_available"],
            # We've verified the ABSENCE guard + Blackwell refusal, not the happy training path (our
            # Blackwell host routes AWAY from Unsloth — needs an Ampere/Ada GPU to functionally verify).
            verification=RecipeVerification.declared,
            notes=["Blackwell/sm_120 forces the math attention path Unsloth does not provide — the "
                   "planner refuses Unsloth there and routes to backend-corpus-studio."],
        ),
    ]


def get_recipe(recipe_id: str) -> EnvironmentRecipe | None:
    """The built-in recipe with this id, or None."""
    return next((r for r in builtin_recipes() if r.recipe_id == recipe_id), None)


def recipes_for_layer(layer: DependencyLayer) -> list[EnvironmentRecipe]:
    """Built-in recipes in one dependency layer."""
    return [r for r in builtin_recipes() if r.layer == layer]


def _parse_min_python(python_requires: str) -> tuple[int, int] | None:
    """Parse a ``>=3.10``-style floor into ``(3, 10)``; None when not a simple floor."""
    spec = python_requires.strip()
    if not spec.startswith(">="):
        return None
    parts = spec[2:].strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _python_tuple(python_version: str) -> tuple[int, int] | None:
    parts = python_version.strip().split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _requirement_string(req: DependencyRequirement) -> str:
    """Render a requirement as a single pip argv token, e.g. ``transformers>=4.44``. The specifier may
    carry an environment marker (``; platform_system != 'Darwin'``) — pip evaluates it, so it stays in
    the token."""
    return f"{req.name}{req.specifier or ''}"


def _env_python_path(os_value: OperatingSystem) -> str:
    """The interpreter path inside a freshly-created venv, with a placeholder root the creation step
    substitutes. Windows puts it in Scripts\\; POSIX in bin/."""
    if os_value == OperatingSystem.windows:
        return "<ENV_ROOT>\\Scripts\\python.exe"
    return "<ENV_ROOT>/bin/python"


def _estimate_download_mb(requirements: list[DependencyRequirement], accelerator_tag: str) -> int:
    total = 0
    for req in requirements:
        if req.name in _TORCH_DISTRIBUTIONS:
            total += _TORCH_CPU_MB if accelerator_tag == "cpu" else _TORCH_CUDA_MB
        else:
            total += _DOWNLOAD_MB.get(req.name, _DEFAULT_PACKAGE_MB)
    return total


def resolve_dependencies(
    recipe: EnvironmentRecipe,
    *,
    os_value: OperatingSystem,
    accelerator_tag: str = "cpu",
    python_version: str = "",
) -> DependencyResolution:
    """Render the argv-structured install PREVIEW for provisioning ``recipe`` on this host — the exact
    steps, the CUDA-aware wheel index, and rough disk/network cost — WITHOUT installing anything.

    Layer-aware: a ``backend_worker`` recipe creates its own isolated venv; a ``capability`` recipe
    installs into the existing control-plane interpreter (it augments the core process). ``resolvable``
    is False (with reasons) when the host can't satisfy the recipe — unmet python floor, unsupported
    OS, or a CUDA-required recipe on a host with no CUDA accelerator.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    # --- feasibility ---
    if recipe.supported_os and os_value not in recipe.supported_os:
        supported = ", ".join(o.value for o in recipe.supported_os)
        blocking.append(f"OS '{os_value.value}' is not supported by this recipe (supported: {supported})")

    floor = _parse_min_python(recipe.python_requires)
    have = _python_tuple(python_version)
    if floor is not None and have is not None and have < floor:
        blocking.append(
            f"Python {have[0]}.{have[1]} is below the recipe floor {recipe.python_requires}"
        )

    if recipe.requires_cuda and accelerator_tag == "cpu":
        blocking.append("recipe requires a CUDA accelerator, but none was detected on this host")

    if recipe.requires_native_build:
        warnings.append("needs a native build — a C/C++ compiler toolchain must be present")

    # --- accelerator / index selection ---
    has_torch = any(r.name in _TORCH_DISTRIBUTIONS for r in recipe.dependency_requirements)
    resolved_indexes: list[str] = []
    torch_index: str | None = None
    if has_torch:
        if accelerator_tag in recipe.cuda_index_urls:
            torch_index = recipe.cuda_index_urls[accelerator_tag]
        elif "cpu" in recipe.cuda_index_urls:
            torch_index = recipe.cuda_index_urls["cpu"]
        if torch_index:
            resolved_indexes.append(torch_index)
        if accelerator_tag == "cpu":
            warnings.append("no CUDA selected — installing the CPU PyTorch build (no GPU training)")

    # --- macOS bitsandbytes caveat (its env marker skips it, but say so) ---
    if os_value == OperatingSystem.macos and any(
        "Darwin" in (r.specifier or "") for r in recipe.dependency_requirements
    ):
        warnings.append("bitsandbytes (4-bit QLoRA) is skipped on macOS — no CPU/MPS 4-bit path")

    # --- render the argv steps (never a shell string) ---
    steps = _build_install_steps(recipe, os_value=os_value, torch_index=torch_index)

    download_mb = _estimate_download_mb(recipe.dependency_requirements, accelerator_tag)
    disk_mb = int(download_mb * _DISK_FOOTPRINT_MULTIPLIER)

    return DependencyResolution(
        recipe_ref=Ref(id=recipe.recipe_id),
        python_version=python_version,
        os=os_value,
        accelerator_tag=accelerator_tag,
        resolved_index_urls=resolved_indexes,
        install_steps=steps,
        estimated_download_bytes=download_mb * _MB,
        estimated_disk_bytes=disk_mb * _MB,
        resolvable=not blocking,
        blocking_reasons=blocking,
        warnings=warnings,
    )


def _build_install_steps(
    recipe: EnvironmentRecipe, *, os_value: OperatingSystem, torch_index: str | None
):
    """Build the ordered, argv-structured install steps for a recipe. Imported lazily-free (returns a
    list of InstallStep). Backend workers get an isolated venv; capability profiles install into the
    control-plane interpreter."""
    from .contracts import InstallStep  # noqa: PLC0415 - avoid a forward-ref cycle at module import.

    reqs = recipe.dependency_requirements
    torch_reqs = [r for r in reqs if r.name in _TORCH_DISTRIBUTIONS]
    other_reqs = [r for r in reqs if r.name not in _TORCH_DISTRIBUTIONS]

    if recipe.layer == DependencyLayer.backend_worker:
        env_py = _env_python_path(os_value)
        steps = [
            InstallStep(
                phase="create_venv",
                description="Create an isolated virtual environment for this backend",
                argv=["<BASE_PYTHON>", "-m", "venv", "<ENV_ROOT>"],
            ),
            InstallStep(
                phase="upgrade_pip",
                description="Upgrade pip in the new environment",
                argv=[env_py, "-m", "pip", "install", "--upgrade", "pip"],
            ),
        ]
        if torch_reqs and torch_index:
            steps.append(
                InstallStep(
                    phase="install",
                    description="Install PyTorch from its accelerator-specific wheel index",
                    argv=[env_py, "-m", "pip", "install", "--index-url", torch_index]
                    + [_requirement_string(r) for r in torch_reqs],
                )
            )
        if other_reqs:
            steps.append(
                InstallStep(
                    phase="install",
                    description="Install the remaining dependencies from PyPI",
                    argv=[env_py, "-m", "pip", "install"] + [_requirement_string(r) for r in other_reqs],
                )
            )
        return steps

    # capability / control_plane: augment the existing control-plane interpreter (no new venv).
    all_reqs = torch_reqs + other_reqs
    return [
        InstallStep(
            phase="install",
            description="Install into the control-plane environment (augments the core process)",
            argv=["<CONTROL_PLANE_PYTHON>", "-m", "pip", "install"]
            + [_requirement_string(r) for r in all_reqs],
        )
    ]


def select_accelerator_tag(
    cuda_runtime_version: str | None, compute_capability_major: int | None, has_gpu: bool
) -> str:
    """Pick the PyTorch wheel tag for a host. Prefers the installed CUDA runtime version; falls back to
    the GPU's compute-capability major; ``cpu`` when there is no GPU. Returns a key of
    :data:`PYTORCH_INDEX_URLS`."""
    if not has_gpu:
        return "cpu"
    if cuda_runtime_version:
        major_minor = ".".join(cuda_runtime_version.split(".")[:2])
        by_runtime = {"12.8": "cu128", "12.6": "cu126", "12.4": "cu126", "12.1": "cu121", "11.8": "cu118"}
        if major_minor in by_runtime:
            return by_runtime[major_minor]
    if compute_capability_major is not None:
        if compute_capability_major >= 12:  # Blackwell
            return "cu128"
        if compute_capability_major >= 8:  # Ampere / Ada / Hopper
            return "cu121"
        return "cu118"
    return "cu121"  # a reasonable default for an unknown CUDA GPU
