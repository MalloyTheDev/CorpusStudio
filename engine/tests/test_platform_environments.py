"""Environment Manager substrate — the recipe registry + the pure install-preview resolver.

Everything here is pure: it renders the argv-structured install PLAN without creating a venv or
installing anything, so the whole 3-layer dependency model is provable in CI with no heavy deps.
"""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from corpus_studio.cli import app
from corpus_studio.platform.contracts import (
    DependencyRequirement,
    DependencyResolution,
    EnvironmentDescriptor,
    EnvironmentHealthReport,
    EnvironmentLock,
    EnvironmentRecipe,
)
from corpus_studio.platform.common import Ref
from corpus_studio.platform.enums import (
    DependencyLayer,
    EnvironmentState,
    OperatingSystem,
    RecipeVerification,
)
from corpus_studio.platform.environments import (
    PYPI_INDEX_URL,
    PYTORCH_INDEX_URLS,
    _parse_min_python,
    _python_tuple,
    builtin_recipes,
    get_recipe,
    recipes_for_layer,
    resolve_dependencies,
    select_accelerator_tag,
)

runner = CliRunner()


# ---- recipe registry ---------------------------------------------------------


def test_builtin_recipes_have_unique_ids_and_cover_all_three_layers():
    recipes = builtin_recipes()
    ids = [r.recipe_id for r in recipes]
    assert len(ids) == len(set(ids))  # unique
    layers = {r.layer for r in recipes}
    assert layers == {
        DependencyLayer.control_plane,
        DependencyLayer.capability,
        DependencyLayer.backend_worker,
    }


def test_control_plane_recipe_pulls_no_ml_framework():
    control = get_recipe("control-plane")
    assert control is not None and control.layer == DependencyLayer.control_plane
    names = {d.name for d in control.dependency_requirements}
    assert "torch" not in names and "transformers" not in names  # the core stays lightweight


def test_corpus_studio_backend_recipe_matches_the_train_extra():
    recipe = get_recipe("backend-corpus-studio")
    assert recipe is not None and recipe.layer == DependencyLayer.backend_worker
    names = {d.name for d in recipe.dependency_requirements}
    assert {"torch", "transformers", "peft", "trl", "bitsandbytes"} <= names
    assert recipe.verification == RecipeVerification.hardware_verified  # ran on a real 5070


def test_readiness_v2_recipe_is_exact_and_probe_changes_reseal_the_plan():
    recipe = get_recipe("backend-corpus-studio-readiness-v2")
    assert recipe is not None and recipe.required_execution_probe is not None
    assert recipe.requires_worker_wheel is True
    assert recipe.verification == RecipeVerification.declared
    assert all(
        requirement.specifier and requirement.specifier.startswith("==")
        for requirement in recipe.dependency_requirements
    )
    first = resolve_dependencies(
        recipe,
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    changed_probe = recipe.required_execution_probe.model_copy(
        update={
            "required_distributions": sorted(
                recipe.required_execution_probe.required_distributions + ["x-runtime"]
            )
        }
    )
    changed_recipe = recipe.model_copy(
        update={"required_execution_probe": changed_probe}
    )
    second = resolve_dependencies(
        changed_recipe,
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    assert first.recipe_ref.hash != second.recipe_ref.hash
    assert first.resolution_hash != second.resolution_hash


def test_readiness_flash_v1_is_exact_forced_flash_and_independent_of_math():
    math_recipe = get_recipe("backend-corpus-studio-readiness-v2")
    flash_recipe = get_recipe("backend-corpus-studio-readiness-flash-v1")
    assert math_recipe is not None and flash_recipe is not None
    assert flash_recipe.required_execution_probe is not None
    assert flash_recipe.requires_worker_wheel is True
    assert flash_recipe.verification == RecipeVerification.declared
    assert flash_recipe.required_execution_probe.probe == "cuda_qlora_sdpa_flash_execution"
    assert flash_recipe.required_execution_probe.flash_sdp_enabled is True
    assert flash_recipe.required_execution_probe.math_sdp_enabled is False
    assert (
        flash_recipe.required_execution_probe.execution_combination.attention_kernel.value
        == "torch_sdpa_flash"
    )
    # Flash readiness is a Linux-positive path (native Windows WDDM flash is refused elsewhere).
    assert flash_recipe.supported_os == [OperatingSystem.linux]
    assert flash_recipe.requires_cuda is True
    assert all(
        requirement.specifier and requirement.specifier.startswith("==")
        for requirement in flash_recipe.dependency_requirements
    )
    math_resolution = resolve_dependencies(
        math_recipe,
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    flash_resolution = resolve_dependencies(
        flash_recipe,
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    assert math_resolution.recipe_ref.hash != flash_resolution.recipe_ref.hash
    assert math_resolution.resolution_hash != flash_resolution.resolution_hash
    # Existing math and flash evidence retain their recipe identities; the concrete new install
    # step is bound by the resolution hash instead.
    from corpus_studio.platform.environments import recipe_digest

    assert (
        recipe_digest(math_recipe)
        == "4c0cb365b596cfe2b1371afd5f95130a40e41c7e5b27df833b0c914bd492289c"
    )
    assert (
        recipe_digest(flash_recipe)
        == "52016adedd5011328efb05e089d54c8edd5c9308e0a38409897cd0f554240fb7"
    )


@pytest.mark.parametrize(
    "recipe_id",
    ["backend-corpus-studio-readiness-v2", "backend-corpus-studio-readiness-flash-v1"],
)
def test_readiness_pins_hash_backed_pytorch_prerequisites_before_cuda_torch(recipe_id):
    recipe = get_recipe(recipe_id)
    assert recipe is not None
    resolution = resolve_dependencies(
        recipe,
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    prerequisite_step = next(
        step
        for step in resolution.install_steps
        if step.evidence_path
        and step.evidence_path.endswith("install-pytorch-prerequisites.json")
    )
    torch_step = _torch_install_step(resolution)

    assert resolution.install_steps.index(prerequisite_step) < resolution.install_steps.index(
        torch_step
    )
    assert prerequisite_step.configured_index_urls == [PYPI_INDEX_URL]
    assert prerequisite_step.argv[prerequisite_step.argv.index("--index-url") + 1] == PYPI_INDEX_URL
    assert "--no-deps" in prerequisite_step.argv
    assert prerequisite_step.argv[-5:] == [
        "cuda-pathfinder==1.2.2",
        "setuptools==78.1.0",
        "typing-extensions==4.15.0",
        "jinja2==3.1.6",
        "markupsafe==3.0.3",
    ]
    assert torch_step.configured_index_urls == [PYTORCH_INDEX_URLS["cu128"]]
    assert all(requirement not in torch_step.argv for requirement in prerequisite_step.argv[-5:])


def test_unpinned_backend_does_not_invent_version_specific_torch_prerequisites():
    resolution = resolve_dependencies(
        _corpus_studio(),
        os_value=OperatingSystem.linux,
        accelerator_tag="cu128",
        python_version="3.12",
    )
    assert all(
        not step.evidence_path
        or not step.evidence_path.endswith("install-pytorch-prerequisites.json")
        for step in resolution.install_steps
    )


def test_complete_qlora_tuple_uses_bf16_autocast_without_sdpa_fallback():
    """Forced flash/math QLoRA probes must match trainer bf16 compute and never soft-fallback.

    Live host evidence: PEFT k-bit prep leaves float32 residual tensors; forced FLASH_ATTENTION
    aborts without CUDA bf16 autocast and must not enable math/mem-efficient as a substitute.
    """
    import inspect

    from corpus_studio.platform import probes as probes_mod

    source = inspect.getsource(probes_mod._probe_cuda_qlora_sdpa_execution_tuple)
    assert "torch.autocast" in source
    assert "bfloat16" in source
    assert "No fallback to math / mem-efficient / eager" in source
    assert "enable_mem_efficient_sdp(False)" in source
    # Flash and math share the tuple helper; flash entrypoint forces FLASH only.
    flash_src = inspect.getsource(probes_mod._probe_cuda_qlora_sdpa_flash_execution)
    assert "FLASH_ATTENTION" in flash_src
    assert "enable_math=False" in flash_src
    math_src = inspect.getsource(probes_mod._probe_cuda_qlora_math_execution)
    assert "enable_flash=False" in math_src
    assert "enable_math=True" in math_src
    assert "adapter_round_trip_verified" in source
    assert "_placement_deviation" in source
    assert "get_rng_state_all" in source


def test_complete_qlora_probe_cleanup_restores_every_global_and_stops_sampler():
    from corpus_studio.platform.probes import _restore_qlora_probe_process_state

    calls = []

    class Stop:
        def set(self):
            calls.append(("stop", True))

    class Thread:
        def join(self, *, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return False

    class Cuda:
        class Backend:
            @staticmethod
            def enable_flash_sdp(value):
                calls.append(("flash", value))

            @staticmethod
            def enable_mem_efficient_sdp(value):
                calls.append(("memory", value))

            @staticmethod
            def enable_math_sdp(value):
                calls.append(("math", value))

        def __init__(self):
            self.set_rng_state_all = lambda value: calls.append(("cuda_rng", value))

    cuda = Cuda()
    cuda.enable_flash_sdp = cuda.Backend.enable_flash_sdp
    cuda.enable_mem_efficient_sdp = cuda.Backend.enable_mem_efficient_sdp
    cuda.enable_math_sdp = cuda.Backend.enable_math_sdp
    torch = type(
        "Torch",
        (),
        {
            "backends": type("Backends", (), {"cuda": cuda})(),
            "cuda": cuda,
            "random": type(
                "Random",
                (),
                {"set_rng_state": staticmethod(lambda value: calls.append(("cpu_rng", value)))},
            )(),
        },
    )()
    _restore_qlora_probe_process_state(
        torch,
        previous_toggles=(True, False, True),
        cpu_rng_state="cpu-state",
        cuda_rng_states="cuda-state",
        sampler_stop=Stop(),
        sampler_thread=Thread(),
    )
    assert calls == [
        ("stop", True),
        ("join", 3),
        ("flash", True),
        ("memory", False),
        ("math", True),
        ("cpu_rng", "cpu-state"),
        ("cuda_rng", "cuda-state"),
    ]


def test_complete_qlora_probe_cleanup_attempts_all_restores_before_failing():
    from corpus_studio.platform.probes import _restore_qlora_probe_process_state

    calls = []

    class Thread:
        def join(self, *, timeout):
            calls.append(("join", timeout))

        def is_alive(self):
            return True

    def broken_flash(value):
        calls.append(("flash", value))
        raise RuntimeError("simulated")

    cuda_backend = type(
        "CudaBackend",
        (),
        {
            "enable_flash_sdp": staticmethod(broken_flash),
            "enable_mem_efficient_sdp": staticmethod(
                lambda value: calls.append(("memory", value))
            ),
            "enable_math_sdp": staticmethod(lambda value: calls.append(("math", value))),
        },
    )()
    cuda = type(
        "Cuda",
        (),
        {"set_rng_state_all": staticmethod(lambda value: calls.append(("cuda_rng", value)))},
    )()
    torch = type(
        "Torch",
        (),
        {
            "backends": type("Backends", (), {"cuda": cuda_backend})(),
            "cuda": cuda,
            "random": type(
                "Random",
                (),
                {"set_rng_state": staticmethod(lambda value: calls.append(("cpu_rng", value)))},
            )(),
        },
    )()
    with pytest.raises(RuntimeError, match="sampler did not terminate.*flash SDPA"):
        _restore_qlora_probe_process_state(
            torch,
            previous_toggles=(True, False, True),
            cpu_rng_state="cpu",
            cuda_rng_states="cuda",
            sampler_stop=None,
            sampler_thread=Thread(),
        )
    assert ("memory", False) in calls
    assert ("math", True) in calls
    assert ("cpu_rng", "cpu") in calls
    assert ("cuda_rng", "cuda") in calls


def test_unsloth_recipe_is_declared_cuda_only_and_conflict_flagged():
    recipe = get_recipe("backend-unsloth")
    assert recipe is not None
    assert recipe.requires_cuda is True
    assert recipe.verification == RecipeVerification.declared  # happy path not verified on our Blackwell
    assert recipe.known_conflicts  # pins torch/xformers → can't share the corpus_studio env
    assert OperatingSystem.macos not in recipe.supported_os


def test_recipes_for_layer_filters():
    caps = recipes_for_layer(DependencyLayer.capability)
    assert caps and all(r.layer == DependencyLayer.capability for r in caps)


def test_get_recipe_unknown_returns_none():
    assert get_recipe("does-not-exist") is None


# ---- resolver: argv, CUDA index, feasibility ---------------------------------


def _corpus_studio():
    recipe = get_recipe("backend-corpus-studio")
    assert recipe is not None
    return recipe


def _torch_install_step(resolution):
    return next(
        step
        for step in resolution.install_steps
        if any(token.startswith("torch") for token in step.argv)
    )


def test_backend_resolution_creates_a_venv_and_uses_the_cuda_index():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.linux, accelerator_tag="cu128", python_version="3.12"
    )
    assert res.resolvable is True
    phases = [s.phase for s in res.install_steps]
    assert phases[0] == "create_venv"  # a backend gets its OWN isolated env
    # torch installs from the cu128 wheel index, as separate argv tokens (never a shell string).
    torch_step = _torch_install_step(res)
    idx = torch_step.argv.index("--index-url")
    assert torch_step.argv[idx + 1] == PYTORCH_INDEX_URLS["cu128"]
    assert any(tok.startswith("torch") for tok in torch_step.argv)
    assert PYTORCH_INDEX_URLS["cu128"] in res.resolved_index_urls
    assert PYPI_INDEX_URL in res.resolved_index_urls
    assert all("--isolated" in step.argv for step in res.install_steps if step.phase != "create_venv")


def test_every_install_step_is_argv_structured_not_a_shell_string():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.windows, accelerator_tag="cu128", python_version="3.12"
    )
    for step in res.install_steps:
        assert isinstance(step.argv, list) and len(step.argv) >= 1
        assert all(isinstance(tok, str) for tok in step.argv)
    # The bitsandbytes env-marker ';' lives INSIDE a single argv token — argv is a list, so pip gets it
    # literally and no shell ever interprets the ';' (the no-shell guarantee is the list structure).
    all_tokens = [tok for step in res.install_steps for tok in step.argv]
    marker = next(tok for tok in all_tokens if tok.startswith("bitsandbytes"))
    assert marker == "bitsandbytes>=0.43; platform_system != 'Darwin'"
    # Windows venv interpreter path is under Scripts\.
    pip_step = next(s for s in res.install_steps if s.phase == "upgrade_pip")
    assert "Scripts" in pip_step.argv[0]


def test_cpu_accelerator_warns_and_uses_cpu_torch_index():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.linux, accelerator_tag="cpu", python_version="3.12"
    )
    assert any("CPU" in w for w in res.warnings)
    torch_step = _torch_install_step(res)
    idx = torch_step.argv.index("--index-url")
    assert torch_step.argv[idx + 1] == PYTORCH_INDEX_URLS["cpu"]


def test_macos_flags_the_bitsandbytes_skip():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.macos, accelerator_tag="cpu", python_version="3.12"
    )
    assert any("bitsandbytes" in w for w in res.warnings)


def test_python_floor_unmet_is_unresolvable():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.linux, accelerator_tag="cu128", python_version="3.9"
    )
    assert res.resolvable is False
    assert any("below the recipe floor" in r for r in res.blocking_reasons)


def test_cuda_required_recipe_on_cpu_host_is_unresolvable():
    res = resolve_dependencies(
        get_recipe("backend-unsloth"),  # requires_cuda=True
        os_value=OperatingSystem.linux,
        accelerator_tag="cpu",
        python_version="3.12",
    )
    assert res.resolvable is False
    assert any("CUDA" in r for r in res.blocking_reasons)


def test_unsupported_os_is_unresolvable():
    res = resolve_dependencies(
        get_recipe("backend-unsloth"),  # no macOS support
        os_value=OperatingSystem.macos,
        accelerator_tag="cpu",
        python_version="3.12",
    )
    assert res.resolvable is False
    assert any("not supported" in r for r in res.blocking_reasons)


def test_capability_recipe_installs_into_the_control_plane_not_a_new_venv():
    res = resolve_dependencies(
        get_recipe("capability-tokenization"),
        os_value=OperatingSystem.linux,
        accelerator_tag="cpu",
        python_version="3.12",
    )
    assert res.resolvable is True
    phases = [s.phase for s in res.install_steps]
    assert "create_venv" not in phases  # a capability augments the core process, not a fresh env
    assert any("CONTROL_PLANE" in " ".join(s.argv) for s in res.install_steps)


def test_size_estimates_are_present_and_disk_exceeds_download():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.linux, accelerator_tag="cu128", python_version="3.12"
    )
    assert res.estimated_download_bytes and res.estimated_download_bytes > 0
    assert res.estimated_disk_bytes and res.estimated_disk_bytes > res.estimated_download_bytes


def test_native_build_recipe_warns():
    recipe = EnvironmentRecipe(
        recipe_id="synthetic-native",
        layer=DependencyLayer.backend_worker,
        requires_native_build=True,
        dependency_requirements=[DependencyRequirement(name="deepspeed")],
        supported_os=[OperatingSystem.linux],
    )
    res = resolve_dependencies(
        recipe, os_value=OperatingSystem.linux, accelerator_tag="cu128", python_version="3.12"
    )
    assert any("native build" in w for w in res.warnings)


def test_unknown_accelerator_tag_falls_back_to_cpu_index():
    res = resolve_dependencies(
        _corpus_studio(), os_value=OperatingSystem.linux, accelerator_tag="cu999", python_version="3.12"
    )
    torch_step = _torch_install_step(res)
    idx = torch_step.argv.index("--index-url")
    assert torch_step.argv[idx + 1] == PYTORCH_INDEX_URLS["cpu"]


def test_python_version_parsers_handle_odd_inputs():
    assert _parse_min_python("==3.11") is None  # not a floor specifier
    assert _parse_min_python(">=oops") is None
    assert _python_tuple("weird") is None
    assert _python_tuple("3.12.1") == (3, 12)


# ---- accelerator selection ---------------------------------------------------


def test_select_accelerator_tag():
    assert select_accelerator_tag(None, None, has_gpu=False) == "cpu"
    assert select_accelerator_tag("12.8", 12, has_gpu=True) == "cu128"
    assert select_accelerator_tag(None, 12, has_gpu=True) == "cu128"  # Blackwell by cc
    assert select_accelerator_tag(None, 8, has_gpu=True) == "cu121"  # Ampere/Ada
    assert select_accelerator_tag(None, 7, has_gpu=True) == "cu118"  # older CUDA arch
    assert select_accelerator_tag(None, None, has_gpu=True) == "cu121"  # unknown CUDA GPU default


# ---- CLI (env-recipes / env-plan) --------------------------------------------


def test_env_recipes_cli_lists_and_filters():
    result = runner.invoke(app, ["env-recipes", "--json"])
    assert result.exit_code == 0
    ids = {r["recipe_id"] for r in json.loads(result.stdout)}
    assert "backend-corpus-studio" in ids and "control-plane" in ids

    filtered = runner.invoke(app, ["env-recipes", "--layer", "backend_worker", "--json"])
    layers = {r["layer"] for r in json.loads(filtered.stdout)}
    assert layers == {"backend_worker"}

    bad = runner.invoke(app, ["env-recipes", "--layer", "nonsense"])
    assert bad.exit_code == 2


def test_env_plan_cli_previews_a_backend_install():
    result = runner.invoke(
        app,
        [
            "env-plan",
            "backend-corpus-studio",
            "--accelerator",
            "cu128",
            "--runtime",
            sys.executable,
            "--json",
        ],
    )
    assert result.exit_code == 0
    resolution = json.loads(result.stdout)
    assert resolution["resolvable"] is True
    assert any(step["phase"] == "create_venv" for step in resolution["install_steps"])


def test_env_plan_cli_unknown_recipe_exits_2():
    result = runner.invoke(app, ["env-plan", "no-such-recipe"])
    assert result.exit_code == 2


def test_env_plan_cli_unresolvable_exits_1():
    # Unsloth requires CUDA; forcing --accelerator cpu makes the plan unresolvable → exit 1.
    result = runner.invoke(app, ["env-plan", "backend-unsloth", "--accelerator", "cpu", "--python", "3.12"])
    assert result.exit_code == 1
    assert "BLOCKED" in result.stdout


# ---- contract round-trips ----------------------------------------------------


def test_environment_contracts_round_trip():
    recipe = get_recipe("backend-corpus-studio")
    assert EnvironmentRecipe.model_validate_json(recipe.model_dump_json()).recipe_id == "backend-corpus-studio"

    res = resolve_dependencies(recipe, os_value=OperatingSystem.linux, accelerator_tag="cu128", python_version="3.12")
    assert DependencyResolution.model_validate_json(res.model_dump_json()).resolvable is True

    lock = EnvironmentLock(lock_id="lock-1", recipe_ref=Ref(id="backend-corpus-studio"), python_version="3.12")
    assert EnvironmentLock.model_validate_json(lock.model_dump_json()).lock_id == "lock-1"

    desc = EnvironmentDescriptor(
        env_id="env-1", recipe_ref=Ref(id="backend-corpus-studio"), layer=DependencyLayer.backend_worker
    )
    assert desc.state == EnvironmentState.not_installed  # a fresh descriptor is NOT_INSTALLED

    health = EnvironmentHealthReport(environment_ref=Ref(id="env-1"), state=EnvironmentState.importable)
    assert EnvironmentHealthReport.model_validate_json(health.model_dump_json()).drift_detected is False
