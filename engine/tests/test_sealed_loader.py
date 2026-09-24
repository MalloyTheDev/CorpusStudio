"""The newer workers lower the pinned model/tokenizer identity and the sealed loader policy (#863).

``training.sealed_loader`` maps a sealed DPO, reward, full-parameter SFT or on-policy RL execution onto the
adapter SFT lane's loader helpers. These tests drive those helpers and the REAL worker bodies
(``run_preference``, ``run_reward``, ``run_full_finetune``, ``run_rollout``) against a deterministic fake
torch/Transformers/PEFT/bitsandbytes stack (``_sealed_loader_fakes``) whose models report exactly what
they were loaded with. Each seal comes from the real planner and is re-sealed with a DISTINCT tokenizer
binding (``review/separate-tokenizer`` at commit ``b * 40``) next to the model binding (``review/model``
at commit ``a * 40``), so asset selection and both revisions are observable in the exact loader calls.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _sealed_loader_fakes import (
    FakeBitsAndBytesConfig,
    FakeLoadedModel,
    FakeStack,
    FakeTorch,
    StopAtKbit,
)
from test_platform_planner import _report
from test_sealed_loader_admission import _execution, _local_model, _reseal, _sealed_plan

import corpus_studio.training.optimizer_config as optimizer_config
import corpus_studio.training.pretraining_evidence as pretraining_evidence
import corpus_studio.training.preference_evidence as preference_evidence
import corpus_studio.training.reward_evidence as reward_evidence
import corpus_studio.training.rollout_evidence as rollout_evidence
import corpus_studio.training.trainer as trainer_module
from corpus_studio.platform.contracts import ExecutionInputBinding
from corpus_studio.platform.execution_config import huggingface_input_ref, stable_directory_sha256
from corpus_studio.training.sealed_inputs import read_verified_dataset
from corpus_studio.training.sealed_loader import (
    describe_sealed_identity,
    load_sealed_model,
    load_sealed_tokenizer,
    prepare_sealed_attention,
    sealed_loader_view,
    sealed_tokenizer_load_args,
    sealed_truncation_permitted,
    verify_full_parameter_storage,
    verify_sealed_adapter_precision,
    verify_sealed_chat_template,
    verify_sealed_model_after_load,
)
from corpus_studio.training.trainer import (
    ExecutionPlacementDeviation,
    TrainerEnvironmentError,
    TrainerError,
    TrainingEvidenceError,
)

A40 = "a" * 40
B40 = "b" * 40
MODEL = "review/model"
TOKENIZER = "review/separate-tokenizer"
TOKENIZER_ARGS = (TOKENIZER, {"trust_remote_code": False, "revision": B40})


def _hf(kind: str, repo: str, revision: str) -> dict[str, Any]:
    return ExecutionInputBinding(
        kind=kind,  # type: ignore[arg-type]
        ref=huggingface_input_ref(kind, repo, revision),
        source="huggingface",
        location=repo,
        resolved_revision=revision,
    ).model_dump(mode="json")


def _distinct_identity(payload: dict[str, Any]) -> None:
    payload["inputs"]["model"] = _hf("model", MODEL, A40)
    payload["inputs"]["tokenizer"] = _hf("tokenizer", TOKENIZER, B40)
    if "reward_source" in payload:
        payload["reward_source"]["reward_base_model"] = MODEL


def _sealed(tmp_path: Path, lane: str, *more: Callable[[dict[str, Any]], None], **plan_kw: Any):
    execution = _reseal(_execution(_sealed_plan(tmp_path, lane, **plan_kw)), _distinct_identity)
    for mutate in more:
        execution = _reseal(execution, mutate)
    return execution


def _precision(**updates: Any) -> Callable[[dict[str, Any]], None]:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["precision"].update(updates)

    return _mutate


class _Stages:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, name: str, message: str) -> None:
        assert message.isascii()
        self.calls.append((name, message))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _qlora_load_kwargs(stack: FakeStack, *, compute: str = "bfloat16") -> dict[str, Any]:
    return {
        "trust_remote_code": False,
        "revision": A40,
        "use_safetensors": True,
        "quantization_config": FakeBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(stack.torch, compute),
            bnb_4bit_use_double_quant=True,
        ),
        "device_map": {"": "cuda:0"},
        "attn_implementation": "sdpa",
    }


# --- the loader view ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("lane", ["preference", "reward", "full_finetune", "rollout"])
def test_the_view_maps_every_sealed_loader_field(tmp_path, lane):
    execution = _sealed(tmp_path, lane)
    view = sealed_loader_view(execution)

    assert (view.base_model, view.model_revision, view.model_source) == (MODEL, A40, "huggingface")
    assert (view.tokenizer_location, view.tokenizer_revision, view.tokenizer_source) == (
        TOKENIZER,
        B40,
        "huggingface",
    )
    assert view.model_content_sha256 is None and view.tokenizer_content_sha256 is None
    assert view.execution_configuration_hash == execution.configuration_hash
    assert (view.dataset_path, view.dataset_sha256) == (
        execution.inputs.dataset.location,
        execution.inputs.dataset.content_sha256,
    )
    assert view.device_map == {"": "cuda:0"} and view.cpu_toy is False
    assert (view.attn_implementation, view.attention_kernel) == ("sdpa", "torch_sdpa_math")
    assert (view.flash_sdp_enabled, view.mem_efficient_sdp_enabled, view.math_sdp_enabled) == (
        False,
        False,
        True,
    )
    assert view.use_safetensors is True and view.trust_remote_code is False
    assert view.forward_compute_dtype == "bf16" and view.dequantization_dtype == "bf16"
    assert view.sequence_len == execution.sequence.max_sequence_len
    assert view.export_format == execution.export_format.value
    assert view.adapter_task_type == execution.adapter_task_type
    if lane == "full_finetune":
        assert (view.quantization_mode, view.weight_storage_dtype) == ("none", "bf16")
        assert view.bnb_4bit_use_double_quant is False
        assert (view.dataset_format, view.formatter_id) == (
            execution.data.dataset_format,
            execution.data.formatter_id,
        )
    else:
        assert (view.quantization_mode, view.weight_storage_dtype) == ("nf4", None)
        assert view.master_weight_dtype == "fp32" and view.bnb_4bit_use_double_quant is True


def _subdir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def test_tokenizer_arguments_select_the_tokenizer_binding(tmp_path):
    view = sealed_loader_view(_sealed(_subdir(tmp_path, "hf"), "preference"))
    assert sealed_tokenizer_load_args(view) == TOKENIZER_ARGS

    model = _local_model(tmp_path)
    local = sealed_loader_view(
        _execution(
            _sealed_plan(
                tmp_path,
                "preference",
                base_model=str(model),
                model_content_sha256=stable_directory_sha256(model),
            )
        )
    )
    assert sealed_tokenizer_load_args(local) == (str(model), {"trust_remote_code": False})


def test_the_identity_description_names_both_pinned_bindings(tmp_path):
    view = sealed_loader_view(_sealed(_subdir(tmp_path, "hf"), "reward"))
    described = describe_sealed_identity(view)
    assert described == (
        f"model {MODEL}@{A40} (huggingface), tokenizer {TOKENIZER}@{B40} (huggingface), "
        "use_safetensors=True, trust_remote_code=False"
    )

    model = _local_model(tmp_path)
    digest = stable_directory_sha256(model)
    local = sealed_loader_view(
        _execution(
            _sealed_plan(tmp_path, "reward", base_model=str(model), model_content_sha256=digest)
        )
    )
    assert f"model {model} sha256:{digest} (local_directory)" in describe_sealed_identity(local)
    no_pin = local.model_copy(update={"model_content_sha256": None})
    assert f"model {model} (local_directory)" in describe_sealed_identity(no_pin)


def test_a_sealed_chat_template_digest_is_verified_on_the_loaded_tokenizer():
    template = "{{ messages }}"
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    good = types.SimpleNamespace(chat_template=template, apply_chat_template=lambda *_a: "")

    verify_sealed_chat_template(good, digest)
    verify_sealed_chat_template(types.SimpleNamespace(), None)  # pinned through the tokenizer binding
    with pytest.raises(TrainerError, match="changed after planning"):
        verify_sealed_chat_template(good, "0" * 64)
    with pytest.raises(TrainerError, match="no usable chat template"):
        verify_sealed_chat_template(types.SimpleNamespace(chat_template=""), digest)
    with pytest.raises(TrainerError, match="cannot apply its chat template"):
        verify_sealed_chat_template(types.SimpleNamespace(chat_template=template), digest)


def test_load_sealed_tokenizer_uses_the_tokenizer_binding_and_verifies_the_template(
    tmp_path, monkeypatch
):
    stack = FakeStack(monkeypatch)
    stages = _Stages()
    view = sealed_loader_view(_sealed(tmp_path, "rollout"))

    tokenizer = load_sealed_tokenizer(stack.transformers.AutoTokenizer, view, stage=stages)

    assert tokenizer.location == TOKENIZER
    assert stack.loads() == [("AutoTokenizer", *TOKENIZER_ARGS)]
    assert stages.names == ["tokenizer_load", "tokenizer_load"]
    assert describe_sealed_identity(view) in stages.calls[0][1]
    with pytest.raises(TrainerError, match="chat template changed after planning"):
        load_sealed_tokenizer(
            stack.transformers.AutoTokenizer,
            view.model_copy(update={"chat_template_sha256": "0" * 64}),
            stage=stages,
        )


# --- attention: toggles applied and observed, kernel probed in isolation, before any weights -----------


def _cpu_toy_report():
    return _report(readiness="cpu_toy_only")


def _attention(kernel: str, api: str, toggles: tuple[bool, bool, bool]):
    def _mutate(payload: dict[str, Any]) -> None:
        payload["attention"].update(
            model_attention_api=api,
            effective_backend_required=kernel,
            flash_sdp_enabled=toggles[0],
            mem_efficient_sdp_enabled=toggles[1],
            math_sdp_enabled=toggles[2],
        )

    return _mutate


@pytest.mark.parametrize(
    ("mutate", "toggles", "backend"),
    [
        (None, (False, False, True), ("MATH",)),
        (
            _attention("torch_sdpa_flash", "sdpa", (True, False, False)),
            (True, False, False),
            ("FLASH_ATTENTION",),
        ),
    ],
)
def test_prepare_sealed_attention_applies_toggles_and_probes_the_exact_kernel(
    tmp_path, monkeypatch, mutate, toggles, backend
):
    torch = FakeStack(monkeypatch).torch  # installed, so the probe's torch.nn imports resolve to it
    stages = _Stages()
    execution = _sealed(tmp_path, "preference", *([mutate] if mutate else []))
    view = sealed_loader_view(execution)

    kernel = prepare_sealed_attention(torch, view, stage=stages)

    assert kernel == execution.attention.effective_backend_required.value
    assert torch.sdp_state() == toggles
    assert ("probe_sdpa", backend) in torch.events
    assert ("probe_randn", "cuda", "bfloat16", True) in torch.events
    assert stages.names == ["attention_policy_applied", "attention_policy_applied"]
    assert "passed its isolated forward/backward probe" in stages.calls[-1][1]


def test_an_eager_seal_applies_the_math_fallback_and_has_no_sdpa_probe(tmp_path):
    torch = FakeTorch(cuda_available=False)
    stages = _Stages()
    view = sealed_loader_view(
        _execution(
            _sealed_plan(
                tmp_path, "full_finetune", report=_cpu_toy_report(), allow_cpu_toy=True
            )
        )
    )

    assert prepare_sealed_attention(torch, view, stage=stages) == "eager"
    assert torch.sdp_state() == (False, False, True)
    assert not [event for event in torch.events if event[0].startswith("probe")]
    assert "no SDPA probe applies" in stages.calls[-1][1]


def test_attention_preparation_refuses_an_unobservable_or_unprobeable_runtime(tmp_path):
    view = sealed_loader_view(_sealed(tmp_path, "reward"))
    with pytest.raises(TrainerEnvironmentError, match="cannot be probed without CUDA"):
        prepare_sealed_attention(FakeTorch(cuda_available=False), view, stage=_Stages())
    no_toggles = FakeTorch()
    no_toggles.backends = types.SimpleNamespace(cuda=None)
    with pytest.raises(TrainerEnvironmentError, match="cannot enforce and observe"):
        prepare_sealed_attention(no_toggles, view, stage=_Stages())


# --- model load: every sealed loader field lowered, then observed ------------------------------------------


def test_the_qlora_load_lowers_revision_safetensors_quantization_placement_and_attention(
    tmp_path, monkeypatch
):
    stack = FakeStack(monkeypatch)
    stages = _Stages()
    view = sealed_loader_view(_sealed(tmp_path, "preference"))

    model = load_sealed_model(
        stack.AutoModelForCausalLM,
        stack.torch,
        view,
        stage=stages,
        bitsandbytes_config_cls=FakeBitsAndBytesConfig,
    )

    assert stack.loads() == [("AutoModelForCausalLM", MODEL, _qlora_load_kwargs(stack))]
    assert model.config._attn_implementation == "sdpa"
    assert stages.names == ["model_load", "model_load", "placement_verified"]


def test_the_load_honours_a_sealed_fp32_compute_dtype_and_a_score_head_shape(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    execution = _sealed(
        tmp_path, "reward", _precision(dequantization_dtype="fp32", forward_compute_dtype="fp32")
    )

    load_sealed_model(
        stack.AutoModelForSequenceClassification,
        stack.torch,
        sealed_loader_view(execution),
        stage=_Stages(),
        bitsandbytes_config_cls=FakeBitsAndBytesConfig,
        num_labels=1,
    )

    expected = {**_qlora_load_kwargs(stack, compute="float32"), "num_labels": 1}
    assert stack.loads() == [("AutoModelForSequenceClassification", MODEL, expected)]


def test_the_full_parameter_load_uses_the_sealed_storage_dtype(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    execution = _sealed(
        tmp_path,
        "full_finetune",
        _precision(
            weight_storage_dtype="fp16", dequantization_dtype="fp16", forward_compute_dtype="fp16"
        ),
    )

    load_sealed_model(
        stack.AutoModelForCausalLM, stack.torch, sealed_loader_view(execution), stage=_Stages()
    )

    assert stack.loads() == [
        (
            "AutoModelForCausalLM",
            MODEL,
            {
                "trust_remote_code": False,
                "revision": A40,
                "use_safetensors": True,
                "torch_dtype": stack.torch.float16,
                "device_map": {"": "cuda:0"},
                "attn_implementation": "sdpa",
            },
        )
    ]


def test_a_task_argument_can_never_override_a_sealed_loader_field(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    view = sealed_loader_view(_sealed(tmp_path, "reward"))

    with pytest.raises(TrainerError, match="cannot override sealed loader fields: device_map, revision"):
        load_sealed_model(
            stack.AutoModelForSequenceClassification,
            stack.torch,
            view,
            stage=_Stages(),
            bitsandbytes_config_cls=FakeBitsAndBytesConfig,
            revision="main",
            device_map={"": 0},
        )
    assert stack.loads() == []


def _loader(stack: FakeStack, edit: Callable[[FakeLoadedModel], None]):
    class _Loader:
        @staticmethod
        def from_pretrained(location: str, **kwargs: Any) -> FakeLoadedModel:
            model = FakeLoadedModel(stack.torch, location, kwargs)
            edit(model)
            return model

    return _Loader


def test_post_load_observation_refuses_an_attention_or_placement_substitution(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    view = sealed_loader_view(_sealed(tmp_path, "preference"))

    def _eager(model: FakeLoadedModel) -> None:
        model.config._attn_implementation = "eager"

    def _other_gpu(model: FakeLoadedModel) -> None:
        model.hf_device_map = {"": "cuda:1"}

    with pytest.raises(TrainerError, match="attention policy deviation"):
        load_sealed_model(
            _loader(stack, _eager), stack.torch, view, stage=_Stages(),
            bitsandbytes_config_cls=FakeBitsAndBytesConfig,
        )
    stages = _Stages()
    with pytest.raises(ExecutionPlacementDeviation, match="outside cuda:0"):
        load_sealed_model(
            _loader(stack, _other_gpu), stack.torch, view, stage=stages,
            bitsandbytes_config_cls=FakeBitsAndBytesConfig,
        )
    assert stages.names[-1] == "placement_deviation"


def test_a_local_model_changed_while_loading_is_refused(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    model_dir = _local_model(tmp_path)
    view = sealed_loader_view(
        _execution(
            _sealed_plan(
                tmp_path,
                "preference",
                base_model=str(model_dir),
                model_content_sha256=stable_directory_sha256(model_dir),
            )
        )
    )

    def _swap_during_load(_model: FakeLoadedModel) -> None:
        (model_dir / "model.safetensors").write_bytes(b"swapped-during-load")

    with pytest.raises(TrainerError, match="changed while loading"):
        load_sealed_model(
            _loader(stack, _swap_during_load), stack.torch, view, stage=_Stages(),
            bitsandbytes_config_cls=FakeBitsAndBytesConfig,
        )
    # Unchanged bytes pass the same post-load re-hash.
    (model_dir / "model.safetensors").write_bytes(b"sealed-weights")
    verify_sealed_model_after_load(
        FakeLoadedModel(stack.torch, str(model_dir), _qlora_load_kwargs(stack)), view, stage=_Stages()
    )


# --- post-adapter precision (QLoRA) and full-parameter storage ------------------------------------------


def _adapter_model(stack: FakeStack, **bnb_updates: Any) -> FakeLoadedModel:
    kwargs = _qlora_load_kwargs(stack)
    kwargs["quantization_config"] = FakeBitsAndBytesConfig(
        **{**kwargs["quantization_config"].kwargs, **bnb_updates}
    )
    model = FakeLoadedModel(stack.torch, MODEL, kwargs)
    model.attach_lora(stack.torch.bfloat16)
    return model


def test_adapter_precision_lowers_the_master_dtype_and_observes_nf4_state(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    stages = _Stages()
    view = sealed_loader_view(_sealed(tmp_path, "preference"))
    model = _adapter_model(stack)
    tracker = pretraining_evidence.register_full_model_gradient_hooks(model, stack.torch)

    verify_sealed_adapter_precision(model, stack.torch, view, tracker, stage=stages)

    lora = dict(model.named_parameters())["model.layers.0.lora_A.weight"]
    assert lora.dtype is stack.torch.float32  # the sealed fp32 master dtype, lowered in place
    assert lora.hooks  # same identity-bound parameter, hook still registered
    assert stages.names == ["placement_verified", "precision_verified"]
    assert "nf4 base storage, bf16 dequantization, and fp32 trainable master weights" in (
        stages.calls[-1][1]
    )


@pytest.mark.parametrize(
    ("field", "substituted", "expected"),
    [
        ("bnb_4bit_quant_type", lambda _torch: "fp4", "quantized storage deviation"),
        ("bnb_4bit_compute_dtype", lambda torch: torch.float16, "dequantization dtype deviation"),
    ],
)
def test_adapter_precision_refuses_a_substituted_quantization(
    tmp_path, monkeypatch, field, substituted, expected
):
    stack = FakeStack(monkeypatch)
    view = sealed_loader_view(_sealed(tmp_path, "reward"))
    model = _adapter_model(stack, **{field: substituted(stack.torch)})
    tracker = pretraining_evidence.register_full_model_gradient_hooks(model, stack.torch)

    with pytest.raises(TrainerError, match=expected):
        verify_sealed_adapter_precision(model, stack.torch, view, tracker, stage=_Stages())


def test_adapter_precision_refuses_a_trainable_inventory_change(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    view = sealed_loader_view(_sealed(tmp_path, "preference"))
    model = _adapter_model(stack)
    tracker = pretraining_evidence.register_full_model_gradient_hooks(model, stack.torch)
    model.attach_lora(stack.torch.float32)  # a trainable tensor with no registered hook

    with pytest.raises(TrainingEvidenceError, match="inventory changed"):
        verify_sealed_adapter_precision(model, stack.torch, view, tracker, stage=_Stages())


def test_full_parameter_storage_is_observed_against_the_seal(tmp_path, monkeypatch):
    stack = FakeStack(monkeypatch)
    stages = _Stages()
    view = sealed_loader_view(_sealed(tmp_path, "full_finetune"))
    kwargs = {"torch_dtype": stack.torch.bfloat16, "device_map": {"": "cuda:0"}}

    verify_full_parameter_storage(FakeLoadedModel(stack.torch, MODEL, kwargs), stack.torch, view, stage=stages)
    assert stages.calls == [
        ("precision_verified", "observed every floating parameter stored as the sealed bf16")
    ]

    mixed = FakeLoadedModel(stack.torch, MODEL, kwargs)
    mixed.attach_lora(stack.torch.float32)
    with pytest.raises(TrainerError, match="observed torch.bfloat16, torch.float32, expected torch.bfloat16"):
        verify_full_parameter_storage(mixed, stack.torch, view, stage=_Stages())
    integer_only = FakeLoadedModel(stack.torch, MODEL, {**kwargs, "torch_dtype": stack.torch.int8})
    with pytest.raises(TrainerError, match="observed none"):
        verify_full_parameter_storage(integer_only, stack.torch, view, stage=_Stages())
    qlora_view = sealed_loader_view(_sealed(tmp_path, "reward"))
    with pytest.raises(TrainerError, match="omitted its weight-storage dtype"):
        verify_full_parameter_storage(mixed, stack.torch, qlora_view, stage=_Stages())


# --- the REAL worker bodies ------------------------------------------------------------------------------


class _StopTraining(Exception):
    """Raised by a stubbed training/evaluation primitive after it has observed the kernel context."""


class _FakeTracker:
    def __init__(self, *, expected_steps: int, gradients: Any) -> None:
        self.expected_steps = expected_steps
        self.gradients = gradients

    def on_train_begin(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def record_step(self, *_a: Any, **_k: Any) -> None:
        return None

    def finalize(self, **_k: Any) -> str:
        return "execution-evidence"


def _stub_training_plane(monkeypatch, stack: FakeStack, **primitives: Any) -> None:
    """Replace only what needs real tensors (state digests, optimizer, evidence trackers) and the named
    training/evaluation primitives, which record the attention state they ran under."""
    monkeypatch.setattr(trainer_module, "capture_trainable_state", lambda *_a, **_k: "state")
    monkeypatch.setattr(trainer_module, "capture_adapter_export_state", lambda *_a, **_k: "state")
    monkeypatch.setattr(trainer_module, "expected_saved_adapter_config_sha256", lambda *_a: "0" * 64)
    monkeypatch.setattr(optimizer_config, "build_torch_optimizer", lambda *_a: "optimizer")
    for module, name in (
        (preference_evidence, "PreferenceExecutionTracker"),
        (reward_evidence, "RewardExecutionTracker"),
        (rollout_evidence, "RolloutExecutionTracker"),
        (pretraining_evidence, "PretrainingExecutionTracker"),
    ):
        monkeypatch.setattr(module, name, _FakeTracker)
    for name, result in primitives.items():
        monkeypatch.setattr(trainer_module, name, _recording_primitive(stack, name, result))


def _recording_primitive(stack: FakeStack, name: str, result: Any):
    def _primitive(*_a: Any, **_k: Any) -> Any:
        stack.events.append(
            (name, tuple(stack.torch.active_kernels), stack.torch.sdp_state())
        )
        if result is None:
            raise _StopTraining(name)
        return result

    return _primitive


def _index(events: list[tuple[Any, ...]], predicate: Callable[[tuple[Any, ...]], bool]) -> int:
    return next(index for index, event in enumerate(events) if predicate(event))


def test_run_preference_loads_the_pinned_identities_and_trains_under_the_sealed_kernel(
    tmp_path, monkeypatch
):
    from corpus_studio.training.preference_worker import run_preference

    stack = FakeStack(monkeypatch)
    _stub_training_plane(monkeypatch, stack, run_dpo_training=None)
    execution = _sealed(tmp_path, "preference")
    stages = _Stages()

    with pytest.raises(_StopTraining):
        run_preference(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
            stage_callback=stages,
        )

    assert stack.loads() == [
        ("AutoTokenizer", *TOKENIZER_ARGS),
        ("AutoModelForCausalLM", MODEL, _qlora_load_kwargs(stack)),
    ]
    events = stack.events
    probe = _index(events, lambda e: e[0] == "probe_sdpa")
    model_load = _index(events, lambda e: e[:2] == ("load", "AutoModelForCausalLM"))
    tokenizer_load = _index(events, lambda e: e[:2] == ("load", "AutoTokenizer"))
    assert tokenizer_load < probe < model_load  # toggles + isolated probe before any weight allocation
    assert events[probe] == ("probe_sdpa", ("MATH",))
    assert ("run_dpo_training", (("MATH",),), (False, False, True)) in events
    assert ("sdpa_kernel_exit", ("MATH",)) in events  # the exclusive context closed on the failure
    lora = dict(stack.models[0].named_parameters())["model.layers.0.lora_A.weight"]
    assert lora.dtype is stack.torch.float32
    assert stages.names == [
        "tokenizer_load",
        "tokenizer_load",
        "attention_policy_applied",
        "attention_policy_applied",
        "model_load",
        "model_load",
        "placement_verified",
        "placement_verified",
        "precision_verified",
    ]


def test_run_preference_honours_a_sealed_fp32_compute_dtype(tmp_path, monkeypatch):
    from corpus_studio.training.preference_worker import run_preference

    stack = FakeStack(monkeypatch, stop_at_kbit=True)
    execution = _sealed(
        tmp_path, "preference", _precision(dequantization_dtype="fp32", forward_compute_dtype="fp32")
    )

    with pytest.raises(StopAtKbit):
        run_preference(execution, dataset=read_verified_dataset(execution.inputs.dataset))

    assert stack.loads()[-1] == (
        "AutoModelForCausalLM",
        MODEL,
        _qlora_load_kwargs(stack, compute="float32"),
    )
    assert ("probe_randn", "cuda", "float32", True) in stack.events


def _split_map(payload: dict[str, Any]) -> None:
    payload["device_map"] = [
        {"module": "", "device": "cuda:0"},
        {"module": "lm_head", "device": "cpu"},
    ]


def _cuda1(payload: dict[str, Any]) -> None:
    payload["device_map"] = [{"module": "", "device": "cuda:1"}]


def _duplicate_root(payload: dict[str, Any]) -> None:
    # Contract-valid on the reward and on-policy RL lanes; a dict built from it keeps only cuda:0.
    payload["device_map"] = [
        {"module": "", "device": "cuda:1"},
        {"module": "", "device": "cuda:0"},
    ]


def _unpinned_reward_base(payload: dict[str, Any]) -> None:
    payload["reward_source"]["reward_base_model"] = "review/other-reward-base"


def _worker(lane: str):
    if lane == "preference":
        from corpus_studio.training.preference_worker import PreferenceWorkerError, run_preference

        return run_preference, PreferenceWorkerError
    if lane == "reward":
        from corpus_studio.training.reward_worker import RewardWorkerError, run_reward

        return run_reward, RewardWorkerError
    if lane == "rollout":
        from corpus_studio.training.rollout_worker import RolloutWorkerError, run_rollout

        return run_rollout, RolloutWorkerError
    from corpus_studio.training.full_finetune_trainer import FullFinetuneError, run_full_finetune

    return run_full_finetune, FullFinetuneError


@pytest.mark.parametrize(
    ("lane", "mutate", "expected"),
    [
        ("preference", _precision(quantized_storage_format="int4"), "rather than run nf4"),
        ("reward", _split_map, "one root placement"),
        ("full_finetune", _cuda1, "sealed device 'cuda:1' cannot be honored"),
        ("rollout", _unpinned_reward_base, "refuse an unpinned reward model"),
        ("reward", _duplicate_root, "exactly one root placement"),
        ("rollout", _duplicate_root, "exactly one root placement"),
        ("preference", _precision(gradient_dtype="bf16"), "sealed gradient dtype 'bf16'"),
        ("reward", _precision(optimizer_state_dtype="int8"), "optimizer-state dtype 'int8'"),
        ("rollout", _precision(optimizer_state_dtype="bf16"), "optimizer-state dtype 'bf16'"),
    ],
)
def test_each_worker_refuses_an_unlowerable_seal_before_anything_loads(
    tmp_path, monkeypatch, lane, mutate, expected
):
    stack = FakeStack(monkeypatch)
    execution = _sealed(tmp_path, lane, mutate)
    run_worker, worker_error = _worker(lane)

    with pytest.raises(worker_error, match=expected):
        run_worker(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
        )
    assert stack.events == []
    assert not (tmp_path / "out").exists()


def test_run_reward_loads_the_pinned_score_model_and_measures_under_the_sealed_kernel(
    tmp_path, monkeypatch
):
    from corpus_studio.training.reward_worker import run_reward

    stack = FakeStack(monkeypatch)
    trained = {
        "losses": [0.6],
        "reward_margins": [0.1],
        "chosen_rewards": [0.2],
        "rejected_rewards": [0.1],
    }
    _stub_training_plane(
        monkeypatch, stack, run_reward_training=trained, evaluate_reward_accuracy=None
    )
    execution = _sealed(tmp_path, "reward")
    stages = _Stages()

    with pytest.raises(_StopTraining, match="evaluate_reward_accuracy"):
        run_reward(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
            stage_callback=stages,
        )

    assert stack.loads() == [
        ("AutoTokenizer", *TOKENIZER_ARGS),
        (
            "AutoModelForSequenceClassification",
            MODEL,
            {**_qlora_load_kwargs(stack), "num_labels": 1},
        ),
    ]
    under_kernel = (("MATH",),), (False, False, True)
    assert ("run_reward_training", *under_kernel) in stack.events
    assert ("evaluate_reward_accuracy", *under_kernel) in stack.events
    assert stages.names.count("precision_verified") == 1


class _TrainingArguments:
    n_gpu = 1

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _full_finetune_stack(monkeypatch, *, n_gpu: int, stop_at_train: bool = True) -> FakeStack:
    stack = FakeStack(monkeypatch)
    _stub_training_plane(monkeypatch, stack)

    class TrainingArguments(_TrainingArguments):
        pass

    TrainingArguments.n_gpu = n_gpu

    class Trainer:
        def __init__(self, *, model, args, **_kwargs: Any) -> None:
            self.model = model
            self.args = args
            stack.events.append(("trainer", args.kwargs))

        def train(self) -> Any:
            stack.events.append(
                ("trainer.train", tuple(stack.torch.active_kernels), stack.torch.sdp_state())
            )
            raise _StopTraining("trainer.train")

    monkeypatch.setattr(stack.transformers, "TrainingArguments", TrainingArguments, raising=False)
    monkeypatch.setattr(stack.transformers, "Trainer", Trainer, raising=False)
    return stack


def test_run_full_finetune_loads_the_tokenizer_first_and_the_sealed_storage_dtype(
    tmp_path, monkeypatch
):
    from corpus_studio.training.full_finetune_trainer import run_full_finetune

    stack = _full_finetune_stack(monkeypatch, n_gpu=1)
    execution = _sealed(
        tmp_path,
        "full_finetune",
        _precision(
            weight_storage_dtype="fp16", dequantization_dtype="fp16", forward_compute_dtype="fp16"
        ),
    )
    stages = _Stages()

    with pytest.raises(_StopTraining):
        run_full_finetune(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
            stage_callback=stages,
        )

    assert stack.loads() == [
        ("AutoTokenizer", *TOKENIZER_ARGS),
        (
            "AutoModelForCausalLM",
            MODEL,
            {
                "trust_remote_code": False,
                "revision": A40,
                "use_safetensors": True,
                "torch_dtype": stack.torch.float16,
                "device_map": {"": "cuda:0"},
                "attn_implementation": "sdpa",
            },
        ),
    ]
    trainer_kwargs = next(event[1] for event in stack.events if event[0] == "trainer")
    assert trainer_kwargs["use_cpu"] is False
    assert ("trainer.train", (("MATH",),), (False, False, True)) in stack.events
    assert stages.names.count("placement_verified") == 2  # after the load and after the HF Trainer
    assert "Trainer-placed model" in [msg for name, msg in stages.calls if name == "placement_verified"][-1]


def test_run_full_finetune_refuses_a_trainer_that_would_replicate_across_gpus(tmp_path, monkeypatch):
    from corpus_studio.training.full_finetune_trainer import run_full_finetune

    stack = _full_finetune_stack(monkeypatch, n_gpu=2)
    execution = _sealed(tmp_path, "full_finetune")

    with pytest.raises(ExecutionPlacementDeviation, match="sees 2 GPUs"):
        run_full_finetune(execution, dataset=read_verified_dataset(execution.inputs.dataset))
    assert not [event for event in stack.events if event[0] == "trainer.train"]


def test_run_full_finetune_takes_the_cpu_smoke_path_from_the_seal(tmp_path, monkeypatch):
    from corpus_studio.training.full_finetune_trainer import run_full_finetune

    stack = _full_finetune_stack(monkeypatch, n_gpu=0)
    execution = _execution(
        _sealed_plan(tmp_path, "full_finetune", report=_cpu_toy_report(), allow_cpu_toy=True)
    )

    with pytest.raises(_StopTraining):
        run_full_finetune(execution, dataset=read_verified_dataset(execution.inputs.dataset))

    model_kwargs = stack.loads()[-1][2]
    assert model_kwargs["device_map"] == {"": "cpu"}
    assert model_kwargs["attn_implementation"] == "eager"
    assert model_kwargs["torch_dtype"] is stack.torch.float32
    trainer_kwargs = next(event[1] for event in stack.events if event[0] == "trainer")
    assert trainer_kwargs["use_cpu"] is True
    assert trainer_kwargs["gradient_checkpointing"] is False
    # eager does not dispatch through SDPA: no exclusive kernel context, math-only fallback toggles
    assert ("trainer.train", (), (False, False, True)) in stack.events


def test_run_rollout_loads_the_policy_and_the_served_reward_base_at_the_pinned_identity(
    tmp_path, monkeypatch
):
    from corpus_studio.training.rollout_worker import run_rollout

    stack = FakeStack(monkeypatch, stop_at_kbit=True)
    monkeypatch.setattr(
        trainer_module, "_seqcls_backbone_and_score_head", lambda _model: ("backbone", "head")
    )
    adapter = tmp_path / "reward-adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"reward-adapter")
    adapter_sha = hashlib.sha256(b"reward-adapter").hexdigest()

    def _local_adapter(payload: dict[str, Any]) -> None:
        payload["reward_source"]["reward_adapter_location"] = str(adapter)
        payload["reward_source"]["reward_ref"]["hash"]["value"] = adapter_sha

    execution = _sealed(tmp_path, "rollout", _local_adapter)
    stages = _Stages()

    with pytest.raises(StopAtKbit):
        run_rollout(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
            stage_callback=stages,
        )

    assert stack.loads() == [
        ("AutoTokenizer", *TOKENIZER_ARGS),  # policy tokenizer: the tokenizer binding, not the model
        ("AutoTokenizer", *TOKENIZER_ARGS),  # served reward tokenizer: the same pinned binding
        (
            "AutoModelForSequenceClassification",
            MODEL,
            {**_qlora_load_kwargs(stack), "num_labels": 1},
        ),
        ("PeftModel", str(adapter), {}),
        ("AutoModelForCausalLM", MODEL, _qlora_load_kwargs(stack)),
    ]
    probe = _index(stack.events, lambda e: e[0] == "probe_sdpa")
    first_model = _index(stack.events, lambda e: e[:2] == ("load", "AutoModelForSequenceClassification"))
    assert probe < first_model
    placement = [msg for name, msg in stages.calls if name == "placement_verified"]
    assert "served reward base" in placement[0] and "policy" in placement[1]


def test_the_sealed_loader_module_is_import_light():
    code = (
        "import sys, corpus_studio.training.sealed_loader, corpus_studio.platform; "
        "print(json.dumps(sorted(m for m in ('torch', 'transformers', 'peft', 'bitsandbytes') "
        "if m in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import json; " + code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


# --- the sealed truncation policy is one rule for every lane --------------------------------------------


def _safe_default(payload: dict[str, Any]) -> None:
    """The contract's documented SAFE default: the sequence flag permits, the data policy refuses.

    ``SequenceSpec.truncation_allowed`` defaults to True and ``truncation_policy`` to ``"refuse"``,
    and the validators reject only the opposite pair, so this combination is contract-valid on every
    lane. A worker that read the sequence flag by itself would truncate under it."""
    payload["sequence"]["truncation_allowed"] = True
    payload["experience" if "experience" in payload else "data"]["truncation_policy"] = "refuse"


def _lossy(payload: dict[str, Any]) -> None:
    payload["sequence"]["truncation_allowed"] = True
    payload["experience" if "experience" in payload else "data"]["truncation_policy"] = "allow"


@pytest.mark.parametrize("lane", ["preference", "reward", "full_finetune", "rollout"])
def test_the_safe_default_seal_refuses_truncation_on_every_lane(tmp_path, lane):
    # The data policy is the enforced key; the sequence flag alone must never permit a cut.
    execution = _sealed(tmp_path, lane, _safe_default)
    assert execution.sequence.truncation_allowed is True
    assert sealed_truncation_permitted(execution) is False
    assert sealed_loader_view(execution).truncation_allowed is False


@pytest.mark.parametrize("lane", ["preference", "reward", "full_finetune", "rollout"])
def test_an_explicitly_lossy_seal_permits_truncation_on_every_lane(tmp_path, lane):
    execution = _sealed(tmp_path, lane, _lossy)
    assert sealed_truncation_permitted(execution) is True
    assert sealed_loader_view(execution).truncation_allowed is True


def test_full_parameter_helper_shares_the_one_rule(tmp_path):
    from corpus_studio.training.full_finetune_trainer import full_finetune_truncation_permitted

    for mutate in (_safe_default, _lossy):
        execution = _sealed(tmp_path, "full_finetune", mutate)
        assert full_finetune_truncation_permitted(execution) is sealed_truncation_permitted(
            execution
        )


@pytest.mark.parametrize(
    ("lane", "primitive"),
    [
        ("preference", "run_dpo_training"),
        ("reward", "run_reward_training"),
        ("rollout", "run_rollout_training"),
    ],
)
def test_the_qlora_workers_hand_the_primitive_the_sealed_policy(
    tmp_path, monkeypatch, lane, primitive
):
    # Each worker used to pass execution.sequence.truncation_allowed straight through, so under the
    # safe default above it told its primitive truncation was permitted.
    stack = FakeStack(monkeypatch)
    _stub_training_plane(monkeypatch, stack)
    captured: dict[str, Any] = {}

    def _capture(*_args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _StopTraining(primitive)

    monkeypatch.setattr(trainer_module, primitive, _capture)
    mutations: list[Callable[[dict[str, Any]], None]] = [_safe_default]
    if lane == "rollout":
        # The served reward adapter has to exist on disk before the policy is trained.
        monkeypatch.setattr(
            trainer_module, "_seqcls_backbone_and_score_head", lambda _model: ("backbone", "head")
        )
        adapter = tmp_path / "reward-adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"reward-adapter")
        adapter_sha = hashlib.sha256(b"reward-adapter").hexdigest()

        def _local_adapter(payload: dict[str, Any]) -> None:
            payload["reward_source"]["reward_adapter_location"] = str(adapter)
            payload["reward_source"]["reward_ref"]["hash"]["value"] = adapter_sha

        mutations.append(_local_adapter)
    execution = _sealed(tmp_path, lane, *mutations)
    run_worker, _error = _worker(lane)

    with pytest.raises(_StopTraining):
        run_worker(
            execution,
            dataset=read_verified_dataset(execution.inputs.dataset),
            output_dir=str(tmp_path / "out"),
        )

    assert captured["truncation_allowed"] is False
