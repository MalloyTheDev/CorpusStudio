"""First-party trainer — the PURE helpers (config load, formatting, arg mapping, run-plan resolution).

These carry no heavy imports, so they run in CI without torch/TRL. The actual `run_training` is
verified separately via the CPU toy path (installing the CPU subset of the [train] extra).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

import corpus_studio.training.trainer as trainer_module
from corpus_studio.platform.trace_records import (
    artifact_trace_source,
    build_reasoning_trace_record,
    imported_trace_producer,
)
from corpus_studio.training.traces import Trace
from corpus_studio.training.environment import TrainingRuntimeReport
from corpus_studio.training.quantization import find_linear4bit_modules
from corpus_studio.training.trainer import (
    TINY_TOY_MODEL,
    ExecutionPlacementDeviation,
    TrainerError,
    TrainRunConfig,
    analyze_truncation,
    apply_attention_execution_policy,
    build_lora_kwargs,
    build_model_load_kwargs,
    build_training_kwargs,
    enforce_trainable_precision,
    format_example_text,
    load_run_config_from_file,
    resolve_attention_implementation,
    resolve_run_plan,
    run_training,
    train_config_from_resolved,
    truncation_warning,
    verify_sealed_runtime,
    verify_loaded_model_execution,
    verify_local_inputs_after_load,
    verify_model_state_execution,
    verify_optimizer_state_precision,
    verify_completed_step_count,
    _list_checkpoints,
    _prepare_training_texts,
)


def _config(tmp_path: Path, **overrides) -> Path:
    data = {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "dataset_path": "train.jsonl",
        "format": "chat",
        "sequence_len": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0002,
        "seed": 42,
    }
    data.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- config load -------------------------------------------------------------


def test_load_config_maps_all_fields(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path))
    assert cfg.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.dataset_path == "train.jsonl"
    assert cfg.dataset_format == "chat"
    assert cfg.lora_r == 16 and cfg.lora_alpha == 32
    assert cfg.seed == 42
    assert cfg.cpu_toy is False


def test_cpu_toy_forces_tiny_model_short_seq_and_steps(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path), cpu_toy=True)
    assert cfg.base_model == TINY_TOY_MODEL  # not the 7B
    assert cfg.sequence_len <= 128
    assert cfg.max_steps == 3
    assert cfg.cpu_toy is True


def test_overrides_win(tmp_path):
    cfg = load_run_config_from_file(
        _config(tmp_path), base_model="my/model", dataset_path="split/train.jsonl", max_steps=10
    )
    assert cfg.base_model == "my/model"
    assert cfg.dataset_path == "split/train.jsonl"
    assert cfg.max_steps == 10


def test_missing_base_or_dataset_raises(tmp_path):
    with pytest.raises(TrainerError):
        load_run_config_from_file(_config(tmp_path, base_model="", dataset_path=""))


def test_run_training_blocks_pending_trace_record_before_runtime_probe(
    tmp_path: Path, monkeypatch
):
    row = {
        "prompt": "Q",
        "thinking": "A sufficiently detailed reasoning process for this training example.",
        "answer": "A",
    }
    record = build_reasoning_trace_record(
        trace=Trace(**row),
        source=artifact_trace_source(
            artifact_ref="source.jsonl",
            artifact_sha256="a" * 64,
            row=row,
            row_index=1,
        ),
        producer=imported_trace_producer(),
        created_at="2026-07-13T12:00:00+00:00",
        trace_id="trace-pending",
    )
    dataset = tmp_path / "traces.jsonl"
    dataset.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    def must_not_probe():
        raise AssertionError("runtime probe must happen after the trace approval gate")

    monkeypatch.setattr(trainer_module, "probe_training_runtime", must_not_probe)
    with pytest.raises(TrainerError, match="review status is pending"):
        run_training(
            TrainRunConfig(
                base_model="unused",
                dataset_path=str(dataset),
                dataset_format="trace",
            )
        )


def test_load_config_accepts_yaml_so_a_named_yaml_does_not_die(tmp_path):
    # train-run parses JSON, but a config named *.yaml (or a hand-written YAML) must still load —
    # the WBG run pointed train-run at wbg7b_corpus.yaml. JSON is a YAML subset, but real YAML too.
    # PyYAML is only needed for the YAML fallback and ships with the [train] extra (transformers/
    # datasets) — where train-run actually runs — so skip when it's absent (the dependency-light gate).
    pytest.importorskip("yaml")
    config = tmp_path / "wbg7b_corpus.yaml"
    config.write_text(
        "base_model: Qwen/Qwen2.5-7B\ndataset_path: train.jsonl\nformat: chat\nsequence_len: 4096\nlora_r: 16\n",
        encoding="utf-8",
    )
    cfg = load_run_config_from_file(config)
    assert cfg.base_model == "Qwen/Qwen2.5-7B"
    assert cfg.dataset_path == "train.jsonl"
    assert cfg.dataset_format == "chat"
    assert cfg.sequence_len == 4096 and cfg.lora_r == 16


def test_load_config_non_mapping_raises_trainer_error(tmp_path):
    # A file that is neither a JSON object nor a YAML mapping is a clean TrainerError (→ CLI exit 2),
    # not a cryptic parser traceback.
    config = tmp_path / "bad.yaml"
    config.write_text("just a plain string, not a config mapping\n", encoding="utf-8")
    with pytest.raises(TrainerError):
        load_run_config_from_file(config)


def test_load_config_reads_attn_implementation(tmp_path):
    # From the config file, and an explicit override wins.
    cfg = load_run_config_from_file(_config(tmp_path, attn_implementation="eager"))
    assert cfg.attn_implementation == "eager"
    override = load_run_config_from_file(_config(tmp_path), attn_implementation="sdpa")
    assert override.attn_implementation == "sdpa"
    assert load_run_config_from_file(_config(tmp_path)).attn_implementation is None


# ---- memory / spill-avoidance levers -----------------------------------------


def test_build_kwargs_sets_optim_and_liger():
    cfg = TrainRunConfig(base_model="m", dataset_path="d", optim="paged_adamw_8bit", use_liger=True)
    kwargs = build_training_kwargs(cfg)
    assert kwargs["optim"] == "paged_adamw_8bit"
    assert kwargs["use_liger_kernel"] is True


def test_build_kwargs_default_optim_and_no_liger():
    kwargs = build_training_kwargs(TrainRunConfig(base_model="m", dataset_path="d"))
    assert kwargs["optim"] == "adamw_torch"
    assert "use_liger_kernel" not in kwargs  # off by default — never requested unless opted in


def test_cpu_toy_forces_plain_optimizer_and_no_liger():
    # The paged optimizer (bitsandbytes) and Liger (Triton) are CUDA-only; the CPU toy must never
    # request them or it would crash on a GPU-less machine, defeating the smoke test.
    cfg = TrainRunConfig(
        base_model="m", dataset_path="d", cpu_toy=True, optim="paged_adamw_8bit", use_liger=True
    )
    kwargs = build_training_kwargs(cfg)
    assert kwargs["optim"] == "adamw_torch"
    assert "use_liger_kernel" not in kwargs
    assert kwargs["use_cpu"] is True


# ---- truncation guardrail ----------------------------------------------------


def test_analyze_truncation_flags_cut_examples():
    # 3 of 5 exceed seq_len 1000 → 60% truncated; zero-truncation needs seq_len >= max (1500).
    report = analyze_truncation([500, 800, 1200, 1400, 1500], 1000)
    assert report.n_examples == 5 and report.n_truncated == 3
    assert report.pct_truncated == 60.0
    assert report.max_tokens == 1500 and report.seq_len_for_zero_truncation == 1500
    assert report.truncates is True
    assert "TRUNCATION" in (truncation_warning(report) or "")


def test_analyze_truncation_no_cut_when_seq_len_covers_all():
    report = analyze_truncation([500, 800, 1200], 2048)
    assert report.n_truncated == 0 and report.truncates is False
    assert truncation_warning(report) is None


def test_analyze_truncation_empty_dataset_is_safe():
    report = analyze_truncation([], 4096)
    assert report.n_examples == 0 and report.n_truncated == 0 and report.truncates is False


def test_analyze_truncation_the_wbg_bug():
    # The real bug this guardrail exists for: every example (min 1802) exceeds seq_len 1536 →
    # 100% truncated, and only seq_len >= 3445 keeps them whole.
    report = analyze_truncation([1802, 2100, 2240, 3445], 1536)
    assert report.pct_truncated == 100.0
    assert report.seq_len_for_zero_truncation == 3445
    assert "1536" in (truncation_warning(report) or "") and "3445" in (truncation_warning(report) or "")


def test_full_dataset_preflight_emits_bounded_same_thread_progress():
    class Tokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, text):
            self.calls += 1
            return {"input_ids": text.split()}

    tokenizer = Tokenizer()
    rows = [
        {"instruction": f"question {index}", "output": f"answer {index}"}
        for index in range(100)
    ]
    events: list[tuple[str, str]] = []

    texts, report = _prepare_training_texts(
        rows,
        _cfg(dataset_format="instruction", sequence_len=128),
        tokenizer,
        stage_callback=lambda stage, message: events.append((stage, message)),
    )

    assert len(texts) == 100
    assert tokenizer.calls == 100
    assert report.n_examples == 100 and report.n_truncated == 0
    formatting = [message for stage, message in events if stage == "dataset_formatting"]
    tokenization = [message for stage, message in events if stage == "truncation_analysis"]
    assert 2 <= len(formatting) <= 22
    assert 2 <= len(tokenization) <= 22
    assert formatting[0].startswith("formatting 100")
    assert formatting[-1].startswith("formatted all 100")
    assert tokenization[0].startswith("tokenizing all 100")
    assert tokenization[-1].startswith("verified 100")


def test_full_dataset_preflight_does_not_emit_fake_completion_after_tokenizer_failure():
    class BrokenTokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, _text):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("tokenizer wedged")
            return {"input_ids": [1]}

    events: list[tuple[str, str]] = []
    with pytest.raises(TrainerError, match="full-dataset truncation analysis failed"):
        _prepare_training_texts(
            [{"instruction": str(index), "output": "answer"} for index in range(5)],
            _cfg(dataset_format="instruction"),
            BrokenTokenizer(),
            stage_callback=lambda stage, message: events.append((stage, message)),
        )

    tokenization = [message for stage, message in events if stage == "truncation_analysis"]
    assert tokenization == [
        "tokenizing all 5 rendered rows for truncation analysis",
        "tokenized 1/5 rendered rows",
        "tokenized 2/5 rendered rows",
    ]
    assert not any(message.startswith("verified") for message in tokenization)


def test_load_config_reads_optim_and_liger(tmp_path):
    cfg = load_run_config_from_file(_config(tmp_path, optim="paged_adamw_8bit", use_liger=True))
    assert cfg.optim == "paged_adamw_8bit"
    assert cfg.use_liger is True
    # An explicit override wins over the config file.
    override = load_run_config_from_file(_config(tmp_path), optim="adamw_8bit", use_liger=True)
    assert override.optim == "adamw_8bit" and override.use_liger is True
    # Defaults when absent — the levers are opt-in.
    base = load_run_config_from_file(_config(tmp_path))
    assert base.optim == "adamw_torch" and base.use_liger is False


# ---- checkpoint policy -------------------------------------------------------


def test_build_kwargs_disables_intermediate_checkpoints_by_default():
    kwargs = build_training_kwargs(TrainRunConfig(base_model="m", dataset_path="d"))
    assert kwargs["save_strategy"] == "no"
    assert "save_steps" not in kwargs
    assert "save_total_limit" not in kwargs


def test_legacy_step_checkpoint_config_parses_but_cannot_execute():
    cfg = TrainRunConfig(
        base_model="m",
        dataset_path="d",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
    )
    with pytest.raises(TrainerError, match="resume compatibility"):
        build_training_kwargs(cfg)
    # The execution guard runs before dataset access or any heavy training-stack import.
    with pytest.raises(TrainerError, match="resume compatibility"):
        run_training(cfg)


def test_checkpoint_execution_guard_rejects_unvalidated_model_copy():
    config = TrainRunConfig(base_model="m", dataset_path="d").model_copy(
        update={"save_steps": 1}
    )
    with pytest.raises(TrainerError, match="resume compatibility"):
        build_training_kwargs(config)
    with pytest.raises(TrainerError, match="resume compatibility"):
        run_training(config)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"save_strategy": "no", "save_steps": 1}, "disabled checkpointing"),
        ({"save_strategy": "no", "save_total_limit": 1}, "disabled checkpointing"),
        ({"save_strategy": "steps"}, "requires save_steps"),
    ],
)
def test_checkpoint_policy_rejects_inconsistent_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainRunConfig(base_model="m", dataset_path="d", **overrides)


def test_load_config_defaults_checkpoint_free_and_parses_legacy_steps(tmp_path):
    base = load_run_config_from_file(_config(tmp_path))
    assert base.save_strategy == "no"
    assert base.save_steps is None and base.save_total_limit is None

    legacy = load_run_config_from_file(
        _config(
            tmp_path,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,
        )
    )
    assert legacy.save_strategy == "steps"
    assert legacy.save_steps == 100 and legacy.save_total_limit == 5


def test_resolve_attention_native_windows_blackwell_disables_flash_sdpa():
    # NATIVE WINDOWS + Blackwell (sm_120 → capability major 12): the fused FLASH SDPA kernel deadlocks
    # on the first backward under the Windows WDDM driver (verified on a real 5070; mem-efficient + math
    # are fine), so keep default SDPA but signal the caller to disable just the flash backend.
    assert resolve_attention_implementation(None, 12, native_windows=True) == (None, True)
    assert resolve_attention_implementation(None, 13, native_windows=True) == (None, True)


def test_resolve_attention_wsl_or_linux_blackwell_keeps_flash_enabled():
    # The deadlock is a Windows WDDM property, NOT an sm_120 kernel bug: on WSL / bare Linux the SAME
    # flash kernel runs fine (verified on a real 5070 under WSL2), so flash must stay ENABLED there —
    # the whole reason to run training under WSL. native_windows=False (WSL Python reports sys.platform
    # 'linux') → no SDP toggling on Blackwell.
    assert resolve_attention_implementation(None, 12, native_windows=False) == (None, False)
    assert resolve_attention_implementation(None, 13, native_windows=False) == (None, False)
    assert resolve_attention_implementation(None, 12) == (None, False)  # default (unknown host) = safe


def test_resolve_attention_older_arch_is_unchanged():
    # Pre-Blackwell arch: no toggling regardless of OS (the deadlock is sm_120-specific).
    assert resolve_attention_implementation(None, 9, native_windows=True) == (None, False)   # Ada/Hopper
    assert resolve_attention_implementation(None, 8, native_windows=True) == (None, False)
    assert resolve_attention_implementation(None, None, native_windows=True) == (None, False)  # no GPU


def test_resolve_attention_explicit_choice_always_wins():
    # An explicit attn_implementation is honored verbatim and never toggles the SDP backends, even on
    # native-Windows Blackwell.
    assert resolve_attention_implementation("eager", 12, native_windows=True) == ("eager", False)
    assert resolve_attention_implementation("flash_attention_2", 8, native_windows=True) == (
        "flash_attention_2",
        False,
    )


# ---- formatting --------------------------------------------------------------


def test_format_instruction():
    text = format_example_text({"instruction": "Explain X.", "output": "It is Y."}, "instruction")
    assert "Explain X." in text and "It is Y." in text and "### Response:" in text


def test_format_chat_without_tokenizer_joins_roles():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    text = format_example_text(row, "chat")
    assert "user: hi" in text and "assistant: hello" in text


def test_format_chat_template_failure_is_blocking():
    class BrokenTokenizer:
        def apply_chat_template(self, messages, *, tokenize):
            raise RuntimeError("template is invalid")

    row = {"messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(TrainerError, match="chat template failed"):
        format_example_text(row, "chat", BrokenTokenizer())


def test_format_empty_row_is_dropped():
    assert format_example_text({"instruction": "", "output": ""}, "instruction") == ""
    assert format_example_text({"messages": []}, "chat") == ""


# ---- arg mapping -------------------------------------------------------------


def test_lora_kwargs_use_all_linear():
    kw = build_lora_kwargs(_cfg(lora_r=16, lora_alpha=32))
    assert kw["r"] == 16 and kw["lora_alpha"] == 32
    assert kw["target_modules"] == "all-linear"
    assert kw["task_type"] == "CAUSAL_LM"


def test_lora_kwargs_are_fully_config_driven():
    kw = build_lora_kwargs(
        _cfg(
            lora_dropout=0.125,
            lora_bias="lora_only",
            lora_target_modules=["q_proj", "v_proj"],
        )
    )
    assert kw["lora_dropout"] == 0.125
    assert kw["bias"] == "lora_only"
    assert kw["target_modules"] == ["q_proj", "v_proj"]


class _FakeSdpBackend:
    def __init__(self, *, ignore_math: bool = False):
        self.flash = True
        self.mem_efficient = True
        self.math = True
        self.ignore_math = ignore_math

    def enable_flash_sdp(self, value):
        self.flash = value

    def enable_mem_efficient_sdp(self, value):
        self.mem_efficient = value

    def enable_math_sdp(self, value):
        if not self.ignore_math:
            self.math = value

    def flash_sdp_enabled(self):
        return self.flash

    def mem_efficient_sdp_enabled(self):
        return self.mem_efficient

    def math_sdp_enabled(self):
        return self.math


class _FakeTorch:
    float32 = object()
    float16 = object()
    bfloat16 = object()
    int8 = object()
    uint8 = object()

    def __init__(self, cuda_backend=None):
        self.backends = type("Backends", (), {"cuda": cuda_backend or _FakeSdpBackend()})()


def _sealed_config(**overrides):
    base = {
        "base_model": "model",
        "dataset_path": "dataset.jsonl",
        "execution_configuration_hash": "a" * 64,
        "model_revision": "b" * 40,
        "attn_implementation": "sdpa",
        "attention_kernel": "torch_sdpa_math",
        "flash_sdp_enabled": False,
        "mem_efficient_sdp_enabled": False,
        "math_sdp_enabled": True,
        "device_map": {"": "cuda:0"},
    }
    base.update(overrides)
    return TrainRunConfig(**base)


def test_attention_policy_applies_and_observes_all_three_sdp_toggles():
    torch = _FakeTorch()
    effective = apply_attention_execution_policy(torch, _sealed_config())
    assert effective == "torch_sdpa_math"
    assert torch.backends.cuda.flash_sdp_enabled() is False
    assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
    assert torch.backends.cuda.math_sdp_enabled() is True


def test_attention_policy_refuses_an_observed_toggle_deviation():
    torch = _FakeTorch(_FakeSdpBackend(ignore_math=True))
    cfg = _sealed_config(math_sdp_enabled=False)
    with pytest.raises(TrainerError, match="attention policy deviation"):
        apply_attention_execution_policy(torch, cfg)


def test_model_load_kwargs_pin_quantization_dtype_revision_and_device_map():
    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    torch = _FakeTorch()
    cfg = _sealed_config(dequantization_dtype="fp16", quantization_mode="nf4")
    kwargs = build_model_load_kwargs(
        cfg,
        torch,
        quantize=True,
        bitsandbytes_config_cls=FakeBitsAndBytesConfig,
    )
    assert kwargs["device_map"] == {"": "cuda:0"}
    assert kwargs["revision"] == "b" * 40
    assert kwargs["use_safetensors"] is True
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["quantization_config"].kwargs["bnb_4bit_compute_dtype"] is torch.float16


def test_model_load_kwargs_refuse_implicit_auto_placement():
    cfg = _sealed_config(device_map={"": "auto"})
    with pytest.raises(TrainerError, match="explicit non-auto device map"):
        build_model_load_kwargs(cfg, _FakeTorch(), quantize=False)


class _PlacementParameter:
    def __init__(self, device="cuda:0", *, name="weight"):
        self.device = device
        self._name = name


class _PlacementBuffer:
    def __init__(self, device="cuda:0", *, floating=True):
        self.device = device
        self._floating = floating

    def is_floating_point(self):
        return self._floating


def _placement_model(
    *,
    hf_device_map,
    parameters=None,
    buffers=None,
    attn="sdpa",
    attributes=None,
    child_modules=None,
):
    param_items = (
        list(parameters)
        if parameters is not None
        else [("layer.weight", _PlacementParameter())]
    )
    buffer_items = list(buffers) if buffers is not None else []
    children = list(child_modules) if child_modules is not None else []

    model = type(
        "Model",
        (),
        {
            "config": type("Config", (), {"_attn_implementation": attn})(),
            "hf_device_map": hf_device_map,
            "parameters": lambda self: iter(parameter for _, parameter in param_items),
            "named_parameters": lambda self: iter(param_items),
            "named_buffers": lambda self: iter(buffer_items),
            "named_modules": lambda self: iter([("", self), *children]),
        },
    )()
    for name, value in (attributes or {}).items():
        setattr(model, name, value)
    return model


def _accelerate_hook(name="AlignDevicesHook", **attributes):
    return type(name, (), {"__module__": "accelerate.hooks", **attributes})()


def test_loaded_model_execution_accepts_root_cuda0_map():
    verify_loaded_model_execution(
        _placement_model(hf_device_map={"": "cuda:0"}),
        _sealed_config(),
    )


def test_loaded_model_execution_accepts_integer_device_zero_in_map():
    verify_loaded_model_execution(
        _placement_model(hf_device_map={"": 0}),
        _sealed_config(),
    )


def test_loaded_model_execution_accepts_expanded_all_cuda0_map():
    verify_loaded_model_execution(
        _placement_model(
            hf_device_map={
                "model.embed_tokens": "cuda:0",
                "model.layers.0": "cuda:0",
                "lm_head": "cuda:0",
            }
        ),
        _sealed_config(),
    )


def test_loaded_model_execution_accepts_missing_hf_device_map_when_tensors_are_cuda0():
    """bitsandbytes NF4 loads often leave hf_device_map as None despite full GPU residency."""

    verify_loaded_model_execution(
        _placement_model(hf_device_map=None),
        _sealed_config(),
    )


def test_loaded_model_execution_accepts_exact_torch_device_values(monkeypatch):
    class TorchDevice:
        def __init__(self, type_, index=None):
            self.type = type_
            self.index = index

        def __str__(self):
            if self.index is None:
                return self.type
            return f"{self.type}:{self.index}"

    TorchDevice.__name__ = "device"
    TorchDevice.__module__ = "torch"
    torch_module = ModuleType("torch")
    torch_module.device = TorchDevice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    verify_loaded_model_execution(
        _placement_model(
            hf_device_map={"": TorchDevice("cuda", 0)},
            parameters=[("w", _PlacementParameter(TorchDevice("cuda", None)))],
            buffers=[("b", _PlacementBuffer(TorchDevice("cuda", 0)))],
        ),
        _sealed_config(),
    )


def test_loaded_model_execution_rejects_arbitrary_device_protocol_without_rendering_it():
    class UntrustedDevice:
        type = "cuda"
        index = 0

        def __str__(self):
            raise AssertionError("untrusted device representation must not be rendered")

    with pytest.raises(ExecutionPlacementDeviation, match="unknown device object type"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map={"": UntrustedDevice()}),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_boolean_torch_device_index(monkeypatch):
    TorchDevice = type(
        "device",
        (),
        {"__module__": "torch", "type": "cuda", "index": True},
    )
    torch_module = ModuleType("torch")
    torch_module.device = TorchDevice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    with pytest.raises(ExecutionPlacementDeviation, match="malformed or unsupported"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map={"": TorchDevice()}),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_cpu_map_entry():
    with pytest.raises(ExecutionPlacementDeviation, match="hf_device_map entries outside"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map={"": "cpu"}),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_disk_map_entry():
    with pytest.raises(ExecutionPlacementDeviation, match="hf_device_map entries outside"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map={"model.layers.0": "cuda:0", "model.layers.1": "disk"}
            ),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_other_gpu_map_entry():
    with pytest.raises(ExecutionPlacementDeviation, match="hf_device_map entries outside"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map={"model.layers.0": "cuda:1"}),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_parameter_on_cpu_despite_all_gpu_map():
    with pytest.raises(ExecutionPlacementDeviation, match="parameters outside"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map={
                    "model.embed_tokens": "cuda:0",
                    "lm_head": "cuda:0",
                },
                parameters=[
                    ("model.embed_tokens.weight", _PlacementParameter("cuda:0")),
                    ("lm_head.weight", _PlacementParameter("cpu")),
                ],
            ),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_meta_parameter():
    with pytest.raises(ExecutionPlacementDeviation, match="parameters outside"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                parameters=[("ghost.weight", _PlacementParameter("meta"))],
            ),
            _sealed_config(),
        )


@pytest.mark.parametrize("floating", [True, False])
def test_loaded_model_execution_rejects_every_buffer_off_device(floating):
    with pytest.raises(ExecutionPlacementDeviation, match="buffers outside"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map={"": "cuda:0"},
                buffers=[("quant.state", _PlacementBuffer("cpu", floating=floating))],
            ),
            _sealed_config(),
        )


@pytest.mark.parametrize(
    "hook_attributes",
    [
        {"offload": True, "weights_map": None, "execution_device": "cuda:0"},
        {"offload": False, "weights_map": {"layer.weight": object()}, "execution_device": "cuda:0"},
        {"offload": False, "weights_map": None, "execution_device": "cpu"},
    ],
)
def test_loaded_model_execution_rejects_accelerate_offload_hook_state(hook_attributes):
    hook = _accelerate_hook(**hook_attributes)
    with pytest.raises(ExecutionPlacementDeviation, match="hidden offload or hook state"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                attributes={"_hf_hook": hook},
            ),
            _sealed_config(),
        )


def test_loaded_model_execution_accepts_non_offloading_hook_on_sealed_device():
    hook = _accelerate_hook(
        offload=False,
        weights_map=None,
        execution_device="cuda:0",
    )
    verify_loaded_model_execution(
        _placement_model(hf_device_map=None, attributes={"_hf_hook": hook}),
        _sealed_config(),
    )


def test_loaded_model_execution_rejects_hook_name_masquerading_as_accelerate():
    hook = type(
        "AlignDevicesHook",
        (),
        {"offload": False, "weights_map": None, "execution_device": "cuda:0"},
    )()
    with pytest.raises(ExecutionPlacementDeviation, match="unsupported _hf_hook runtime"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map=None, attributes={"_hf_hook": hook}),
            _sealed_config(),
        )


def test_loaded_model_execution_rejects_module_hook_and_disk_offload_structures():
    hook = _accelerate_hook(
        offload=True,
        weights_map=None,
        execution_device="cuda:0",
    )
    child = type("Layer", (), {"_hf_hook": hook})()
    with pytest.raises(ExecutionPlacementDeviation, match="model.layers.0"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                child_modules=[("model.layers.0", child)],
            ),
            _sealed_config(),
        )
    with pytest.raises(ExecutionPlacementDeviation, match="weights_map"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                attributes={"weights_map": {}},
            ),
            _sealed_config(),
        )
    with pytest.raises(ExecutionPlacementDeviation, match="disk_offload"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                attributes={"disk_offload": True},
            ),
            _sealed_config(),
        )
    with pytest.raises(ExecutionPlacementDeviation, match="offload_index"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                attributes={"offload_index": {"layer.weight": "weights.bin"}},
            ),
            _sealed_config(),
        )


def test_loaded_model_execution_checks_alias_inventory_without_deduplication():
    cuda_parameter = _PlacementParameter("cuda:0")
    cpu_alias = _PlacementParameter("cpu")

    class Model:
        config = type("Config", (), {"_attn_implementation": "sdpa"})()
        hf_device_map = None

        def named_parameters(self, *, remove_duplicate=True):
            items = [("tied.weight", cuda_parameter)]
            if not remove_duplicate:
                items.append(("alias.weight", cpu_alias))
            return iter(items)

        def named_buffers(self, *, remove_duplicate=True):
            return iter(())

        def named_modules(self, *, remove_duplicate=True):
            return iter([("", self)])

    with pytest.raises(ExecutionPlacementDeviation, match="alias.weight=cpu"):
        verify_loaded_model_execution(Model(), _sealed_config())


def test_non_singleton_device_map_still_requires_exact_structure():
    expected = {"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"}
    verify_loaded_model_execution(
        _placement_model(
            hf_device_map=expected,
            parameters=[
                ("model.layers.0.weight", _PlacementParameter("cuda:0")),
                ("model.layers.1.weight", _PlacementParameter("cuda:1")),
            ],
            buffers=[("model.layers.1.counter", _PlacementBuffer("cuda:1", floating=False))],
        ),
        _sealed_config(device_map=expected),
    )
    with pytest.raises(ExecutionPlacementDeviation, match="requested device map"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map=None),
            _sealed_config(device_map=expected),
        )
    with pytest.raises(ExecutionPlacementDeviation, match="parameters disagree"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=expected,
                parameters=[
                    ("model.layers.0.weight", _PlacementParameter("cuda:0")),
                    ("model.layers.1.weight", _PlacementParameter("cpu")),
                ],
            ),
            _sealed_config(device_map=expected),
        )


def test_non_singleton_device_map_checks_each_shared_hook_attachment():
    expected = {"model.layers.0": "cuda:0", "model.layers.1": "cuda:1"}
    shared_hook = _accelerate_hook(
        offload=False,
        weights_map=None,
        execution_device="cuda:0",
    )
    children = [
        ("model.layers.0", type("Layer0", (), {"_hf_hook": shared_hook})()),
        ("model.layers.1", type("Layer1", (), {"_hf_hook": shared_hook})()),
    ]
    with pytest.raises(ExecutionPlacementDeviation, match="model.layers.1"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=expected,
                parameters=[
                    ("model.layers.0.weight", _PlacementParameter("cuda:0")),
                    ("model.layers.1.weight", _PlacementParameter("cuda:1")),
                ],
                child_modules=children,
            ),
            _sealed_config(device_map=expected),
        )


def test_cpu_toy_semantic_placement_checks_parameters_and_buffers():
    cfg = _sealed_config(cpu_toy=True, device_map={"": "cpu"})
    verify_loaded_model_execution(
        _placement_model(
            hf_device_map=None,
            parameters=[("weight", _PlacementParameter("cpu"))],
            buffers=[("counter", _PlacementBuffer("cpu", floating=False))],
        ),
        cfg,
    )
    with pytest.raises(ExecutionPlacementDeviation, match="buffers outside"):
        verify_loaded_model_execution(
            _placement_model(
                hf_device_map=None,
                parameters=[("weight", _PlacementParameter("cpu"))],
                buffers=[("counter", _PlacementBuffer("meta", floating=False))],
            ),
            cfg,
        )


def test_loaded_model_execution_rejects_missing_map_without_parameters():
    with pytest.raises(ExecutionPlacementDeviation, match="no parameters"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map=None, parameters=[]),
            _sealed_config(),
        )


def test_loaded_model_execution_emits_a_typed_placement_deviation():
    with pytest.raises(ExecutionPlacementDeviation, match="PLACEMENT_DEVIATION"):
        verify_loaded_model_execution(
            _placement_model(hf_device_map={"": "cpu"}),
            _sealed_config(),
        )


def test_post_adapter_state_verifies_placement_and_unquantized_storage_dtype():
    torch = _FakeTorch()

    class Parameter:
        def __init__(self, *, trainable, device="cuda:0", dtype=None):
            self.requires_grad = trainable
            self.device = device
            self.dtype = dtype or torch.float32

        def is_floating_point(self):
            return True

    parameters = [Parameter(trainable=False), Parameter(trainable=True)]
    model = type(
        "Model",
        (),
        {
            "named_parameters": lambda self: iter(
                [("base.weight", parameters[0]), ("adapter.weight", parameters[1])]
            ),
            "named_buffers": lambda self: iter(()),
            "parameters": lambda self: iter(parameters),
            "modules": lambda self: iter(()),
        },
    )()
    cfg = _sealed_config(
        quantization_mode="none",
        weight_storage_dtype="fp32",
        master_weight_dtype="fp32",
    )
    verify_model_state_execution(model, torch, cfg, quantize=False)

    parameters[1].device = "cpu"
    with pytest.raises(ExecutionPlacementDeviation, match="post-adapter"):
        verify_model_state_execution(model, torch, cfg, quantize=False)


def test_post_adapter_state_observes_nf4_and_dequantization_dtype():
    torch = _FakeTorch()

    class Parameter:
        requires_grad = True
        device = "cuda:0"
        dtype = torch.float32

        def is_floating_point(self):
            return True

    weight = type(
        "Weight",
        (),
        {"quant_state": type("QuantState", (), {"quant_type": "nf4"})()},
    )()
    BnbLinear4bit = type("Linear4bit", (), {})
    linear = BnbLinear4bit()
    linear.weight = weight
    linear.compute_dtype = torch.bfloat16
    peft_wrapper = type(
        "Linear4bit",
        (),
        {"weight": None, "compute_dtype": None},
    )()
    parameter = Parameter()
    model = type(
        "Model",
        (),
        {
            "named_parameters": lambda self: iter([("adapter.weight", parameter)]),
            "named_buffers": lambda self: iter(()),
            "parameters": lambda self: iter([parameter]),
            "modules": lambda self: iter([peft_wrapper, linear]),
        },
    )()
    cfg = _sealed_config(
        quantization_mode="nf4",
        dequantization_dtype="bf16",
        master_weight_dtype="fp32",
    )
    assert find_linear4bit_modules(model, BnbLinear4bit) == [linear]
    verify_model_state_execution(
        model,
        torch,
        cfg,
        quantize=True,
        linear4bit_type=BnbLinear4bit,
    )
    weight.quant_state.quant_type = "fp4"
    with pytest.raises(TrainerError, match="quantized storage deviation"):
        verify_model_state_execution(
            model,
            torch,
            cfg,
            quantize=True,
            linear4bit_type=BnbLinear4bit,
        )


def test_optimizer_state_precision_accepts_sealed_primary_and_auxiliary_dtypes():
    torch = _FakeTorch()

    def tensor(dtype):
        return type("Tensor", (), {"dtype": dtype})()

    optimizer = type(
        "Optimizer",
        (),
        {"state": {"parameter": {"state1": tensor(torch.uint8), "scale": tensor(torch.float32)}}},
    )()
    cfg = _sealed_config(optimizer_state_dtype="int8", optimizer_auxiliary_dtype="fp32")
    verify_optimizer_state_precision(optimizer, torch, cfg)


def test_optimizer_state_precision_refuses_runtime_drift():
    torch = _FakeTorch()
    bad = type("Tensor", (), {"dtype": torch.float16})()
    optimizer = type("Optimizer", (), {"state": {"parameter": {"exp_avg": bad}}})()
    with pytest.raises(TrainerError, match="optimizer-state dtype deviation"):
        verify_optimizer_state_precision(optimizer, torch, _sealed_config())


def test_optimizer_state_precision_recurses_into_nested_materialized_tensors():
    torch = _FakeTorch()
    bad = type("Tensor", (), {"dtype": torch.float16})()
    optimizer = type(
        "Optimizer",
        (),
        {"state": {"parameter": {"nested": [{"exp_avg": bad}]}}},
    )()
    with pytest.raises(TrainerError, match=r"optimizer_state\.nested\[0\]\.exp_avg"):
        verify_optimizer_state_precision(optimizer, torch, _sealed_config())


def test_sealed_max_steps_must_match_the_completed_global_step():
    config = _sealed_config(max_steps=3)
    verify_completed_step_count(config, 3)
    with pytest.raises(TrainerError, match="expected 3, observed 4"):
        verify_completed_step_count(config, 4)
    verify_completed_step_count(config.model_copy(update={"execution_configuration_hash": None}), 4)


def test_training_kwargs_capped_steps_vs_epochs(tmp_path):
    with_steps = build_training_kwargs(load_run_config_from_file(_config(tmp_path), max_steps=5))
    assert with_steps["max_steps"] == 5 and "num_train_epochs" not in with_steps
    without = build_training_kwargs(load_run_config_from_file(_config(tmp_path)))
    assert without["num_train_epochs"] == 1 and "max_steps" not in without
    assert without["report_to"] == [] and without["dataset_text_field"] == "text"
    assert without["disable_tqdm"] is True
    assert "use_cpu" not in without  # only the toy forces CPU


def test_cpu_toy_kwargs_force_cpu(tmp_path):
    kw = build_training_kwargs(load_run_config_from_file(_config(tmp_path), cpu_toy=True))
    assert kw["use_cpu"] is True and kw["bf16"] is False and kw["fp16"] is False


# ---- run-plan resolution -----------------------------------------------------


def _report(ready: bool, cpu_toy_ready: bool) -> TrainingRuntimeReport:
    return TrainingRuntimeReport(ready=ready, cpu_toy_ready=cpu_toy_ready)


def test_cpu_toy_plan_requires_cpu_toy_ready():
    cfg = _cfg(cpu_toy=True)
    plan = resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=True))
    assert plan == {"device": "cpu", "quantize": False}
    with pytest.raises(TrainerError):
        resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=False))


def test_real_plan_requires_full_ready():
    cfg = _cfg(cpu_toy=False)
    plan = resolve_run_plan(cfg, _report(ready=True, cpu_toy_ready=True))
    assert plan == {"device": "cuda", "quantize": True}
    with pytest.raises(TrainerError):
        resolve_run_plan(cfg, _report(ready=False, cpu_toy_ready=True))


def test_resolved_execution_maps_without_reintroducing_trainer_defaults():
    from corpus_studio.platform.runners import demo_training_plan

    execution = demo_training_plan().resolved_execution
    assert execution is not None
    cfg = train_config_from_resolved(execution)
    assert cfg.execution_configuration_hash == execution.configuration_hash
    assert cfg.max_steps == 2
    assert cfg.device_map == {"": "cpu"}
    assert cfg.lora_dropout == execution.adapter.lora_dropout
    assert cfg.lora_bias == execution.adapter.bias
    assert cfg.package_versions["transformers"]


def test_sealed_runtime_refuses_dataset_byte_drift(tmp_path):
    from corpus_studio.platform.execution_config import stable_file_sha256

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    cfg = _sealed_config(
        dataset_path=str(dataset),
        dataset_sha256=stable_file_sha256(dataset),
        package_versions={},
    )
    assert verify_sealed_runtime(cfg) == dataset.read_bytes()
    dataset.write_text('{"instruction":"changed","output":"b"}\n', encoding="utf-8")
    with pytest.raises(TrainerError, match="dataset bytes changed"):
        verify_sealed_runtime(cfg)


def test_sealed_runtime_refuses_package_drift(tmp_path, monkeypatch):
    from corpus_studio.platform.execution_config import stable_file_sha256

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    cfg = _sealed_config(
        dataset_path=str(dataset),
        dataset_sha256=stable_file_sha256(dataset),
        package_versions={"transformers": "1.2.3"},
    )
    monkeypatch.setattr(trainer_module.importlib.metadata, "version", lambda _name: "9.9.9")
    with pytest.raises(TrainerError, match="sealed package drift"):
        verify_sealed_runtime(cfg)


def test_local_model_and_tokenizer_are_rehashed_after_loading(tmp_path):
    from corpus_studio.platform.execution_config import stable_directory_sha256

    model = tmp_path / "model"
    model.mkdir()
    config_file = model / "config.json"
    config_file.write_text("{}", encoding="utf-8")
    digest = stable_directory_sha256(model)
    cfg = _sealed_config(
        base_model=str(model),
        model_source="local_directory",
        model_content_sha256=digest,
        tokenizer_source="local_directory",
        tokenizer_location=str(model),
        tokenizer_content_sha256=digest,
    )
    verify_local_inputs_after_load(cfg)

    config_file.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(TrainerError, match="bytes changed while loading"):
        verify_local_inputs_after_load(cfg)


def test_local_input_recheck_rejects_unsupported_conflicting_and_missing_bindings(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TrainerError, match="unsupported local binding"):
        verify_local_inputs_after_load(
            _sealed_config(
                base_model=str(model),
                model_source="local_file",
                model_content_sha256="a" * 64,
            )
        )
    with pytest.raises(TrainerError, match="bindings disagree"):
        verify_local_inputs_after_load(
            _sealed_config(
                base_model=str(model),
                model_source="local_directory",
                model_content_sha256="a" * 64,
                tokenizer_source="local_directory",
                tokenizer_location=str(model),
                tokenizer_content_sha256="b" * 64,
            )
        )
    with pytest.raises(TrainerError, match="directory does not exist"):
        verify_local_inputs_after_load(
            _sealed_config(
                base_model=str(tmp_path / "missing"),
                model_source="local_directory",
                model_content_sha256="a" * 64,
                tokenizer_source="huggingface",
            )
        )
    verify_local_inputs_after_load(TrainRunConfig(base_model="m", dataset_path="d"))


def test_trainable_precision_enforces_master_weights_and_gradient_contract():
    torch = _FakeTorch()

    class Data:
        def __init__(self, owner):
            self.owner = owner

        def to(self, *, dtype):
            self.owner.dtype = dtype
            return self

    class Parameter:
        def __init__(self, *, trainable):
            self.requires_grad = trainable
            self.dtype = torch.float16
            self.device = "cuda:0"
            self.data = Data(self)
            self.hook = None

        def register_hook(self, hook):
            self.hook = hook

    frozen = Parameter(trainable=False)
    adapter = Parameter(trainable=True)
    model = type(
        "Model",
        (),
        {"named_parameters": lambda self: iter([("base", frozen), ("adapter", adapter)])},
    )()
    cfg = _sealed_config(master_weight_dtype="fp32", gradient_dtype="fp32")
    enforce_trainable_precision(model, torch, cfg)
    assert adapter.dtype is torch.float32 and adapter.hook is not None

    good = type("Gradient", (), {"dtype": torch.float32, "device": "cuda:0"})()
    assert adapter.hook(good) is good
    bad_dtype = type("Gradient", (), {"dtype": torch.float16, "device": "cuda:0"})()
    with pytest.raises(TrainerError, match="gradient dtype deviation"):
        adapter.hook(bad_dtype)
    bad_device = type("Gradient", (), {"dtype": torch.float32, "device": "cpu"})()
    with pytest.raises(ExecutionPlacementDeviation, match="gradient adapter"):
        adapter.hook(bad_device)


def test_trainable_precision_refuses_missing_or_empty_adapter_state():
    torch = _FakeTorch()
    empty = type("Model", (), {"named_parameters": lambda self: iter(())})()
    with pytest.raises(TrainerError, match="master-weight dtype"):
        enforce_trainable_precision(empty, torch, _sealed_config(master_weight_dtype=None))
    with pytest.raises(TrainerError, match="no trainable parameters"):
        enforce_trainable_precision(empty, torch, _sealed_config(master_weight_dtype="fp32"))


def test_formatter_identity_hashes_the_renderer_implementation(monkeypatch):
    import corpus_studio.platform.execution_config as execution_module

    formatter_id, original_hash = execution_module.formatter_identity("instruction")
    real_getsource = execution_module.inspect.getsource
    monkeypatch.setattr(
        execution_module.inspect,
        "getsource",
        lambda value: real_getsource(value) + "\n# synthetic implementation change\n",
    )
    changed_id, changed_hash = execution_module.formatter_identity("instruction")
    assert changed_id == formatter_id
    assert changed_hash != original_hash


def test_stabilized_dataset_bytes_are_the_bytes_that_get_parsed(tmp_path):
    from corpus_studio.importers.jsonl_importer import read_jsonl_bytes
    from corpus_studio.platform.execution_config import stable_file_bytes

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"instruction":"old","output":"a"}\n', encoding="utf-8")
    content, digest = stable_file_bytes(dataset)
    dataset.write_text('{"instruction":"new","output":"b"}\n', encoding="utf-8")
    assert list(read_jsonl_bytes(content))[0]["instruction"] == "old"
    assert digest != stable_file_bytes(dataset)[1]


def test_list_checkpoints(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "not-a-checkpoint").mkdir()
    found = _list_checkpoints(tmp_path)
    assert len(found) == 2 and all("checkpoint-" in c for c in found)


# ---- helpers -----------------------------------------------------------------


def _cfg(**overrides):
    from corpus_studio.training.trainer import TrainRunConfig

    base = {"base_model": "m", "dataset_path": "d.jsonl"}
    base.update(overrides)
    return TrainRunConfig(**base)
