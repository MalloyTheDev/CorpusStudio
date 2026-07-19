"""S1: eval-run decodes deterministically (seeded, greedy, non-truncating) and records HOW it decoded,
so every eval number is reproducible and a long structured output is not fake-truncated by a small cap."""
from corpus_studio.evaluation.evaluator import (
    EvaluationDatasetExample,
    EvaluationRunConfig,
    run_evaluation,
)
from corpus_studio.model_backends.base import BackendGenerateRequest, BackendGenerateResponse
from corpus_studio.model_backends.ollama import OllamaBackend, default_ollama_config
from corpus_studio.model_backends.openai_compatible import (
    OpenAICompatibleBackend,
    default_openai_compatible_config,
)


def test_ollama_options_carry_seed_and_cap_and_omit_seed_when_none():
    backend = OllamaBackend(default_ollama_config("m"))
    opts = backend._options(BackendGenerateRequest(seed=123, temperature=0.0, max_tokens=2048))
    assert opts["seed"] == 123
    assert opts["num_predict"] == 2048
    assert opts["temperature"] == 0.0
    assert "seed" not in backend._options(BackendGenerateRequest())  # omitted, never a null


def test_openai_payload_carries_seed_and_omits_it_when_none(monkeypatch):
    backend = OpenAICompatibleBackend(default_openai_compatible_config("m"))
    captured: dict = {}

    def _fake(method, path, payload=None, **kwargs):
        captured["body"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(backend, "_request_json", _fake)
    backend.generate(BackendGenerateRequest(prompt="hi", seed=7, temperature=0.0, max_tokens=2048))
    assert captured["body"]["seed"] == 7
    assert captured["body"]["max_tokens"] == 2048
    backend.generate(BackendGenerateRequest(prompt="hi"))  # no seed
    assert "seed" not in captured["body"]


class _CapturingBackend:
    def __init__(self):
        self.requests: list[BackendGenerateRequest] = []

    def generate(self, request):
        self.requests.append(request)
        return BackendGenerateResponse(text="answer", model_name="m")


def test_run_evaluation_threads_decode_config_into_request_and_records_it():
    config = EvaluationRunConfig(
        dataset="d", model="m", schema_id="instruction",
        seed=99, temperature=0.0, max_output_tokens=1500,
    )
    backend = _CapturingBackend()
    examples = [EvaluationDatasetExample(example_id="1", prompt="p", expected_output="answer")]
    report = run_evaluation(config, examples, backend)

    # the decode knobs reach the backend request
    req = backend.requests[0]
    assert req.seed == 99
    assert req.max_tokens == 1500
    assert req.temperature == 0.0
    # ... and are recorded in the saved run settings (reproducibility evidence)
    assert report.run_settings is not None
    assert report.run_settings.seed == 99
    assert report.run_settings.max_output_tokens == 1500
    assert report.run_settings.temperature == 0.0


def test_decode_defaults_are_deterministic_greedy_with_a_generous_cap():
    config = EvaluationRunConfig(dataset="d", model="m", schema_id="instruction")
    assert config.seed == 0
    assert config.temperature == 0.0  # greedy by default
    assert config.max_output_tokens >= 1358  # >= the longest WBG gold completion, so no fake truncation
