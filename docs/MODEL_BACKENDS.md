# Model Backends

Model backends are the abstraction that lets Evaluation Lab and AI Assist Lab
run models without hardcoding one provider.

Corpus Studio should remain local-first. Hosted providers can be supported later
through explicit user configuration, but local backends should be the default
path.

## Backend Roadmap

Current local backends:

- Ollama
- LM Studio or another OpenAI-compatible local endpoint

Planned explicit opt-in backends:

- Custom HTTP endpoint
- OpenAI-compatible hosted provider
- Anthropic later
- Hugging Face Inference later

Current engine status: Ollama and OpenAI-compatible adapters exist for explicit
Evaluation Lab and AI Assist Lab MVP runs. The engine also exposes
`backend-health` and `model-list` commands so the app can check connectivity
and list locally available models, including Ollama models returned from
`/api/tags`. They use the Python standard library for HTTP calls and do not add
heavy ML dependencies. Unit tests inject fake openers or fake backends so test
runs do not contact real endpoints.

Current desktop status: the Evaluation and AI Assist tabs pass backend settings
to engine commands, expose Check Backend and Refresh Models buttons, display
resulting health/model-list output, and let users pick from discovered model
names while still allowing manual entry. The Settings tab can save Evaluation
and AI Assist backend choices into project-local `lab_settings` metadata. The
desktop app should continue to treat providers as engine-owned adapters rather
than duplicating provider-specific HTTP code in C#.

## Conceptual Backend Interface

Every backend should eventually support:

```text
list_models()
generate(prompt/messages, options)
stream_generate(prompt/messages, options)
health_check()
```

Implementation details may differ, but the app should be able to ask the same
questions:

- Is the backend reachable?
- Which models are available?
- Can this backend accept a plain prompt?
- Can this backend accept chat messages?
- Does this backend support streaming?
- What timeout and token limits should be used?

## Provider Config Fields

Provider config should include:

- `provider_name`
- `base_url`
- `api_key_optional`
- `model_name`
- `timeout_seconds`
- `max_tokens`
- `temperature`
- `top_p`
- `streaming_enabled`

API keys must be optional for local providers. Secrets should never be committed
to the repository or stored in project files without explicit user consent.

## Local-First Defaults

Recommended local defaults:

- Ollama base URL: `http://localhost:11434`
- OpenAI-compatible local base URL: `http://localhost:1234/v1`
- timeout: 120 seconds
- max tokens: 1024
- temperature: 0.2 for evaluation
- streaming disabled for first report exports

## Current Commands

```bash
python -m corpus_studio.cli model-list --backend ollama
python -m corpus_studio.cli backend-health --backend ollama --model qwen2.5-coder:7b
python -m corpus_studio.cli eval-run examples/datasets/instruction/train.jsonl instruction --backend ollama --model qwen2.5-coder:7b --limit 5
python -m corpus_studio.cli ai-assist examples/datasets/instruction/train.jsonl instruction --action review --backend ollama --model qwen2.5-coder:7b
```

`model-list` is safe to use for discovery. It lists model names exposed by the
configured backend and returns clean JSON when the backend is unreachable. It
does not pull, download, train, or run a model.

## Backend Responsibilities

Backends should own:

- provider-specific request shape
- provider-specific response parsing
- authentication header construction
- timeout handling
- retry with bounded backoff for transient failures (HTTP 429/5xx + connection
  errors; other 4xx fail fast) — health/model-list probes stay single-attempt
- health checks
- stream parsing when enabled

Backends should not own:

- dataset schema validation
- evaluation scoring policy
- accepted-example mutation
- training configuration
- cloud account management

## Testing Rule

Unit tests for backend config must not make network calls. Network behavior is
covered by opt-in integration tests (`engine/tests/test_ollama_integration.py`)
that require `CORPUS_STUDIO_OLLAMA_INTEGRATION=1` and a running local backend;
the tests self-skip when the backend is unavailable.
