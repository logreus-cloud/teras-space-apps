from types import SimpleNamespace

import pytest

from teras import llm


@pytest.fixture(autouse=True)
def empty_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "CLAUDE_MODEL", "GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "CLAUDE_EFFORT",
        "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY",
        "LLM_CLOUD_FALLBACK", "OLLAMA_NUM_CTX",
    ):
        monkeypatch.setenv(name, "")


@pytest.mark.parametrize(
    ("selected", "anthropic_key", "gemini_key", "expected"),
    [
        ("", "claude-key", "gemini-key", "claude"),
        ("", "", "gemini-key", "gemini"),
        ("claude", "claude-key", "gemini-key", "claude"),
        ("gemini", "claude-key", "gemini-key", "gemini"),
        ("claude", "", "gemini-key", None),
        ("", "", "", None),
    ],
)
def test_provider_selection(monkeypatch, selected, anthropic_key, gemini_key, expected):
    monkeypatch.setenv("LLM_PROVIDER", selected)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    assert llm.provider() == expected
    assert llm.is_configured() is (expected is not None)


def test_no_keys():
    with pytest.raises(llm.LLMNotConfigured):
        llm.generate("prompt", "system")
    with pytest.raises(llm.LLMNotConfigured):
        llm.model_name()


def test_model_names(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    assert llm.model_name() == "claude-opus-5-5"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert llm.model_name() == "gemini-3.5-flash"


def test_claude_text_and_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="first"),
                SimpleNamespace(type="thinking", text="skip"),
                SimpleNamespace(type="text", text="second"),
            ],
        )

    def client(*, api_key):
        assert api_key == "claude-key"
        return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm.anthropic, "Anthropic", client)
    assert llm.generate("prompt", "system", json_mode=True) == "firstsecond"
    request = calls[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["max_tokens"] == 16000
    assert request["system"] == "system"
    assert request["messages"] == [
        {"role": "user", "content": "prompt\n\nReturn only valid JSON, no markdown fences."}
    ]
    assert request["output_config"] == {"effort": "low"}
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert not {"thinking", "temperature", "top_p"} & request.keys()


def test_claude_refusal(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    response = SimpleNamespace(stop_reason="refusal", content=[])
    messages = SimpleNamespace(create=lambda **kwargs: response)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(llm.anthropic, "Anthropic", lambda api_key: client)
    with pytest.raises(RuntimeError, match="Claude declined the request"):
        llm.generate("prompt", "system")


def test_load_dotenv_does_not_override_env(monkeypatch):
    overrides = []
    monkeypatch.setattr(llm, "load_dotenv", lambda *, override: overrides.append(override))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    assert llm.provider() == "claude"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert llm.provider() == "gemini"
    assert overrides == [False, False]


class GeminiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def _gemini_scenario(monkeypatch, outcomes):
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs["model"])
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "main")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup")
    monkeypatch.setattr(llm.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    return calls


def test_gemini_retries_then_uses_fallback(monkeypatch):
    calls = _gemini_scenario(monkeypatch, [GeminiError(503), GeminiError(503), "OK"])
    assert llm.generate("prompt", "system") == "OK"
    assert calls == ["main", "main", "backup"]


def test_gemini_all_models_overloaded(monkeypatch):
    calls = _gemini_scenario(monkeypatch, [GeminiError(503) for _ in range(4)])
    with pytest.raises(RuntimeError, match="Gemini is overloaded"):
        llm.generate("prompt", "system")
    assert calls == ["main", "main", "backup", "backup"]


def test_gemini_skips_missing_model(monkeypatch):
    calls = _gemini_scenario(monkeypatch, [GeminiError(404), "OK"])
    assert llm.generate("prompt", "system") == "OK"
    assert calls == ["main", "backup"]


def test_gemini_stops_on_bad_request(monkeypatch):
    calls = _gemini_scenario(monkeypatch, [GeminiError(400)])
    with pytest.raises(RuntimeError, match="Gemini request failed: HTTP 400"):
        llm.generate("prompt", "system")
    assert calls == ["main"]


def test_ollama_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm.provider() == "ollama"
    assert llm.is_configured()
    assert llm.model_name() == "qwen3:8b"
    monkeypatch.setenv("OLLAMA_MODEL", "custom:14b")
    assert llm.model_name() == "custom:14b"


@pytest.mark.parametrize(("token", "json_mode"), [("", False), ("tunnel-token", True)])
def test_ollama_request(monkeypatch, token, json_mode):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_API_KEY", token)
    calls = []

    def post(url, *, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "<think>work\nmore</think> result "}},
        )

    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.generate("prompt", "system", json_mode=json_mode) == "result"
    assert llm.last_model_used() == "qwen3:8b"
    url, payload, headers, timeout = calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen3:8b"
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.2, "num_ctx": 8192}
    assert payload["think"] is False
    assert payload.get("format") == ("json" if json_mode else None)
    assert headers == ({"Authorization": f"Bearer {token}"} if token else {})
    assert timeout == (5, 180)


def test_ollama_falls_back_to_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "cloud-model")

    def fail(*args, **kwargs):
        raise llm.requests.ConnectionError("offline")

    monkeypatch.setattr(llm.requests, "post", fail)
    monkeypatch.setattr(llm.genai, "Client", lambda api_key: object())
    calls = []

    def generate_gemini(client, prompt, system, json_mode):
        calls.append((prompt, system, json_mode))
        return "cloud"

    monkeypatch.setattr(llm, "_generate_gemini", generate_gemini)
    assert llm.generate("prompt", "system", json_mode=True) == "cloud"
    assert calls == [("prompt", "system", True)]
    assert llm.last_model_used() == "cloud-model"


@pytest.mark.parametrize(("fallback", "gemini_key"), [("0", "gemini-key"), ("1", "")])
def test_ollama_unreachable_without_fallback(monkeypatch, fallback, gemini_key):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_CLOUD_FALLBACK", fallback)
    monkeypatch.setenv("GEMINI_API_KEY", gemini_key)
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")

    def fail(*args, **kwargs):
        raise llm.requests.ConnectionError("secret-token")

    monkeypatch.setattr(llm.requests, "post", fail)
    with pytest.raises(RuntimeError, match="unreachable") as error:
        llm.generate("prompt", "system")
    assert "secret-token" not in str(error.value)
    assert llm.last_model_used() is None


def test_ollama_rejects_non_http_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "file:///tmp/ollama")
    with pytest.raises(llm.LLMNotConfigured):
        llm.generate("prompt", "system")
