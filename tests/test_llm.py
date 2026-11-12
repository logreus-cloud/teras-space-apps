from types import SimpleNamespace

import pytest

from teras import llm


@pytest.fixture(autouse=True)
def empty_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "CLAUDE_MODEL", "GEMINI_MODEL", "CLAUDE_EFFORT",
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
    assert llm.model_name() == "gemini-2.5-flash"


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
