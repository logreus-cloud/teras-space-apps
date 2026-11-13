import json
import os
import re
import time
from urllib.parse import urlparse

import anthropic
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types


class LLMNotConfigured(Exception):
    pass


# Mutable state records the model used by the latest successful generation.
_last_model_used: str | None = None


def last_model_used() -> str | None:
    return _last_model_used


def _ollama_base_url() -> str:
    base = os.getenv("OLLAMA_BASE_URL", "").strip() or "http://localhost:11434"
    try:
        parsed = urlparse(base)
        valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
        parsed.port  # raises ValueError on a bad port
    except ValueError:
        valid = False
    if not valid:
        raise LLMNotConfigured("OLLAMA_BASE_URL must be an http or https URL.")
    return base


def provider() -> str | None:
    load_dotenv(override=False)
    selected = os.getenv("LLM_PROVIDER", "").strip().lower()
    if selected:
        if selected == "ollama":
            try:
                _ollama_base_url()
            except LLMNotConfigured:
                return None
            return "ollama"
        if selected not in ("claude", "gemini"):
            return None
        key_name = "ANTHROPIC_API_KEY" if selected == "claude" else "GEMINI_API_KEY"
        return selected if os.getenv(key_name, "").strip() else None
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "claude"
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini"
    return None


def is_configured() -> bool:
    return provider() is not None


def get_client():
    return _client_for(provider())


def _client_for(active):
    if active == "claude":
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
    if active == "gemini":
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    raise LLMNotConfigured("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI features.")


def model_name() -> str:
    return _model_for(provider())


def _model_for(active: str | None) -> str:
    if active == "ollama":
        return os.getenv("OLLAMA_MODEL", "").strip() or "qwen3:8b"
    if active == "claude":
        return os.getenv("CLAUDE_MODEL", "").strip() or "claude-opus-5-5"
    if active == "gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
    raise LLMNotConfigured("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI features.")


def _generate_claude(client, prompt, system, json_mode):
    effort = os.getenv("CLAUDE_EFFORT", "").strip().lower() or "low"
    if effort not in ("low", "medium", "high"):
        raise ValueError("CLAUDE_EFFORT must be low, medium, or high.")
    if json_mode:
        prompt += "\n\nReturn only valid JSON, no markdown fences."
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    try:
        response = client.beta.messages.create(
            model=_model_for("claude"),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude request failed: authentication failed; check ANTHROPIC_API_KEY.") from None
    except anthropic.RateLimitError:
        raise RuntimeError("Claude request failed: rate limit exceeded.") from None
    except anthropic.APIStatusError as exc:
        message = str(exc).replace(key, "[redacted]") if key else str(exc)
        raise RuntimeError(f"Claude request failed: HTTP {exc.status_code}: {message}") from None
    except anthropic.APIConnectionError:
        raise RuntimeError("Claude request failed: connection error.") from None
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined the request")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude response was cut off (max_tokens reached)")
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_gemini(client, prompt, system, json_mode):
    global _last_model_used
    fallbacks = os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    )
    models = [_model_for("gemini")]
    for name in fallbacks.split(","):
        name = name.strip()
        if name and name not in models:
            models.append(name)
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.2,
        response_mime_type="application/json" if json_mode else None,
    )
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                _last_model_used = model
                return response.text or ""
            except Exception as exc:
                codes = (getattr(exc, "code", None), getattr(exc, "status_code", None))
                if 404 in codes:
                    break
                message = str(exc)
                definite = any(code in (400, 401, 403) for code in codes)
                temporary = not definite and (
                    any(code in (429, 500, 503, 504) for code in codes)
                    or any(
                        marker in message.upper()
                        for marker in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "503")
                    )
                )
                if temporary:
                    if attempt == 0:
                        time.sleep(1.5)
                    continue
                key = os.getenv("GEMINI_API_KEY", "").strip()
                message = message.replace(key, "[redacted]") if key else message
                raise RuntimeError(f"Gemini request failed: {message}") from None
    raise RuntimeError(
        "Gemini is overloaded right now (all models busy). Try again in a minute."
    )


def _ollama_num_ctx() -> int:
    raw = os.getenv("OLLAMA_NUM_CTX", "").strip() or "8192"
    try:
        return max(2048, min(int(raw), 131072))
    except ValueError:
        raise LLMNotConfigured("OLLAMA_NUM_CTX must be an integer.") from None


def _generate_ollama(prompt, system, json_mode):
    base = _ollama_base_url()
    payload = {
        "model": _model_for("ollama"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": _ollama_num_ctx()},
    }
    if payload["model"].startswith(("qwen3", "deepseek-r1")):
        payload["think"] = False
    if json_mode:
        # A dict is a JSON schema; plain "json" only allows a top-level object.
        payload["format"] = json_mode if isinstance(json_mode, dict) else "json"
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        response = requests.post(
            f"{base.rstrip('/')}/api/chat", json=payload, headers=headers, timeout=(5, 180)
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Invalid content")
    except (requests.ConnectionError, requests.Timeout):
        raise RuntimeError(
            f"Local model (Ollama) is unreachable at {urlparse(base).hostname}"
        ) from None
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(f"Ollama request failed: HTTP {code}") from None
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Ollama returned an invalid JSON response.") from None
    except requests.RequestException:
        raise RuntimeError("Ollama request failed: transport error") from None
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _cloud_fallbacks() -> list:
    names = [("claude", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")]
    return [name for name, key in names if os.getenv(key, "").strip()]


def _generate_cloud(active, prompt, system, json_mode) -> str:
    global _last_model_used
    client = _client_for(active)
    if active == "claude":
        result = _generate_claude(client, prompt, system, json_mode)
        _last_model_used = _model_for("claude")
        return result
    result = _generate_gemini(client, prompt, system, json_mode)
    if _last_model_used is None:
        _last_model_used = _model_for("gemini")
    return result


def generate(prompt: str, system: str, json_mode: bool | dict = False) -> str:
    global _last_model_used
    _last_model_used = None
    active = provider()
    if active != "ollama":
        return _generate_cloud(active, prompt, system, json_mode)
    try:
        result = _generate_ollama(prompt, system, json_mode)
        _last_model_used = _model_for("ollama")
        return result
    except RuntimeError as local_error:
        if os.getenv("LLM_CLOUD_FALLBACK", "1").strip() == "0":
            raise
        error = local_error
    for cloud in _cloud_fallbacks():
        try:
            return _generate_cloud(cloud, prompt, system, json_mode)
        except RuntimeError as cloud_error:
            error = cloud_error
    raise error


def generate_json(prompt, system, schema: dict | None = None) -> dict | list:
    response = generate(prompt, system, json_mode=schema or True)
    source = (provider() or "LLM").title()
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for position, char in enumerate(response):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(response[position:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError(f"{source} did not return valid JSON.") from None
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{source} JSON response must be an object or array.")
    return value
