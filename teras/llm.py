import json
import os

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types


class LLMNotConfigured(Exception):
    pass


def provider() -> str | None:
    load_dotenv(override=False)
    selected = os.getenv("LLM_PROVIDER", "").strip().lower()
    if selected:
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
    active = provider()
    if active == "claude":
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
    if active == "gemini":
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    raise LLMNotConfigured("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI features.")


def model_name() -> str:
    active = provider()
    if active == "claude":
        return os.getenv("CLAUDE_MODEL", "").strip() or "claude-opus-5-5"
    if active == "gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
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
            model=model_name(),
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


def generate(prompt: str, system: str, json_mode: bool = False) -> str:
    active = provider()
    client = get_client()
    if active == "claude":
        return _generate_claude(client, prompt, system, json_mode)
    try:
        response = client.models.generate_content(
            model=model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                response_mime_type="application/json" if json_mode else None,
            ),
        )
        return response.text or ""
    except Exception as exc:
        key = os.getenv("GEMINI_API_KEY", "")
        message = str(exc).replace(key, "[redacted]") if key else str(exc)
        raise RuntimeError(f"Gemini request failed: {message}") from None


def generate_json(prompt, system) -> dict | list:
    response = generate(prompt, system, json_mode=True)
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
