"""Provider integrations with safe parsing, retry, and fallback behavior."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import Settings
from postprocess import deduplicate_captions, enforce_platform
from prompts import build_caption_prompt, build_refinement_prompt, build_vision_prompt


class ProviderError(RuntimeError):
    """A provider call failed in a user-actionable way."""


class InvalidModelResponse(ValueError):
    """A provider returned unusable structured output."""


def _safe_provider_detail(exc: Exception) -> str:
    """Return useful provider diagnostics without exposing credentials or payloads."""
    detail = str(exc).replace("\r", " ").replace("\n", " ").strip()
    detail = re.sub(r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*\S+", r"\1=[redacted]", detail)
    detail = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted-key]", detail)
    if not detail:
        detail = exc.__class__.__name__
    return detail[:300]


@dataclass
class GenerationResult:
    captions: list[dict[str, Any]]
    provider: str
    model: str


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise InvalidModelResponse("The AI returned invalid JSON.") from exc
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as nested:
            raise InvalidModelResponse("The AI returned invalid JSON.") from nested
    if not isinstance(value, dict) or not isinstance(value.get("captions"), list):
        raise InvalidModelResponse("The AI response is missing a captions list.")
    captions = []
    for item in value["captions"]:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        tags = item.get("hashtags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        captions.append({"text": str(item["text"]), "hashtags": tags, "score": item.get("score"), "reason": str(item.get("reason", ""))})
    if not captions:
        raise InvalidModelResponse("The AI response did not contain usable captions.")
    value["captions"] = captions
    return value


def _groq_call(settings: Settings, prompt: str) -> str:
    if not settings.groq_api_key:
        raise ProviderError("Groq credentials are not configured.")
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "system", "content": "Return only the requested JSON."}, {"role": "user", "content": prompt}],
            temperature=0.8,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise ProviderError(f"Groq generation failed: {_safe_provider_detail(exc)}") from exc


def _gemini_call(settings: Settings, prompt: str, image_bytes: bytes | None = None) -> str:
    if not settings.gemini_api_key:
        raise ProviderError("Gemini credentials are not configured.")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        contents: list[Any] = [prompt]
        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        response = client.models.generate_content(model=settings.gemini_model, contents=contents)
        return response.text or ""
    except Exception as exc:
        raise ProviderError(f"Gemini generation failed: {_safe_provider_detail(exc)}") from exc


@retry(retry=retry_if_exception_type((TimeoutError, ConnectionError)), wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
def _vision_call(settings: Settings, image_bytes: bytes) -> str:
    return _gemini_call(settings, build_vision_prompt(), image_bytes)


def describe_image(settings: Settings, image_bytes: bytes) -> str:
    return _vision_call(settings, image_bytes)


def generate_captions(settings: Settings, prompt_args: dict[str, Any], *, image_bytes: bytes | None = None) -> GenerationResult:
    prompt = build_caption_prompt(**prompt_args)
    raw: str | None = None
    provider = ""
    model = ""
    errors: list[str] = []
    for name, call, configured, selected_model in (
        ("Groq", lambda request: _groq_call(settings, request), settings.groq_configured, settings.groq_model),
        ("Gemini", lambda request: _gemini_call(settings, request, image_bytes), settings.gemini_configured, settings.gemini_model),
    ):
        if not configured:
            continue
        try:
            raw = call(prompt)
            try:
                parsed = parse_json_response(raw)
            except InvalidModelResponse:
                correction = prompt + "\nYour previous response was invalid. Return corrected JSON only."
                parsed = parse_json_response(call(correction))
            provider, model = name, selected_model
            captions = [enforce_platform(item, prompt_args["platform"], prompt_args.get("emoji_enabled", True)) for item in parsed["captions"]]
            return GenerationResult(deduplicate_captions(captions), provider, model)
        except (ProviderError, InvalidModelResponse) as exc:
            errors.append(str(exc))
    raise ProviderError("No caption provider succeeded. " + " ".join(errors))


def refine_caption(settings: Settings, caption: dict[str, Any], *, action: str, platform: str, tone: str, language: str, emoji_enabled: bool = True) -> GenerationResult:
    prompt = build_refinement_prompt(caption.get("text", ""), action, platform, tone, language)
    errors: list[str] = []
    for name, call, configured, model in (
        ("Groq", lambda: _groq_call(settings, prompt), settings.groq_configured, settings.groq_model),
        ("Gemini", lambda: _gemini_call(settings, prompt), settings.gemini_configured, settings.gemini_model),
    ):
        if not configured:
            continue
        try:
            raw = call()
            data = parse_json_response(raw) if '"captions"' in raw else {"captions": [json.loads(raw)]}
            result = enforce_platform(data["captions"][0], platform, emoji_enabled)
            return GenerationResult([result], name, model)
        except (ProviderError, InvalidModelResponse, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    raise ProviderError("Caption refinement failed. " + " ".join(errors))
