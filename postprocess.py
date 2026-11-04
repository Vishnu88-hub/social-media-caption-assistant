"""Deterministic caption cleanup and platform-limit enforcement."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from prompts import PlatformRule, platform_rule

EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U0001F1E6-\U0001F1FF]", flags=re.UNICODE
)
HASHTAG_RE = re.compile(r"(?<!\w)#([\w\u0080-\uffff]+)", re.UNICODE)


def normalize_hashtags(tags: list[str] | str, rule: PlatformRule) -> list[str]:
    values = tags.split() if isinstance(tags, str) else tags
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        cleaned = re.sub(r"[^\w\u0080-\uffff]", "", str(raw).lstrip("#"), flags=re.UNICODE)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(f"#{cleaned}")
    return result[: rule.hashtag_max]


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub("", text).replace("  ", " ").strip()


def trim_readably(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    if limit <= 1:
        return text[:limit]
    candidate = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return ((candidate or text[: limit - 1]).rstrip() + "…")[:limit]


def enforce_platform(caption: dict[str, Any], platform: str, emoji_enabled: bool = True) -> dict[str, Any]:
    rule = platform_rule(platform)
    result = dict(caption)
    text = str(result.get("text", "")).strip()
    if not emoji_enabled:
        text = remove_emojis(text)
    result["hashtags"] = normalize_hashtags(result.get("hashtags", []), rule)
    suffix = " " + " ".join(result["hashtags"]) if result["hashtags"] else ""
    if platform == "YouTube Shorts":
        result["title"] = trim_readably(str(result.get("title") or text), rule.title_limit or 100)
        result["text"] = text
    else:
        available = max(1, rule.caption_limit - len(suffix))
        result["text"] = trim_readably(text, available)
    result["character_count"] = len(result["text"]) + (len(suffix) if platform != "YouTube Shorts" else 0)
    result["character_limit"] = rule.caption_limit
    return result


def deduplicate_captions(captions: list[dict[str, Any]], threshold: float = 0.92) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for caption in captions:
        text = str(caption.get("text", "")).strip()
        if not text:
            continue
        if not any(SequenceMatcher(None, text.casefold(), str(item.get("text", "")).casefold()).ratio() >= threshold for item in unique):
            unique.append(caption)
    return unique
