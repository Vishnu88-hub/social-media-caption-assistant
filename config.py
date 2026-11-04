"""Application configuration loaded from environment variables or Streamlit secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

load_dotenv()


def _has_local_secrets_file() -> bool:
    project_secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    user_secrets = Path.home() / ".streamlit" / "secrets.toml"
    return project_secrets.is_file() or user_secrets.is_file()


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    if not _has_local_secrets_file():
        return default
    try:
        import streamlit as st

        value = st.secrets.get(name, default)
    except (ImportError, FileNotFoundError, RuntimeError, KeyError):
        value = default
    return str(value or "").strip()


def _model_list(value: str, fallback: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    return models or (fallback,)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    groq_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "openai/gpt-oss-120b"
    gemini_models: tuple[str, ...] = ("gemini-2.5-flash", "gemini-2.5-flash-lite")
    groq_models: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    database_path: str = "captions.db"
    max_image_bytes: int = 10 * 1024 * 1024
    max_idea_chars: int = 500
    generation_cooldown_seconds: float = 2.0
    session_request_limit: int = 30
    rag_path: str = "rag_data"
    embedding_model: str = "all-MiniLM-L6-v2"

    @classmethod
    def load(cls, overrides: Mapping[str, Any] | None = None) -> "Settings":
        values: dict[str, Any] = {
            "gemini_api_key": _secret("GEMINI_API_KEY"),
            "groq_api_key": _secret("GROQ_API_KEY"),
            "gemini_model": _secret("GEMINI_MODEL", "gemini-2.5-flash"),
            "groq_model": _secret("GROQ_MODEL", "openai/gpt-oss-120b"),
            "database_path": _secret("CAPTION_DB_PATH", "captions.db"),
            "max_image_bytes": int(_secret("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
            "max_idea_chars": int(_secret("MAX_IDEA_CHARS", "500")),
            "generation_cooldown_seconds": float(_secret("GENERATION_COOLDOWN_SECONDS", "2")),
            "session_request_limit": int(_secret("SESSION_REQUEST_LIMIT", "30")),
            "rag_path": _secret("RAG_PATH", "rag_data"),
            "embedding_model": _secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        }
        values["gemini_models"] = _model_list(
            _secret("GEMINI_MODELS", values["gemini_model"]), values["gemini_model"]
        )
        values["groq_models"] = _model_list(
            _secret("GROQ_MODELS", values["groq_model"]), values["groq_model"]
        )
        if overrides:
            values.update(overrides)
        return cls(**values)

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def ai_configured(self) -> bool:
        return self.groq_configured or self.gemini_configured

    def safe_status(self) -> dict[str, str]:
        return {
            "Groq": f"configured ({', '.join(self.groq_models)})" if self.groq_configured else "not configured",
            "Gemini": f"configured ({', '.join(self.gemini_models)})" if self.gemini_configured else "not configured",
        }


def database_path(settings: Settings | None = None) -> Path:
    return Path((settings or Settings.load()).database_path)
