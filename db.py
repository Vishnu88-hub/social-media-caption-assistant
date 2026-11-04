"""SQLite persistence for history, favourites, and brand voice samples."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        initialize(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, hashtags TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, platform TEXT NOT NULL,
            tone TEXT NOT NULL, language TEXT NOT NULL, idea TEXT NOT NULL DEFAULT '',
            score REAL, reason TEXT NOT NULL DEFAULT '', favourite INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS brand_voice (
            id INTEGER PRIMARY KEY AUTOINCREMENT, sample TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def save_caption(path: str | Path, caption: dict, *, platform: str, tone: str, language: str, idea: str) -> int:
    with connection(path) as conn:
        cursor = conn.execute(
            "INSERT INTO captions (text, hashtags, platform, tone, language, idea, score, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (caption.get("text", ""), " ".join(caption.get("hashtags", [])), platform, tone, language, idea,
             caption.get("score"), caption.get("reason", "")),
        )
        return int(cursor.lastrowid)


def list_captions(path: str | Path, *, platform: str | None = None, favourites_only: bool = False) -> list[dict]:
    query = "SELECT * FROM captions WHERE 1=1"
    params: list[object] = []
    if platform:
        query += " AND platform = ?"
        params.append(platform)
    if favourites_only:
        query += " AND favourite = 1"
    query += " ORDER BY created_at DESC"
    with connection(path) as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def set_favourite(path: str | Path, caption_id: int, favourite: bool) -> None:
    with connection(path) as conn:
        conn.execute("UPDATE captions SET favourite = ? WHERE id = ?", (int(favourite), caption_id))


def delete_caption(path: str | Path, caption_id: int) -> None:
    with connection(path) as conn:
        conn.execute("DELETE FROM captions WHERE id = ?", (caption_id,))


def add_brand_voice(path: str | Path, sample: str) -> int:
    if not sample.strip():
        raise ValueError("Brand voice sample cannot be empty.")
    with connection(path) as conn:
        cursor = conn.execute("INSERT INTO brand_voice (sample) VALUES (?)", (sample.strip(),))
        return int(cursor.lastrowid)


def list_brand_voice(path: str | Path) -> list[dict]:
    with connection(path) as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM brand_voice ORDER BY created_at DESC")]


def delete_brand_voice(path: str | Path, sample_id: int) -> None:
    with connection(path) as conn:
        conn.execute("DELETE FROM brand_voice WHERE id = ?", (sample_id,))
