"""Optional local retrieval-augmented generation knowledge base.

The vector store contains only explicitly approved caption examples and
brand guidance. Provider calls and uploaded images are never stored here.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RagError(RuntimeError):
    """Raised when the optional RAG store cannot be used."""


def _metadata_value(value: Any) -> str:
    return str(value or "").strip()


@lru_cache(maxsize=2)
def _embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RagError("Sentence Transformers is not installed.") from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RagError("The local embedding model could not be loaded.") from exc


class RagStore:
    """Persistent ChromaDB store with lazy embedding-model loading."""

    def __init__(self, path: str | Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.path = Path(path)
        self.model_name = model_name
        self._client = None
        self._collection = None

    def _collection_or_raise(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(
                name="caption_knowledge",
                metadata={"description": "Explicitly approved caption and brand examples"},
            )
            return self._collection
        except ImportError as exc:
            raise RagError("ChromaDB is not installed.") from exc
        except Exception as exc:
            raise RagError("The local RAG database could not be opened.") from exc

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = _embedding_model(self.model_name)
        try:
            return model.encode(texts, normalize_embeddings=True).tolist()
        except Exception as exc:
            raise RagError("The local embedding model failed to encode text.") from exc

    @staticmethod
    def _id(text: str, metadata: dict[str, Any]) -> str:
        stable = text + "|" + "|".join(f"{key}={metadata.get(key, '')}" for key in sorted(metadata))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()

    def add_example(self, text: str, metadata: dict[str, Any]) -> bool:
        text = text.strip()
        if not text:
            raise ValueError("Cannot add an empty RAG example.")
        collection = self._collection_or_raise()
        clean_metadata = {key: _metadata_value(value) for key, value in metadata.items()}
        item_id = self._id(text, clean_metadata)
        if collection.get(ids=[item_id], include=[]).get("ids"):
            return False
        collection.add(
            ids=[item_id],
            documents=[text],
            embeddings=self._embed([text]),
            metadatas=[clean_metadata],
        )
        return True

    def add_caption(self, caption: dict[str, Any], *, platform: str, tone: str, language: str, length: str = "", source: str = "approved") -> bool:
        text = str(caption.get("text", "")).strip()
        hashtags = " ".join(str(tag) for tag in caption.get("hashtags", []))
        document = f"{text}\nHashtags: {hashtags}".strip()
        return self.add_example(document, {
            "platform": platform, "tone": tone, "language": language,
            "length": length, "source": source, "approval": "approved",
        })

    def add_brand_guideline(self, text: str, *, platform: str = "", tone: str = "", language: str = "") -> bool:
        return self.add_example(text, {
            "platform": platform, "tone": tone, "language": language,
            "source": "brand_guideline", "approval": "approved",
        })

    def count(self) -> int:
        return int(self._collection_or_raise().count())

    def retrieve(self, query: str, *, platform: str = "", tone: str = "", language: str = "", limit: int = 4) -> list[dict[str, Any]]:
        if not query.strip() or limit < 1:
            return []
        collection = self._collection_or_raise()
        if collection.count() == 0:
            return []
        where_parts = [{key: value} for key, value in (("platform", platform), ("tone", tone), ("language", language)) if value]
        where = {"$and": where_parts} if len(where_parts) > 1 else (where_parts[0] if where_parts else None)
        request: dict[str, Any] = {
            "query_embeddings": self._embed([query]),
            "n_results": min(limit, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            request["where"] = where
        try:
            result = collection.query(**request)
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            return [
                {"text": document, "metadata": metadata or {}, "distance": distance}
                for document, metadata, distance in zip(documents, metadatas, distances)
            ]
        except Exception as exc:
            raise RagError("The local RAG retrieval failed.") from exc

    def has_data(self) -> bool:
        return self.count() > 0


def format_retrieved_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    examples = "\n\n".join(f"Reference {index}: {item['text']}" for index, item in enumerate(items, 1))
    return (
        "Relevant approved references follow. Use them only as style and guidance; "
        "do not copy them verbatim or repeat their unique wording.\n" + examples
    )
