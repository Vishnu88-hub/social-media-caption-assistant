from io import BytesIO

import pytest
from PIL import Image

from db import add_brand_voice, delete_caption, list_brand_voice, list_captions, save_caption, set_favourite
from image_utils import ImageValidationError, process_image
from config import Settings
from llm import InvalidModelResponse, ProviderError, _safe_provider_detail, generate_captions, parse_json_response
from postprocess import deduplicate_captions, enforce_platform, normalize_hashtags
from prompts import build_caption_prompt
from rag import RagStore, format_retrieved_context


def image_bytes(fmt="PNG"):
    output = BytesIO()
    Image.new("RGB", (20, 10), "red").save(output, fmt)
    return output.getvalue()


def test_image_processing_and_limits():
    result = process_image(image_bytes(), max_dimension=10)
    assert result[:2] == b"\xff\xd8"
    with pytest.raises(ImageValidationError):
        process_image(b"bad")


def test_prompt_contains_preferences_and_rules():
    prompt = build_caption_prompt(platform="X (Twitter)", tone="funny", language="Tamil", length="short", variants=3, idea="hello")
    assert "Tamil" in prompt and "280" in prompt and "funny" in prompt


def test_json_parser_rejects_bad_and_accepts_fences():
    assert len(parse_json_response('```json\n{"captions":[{"text":"Hi","hashtags":[]}]} \n```')["captions"]) == 1
    with pytest.raises(InvalidModelResponse):
        parse_json_response("not json")


def test_postprocessing_limits_and_duplicates():
    assert normalize_hashtags(["#One", "#one", "two"], __import__("prompts").platform_rule("X (Twitter)")) == ["#One", "#two"]
    values = [{"text": "Same caption", "hashtags": []}, {"text": "same caption", "hashtags": []}]
    assert len(deduplicate_captions(values)) == 1
    processed = enforce_platform({"text": "word " * 200, "hashtags": ["#one", "#ONE"]}, "X (Twitter)")
    assert processed["character_count"] <= 280


def test_sqlite_history_favourites_and_brand_voice(tmp_path):
    path = tmp_path / "test.db"
    caption_id = save_caption(path, {"text": "Hello", "hashtags": ["#hi"]}, platform="Instagram", tone="casual", language="English", idea="x")
    set_favourite(path, caption_id, True)
    assert list_captions(path, favourites_only=True)[0]["favourite"] == 1
    sample_id = add_brand_voice(path, "Warm and concise")
    assert list_brand_voice(path)[0]["id"] == sample_id
    delete_caption(path, caption_id)
    assert not list_captions(path)


def test_provider_fallback_and_missing_credentials(monkeypatch):
    settings = Settings(groq_api_key="groq", gemini_api_key="gemini")
    valid = '{"captions":[{"text":"Fallback","hashtags":["#ok"]}]}'
    monkeypatch.setattr("llm._groq_call", lambda *_: (_ for _ in ()).throw(ProviderError("temporary")))
    monkeypatch.setattr("llm._gemini_call", lambda *_: valid)
    result = generate_captions(settings, {
        "platform": "Instagram", "tone": "casual", "language": "English", "length": "short",
        "variants": 3, "idea": "test", "image_description": "", "emoji_enabled": True,
        "call_to_action": "", "brand_voice": "", "score_enabled": False,
    })
    assert result.provider == "Gemini"
    with pytest.raises(ProviderError):
        generate_captions(Settings(), {
            "platform": "Instagram", "tone": "casual", "language": "English", "length": "short",
            "variants": 3, "idea": "test", "image_description": "", "emoji_enabled": True,
            "call_to_action": "", "brand_voice": "", "score_enabled": False,
        })


def test_provider_rotates_models_from_original_prompt(monkeypatch):
    settings = Settings(
        groq_api_key="groq",
        groq_models=("unavailable-model", "working-model"),
        gemini_api_key="",
    )
    prompts = []

    def fake_groq(_settings, prompt, model=None):
        prompts.append((model, prompt))
        if model == "unavailable-model":
            raise ProviderError("model not found")
        return '{"captions":[{"text":"Fresh result","hashtags":[]}]}'

    monkeypatch.setattr("llm._groq_call", fake_groq)
    prompt_args = {
        "platform": "Instagram", "tone": "casual", "language": "English", "length": "short",
        "variants": 3, "idea": "original idea", "image_description": "", "emoji_enabled": True,
        "call_to_action": "", "brand_voice": "", "score_enabled": False,
    }
    result = generate_captions(settings, prompt_args)
    assert result.model == "working-model"
    assert [model for model, _ in prompts] == ["unavailable-model", "working-model"]
    assert prompts[0][1] == prompts[1][1]


def test_provider_error_detail_masks_credentials():
    detail = _safe_provider_detail(RuntimeError("api_key=sk-abcdefghijklmnop"))
    assert "sk-abcdefghijklmnop" not in detail
    assert "[redacted]" in detail or "[redacted-key]" in detail


class FakeCollection:
    def __init__(self):
        self.items = {}

    def get(self, ids, include=None):
        return {"ids": [item_id for item_id in ids if item_id in self.items]}

    def add(self, ids, documents, embeddings, metadatas):
        self.items[ids[0]] = (documents[0], metadatas[0])

    def count(self):
        return len(self.items)

    def query(self, **kwargs):
        values = list(self.items.values())
        return {
            "documents": [[item[0] for item in values]],
            "metadatas": [[item[1] for item in values]],
            "distances": [[0.1 for _ in values]],
        }


def test_rag_add_retrieve_duplicate_and_format(monkeypatch, tmp_path):
    store = RagStore(tmp_path / "rag")
    collection = FakeCollection()
    monkeypatch.setattr(store, "_collection_or_raise", lambda: collection)
    monkeypatch.setattr(store, "_embed", lambda texts: [[0.1, 0.2] for _ in texts])
    metadata = {"platform": "Instagram", "tone": "casual", "language": "English"}
    assert store.add_example("Warm launch caption", metadata)
    assert not store.add_example("Warm launch caption", metadata)
    results = store.retrieve("launch", platform="Instagram", tone="casual", language="English")
    assert len(results) == 1
    assert "Reference 1" in format_retrieved_context(results)
    assert store.has_data()
