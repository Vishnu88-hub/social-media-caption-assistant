"""Streamlit user interface for the Social Media Caption Assistant."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from config import Settings, database_path
from db import add_brand_voice, delete_brand_voice, delete_caption, list_brand_voice, list_captions, save_caption, set_favourite
from image_utils import ImageValidationError, process_image
from llm import ProviderError, describe_image, generate_captions, refine_caption
from prompts import LANGUAGES, LENGTHS, TONES
from rag import RagError, RagStore, format_retrieved_context

st.set_page_config(page_title="Social Media Caption Assistant", page_icon="✍️", layout="wide")
settings = Settings.load()
db_path = database_path(settings)


def _init_state() -> None:
    defaults = {
        "results": [], "provider": "", "model": "", "requests": 0,
        "last_generation": 0.0, "image_description": "", "retrieved_examples": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _preferences() -> dict:
    with st.sidebar:
        st.header("Preferences")
        platform = st.selectbox("Platform", ["Instagram", "LinkedIn", "X (Twitter)", "YouTube Shorts"])
        tone = st.selectbox("Tone", list(TONES), index=list(TONES).index("casual"))
        language = st.selectbox("Language", list(LANGUAGES))
        length = st.selectbox("Caption length", list(LENGTHS), index=1)
        variants = st.slider("Caption variants", 3, 5, 3)
        emoji_enabled = st.toggle("Use emojis", value=True)
        score_enabled = st.toggle("AI quality scoring", value=False)
        rag_enabled = st.toggle("Use approved examples (RAG)", value=False)
        call_to_action = st.text_input("Optional call to action", max_chars=200)
        brand_voice = st.text_area("Brand voice sample", max_chars=1000, height=80)
        st.divider()
        st.caption("Provider status")
        for provider, status in settings.safe_status().items():
            st.write(f"**{provider}:** {status}")
        with st.expander("Privacy and settings"):
            st.write("Images are processed in memory and are not stored. Text and images sent to enabled providers are subject to their terms and free-tier data policies.")
            st.write(f"Session requests: {st.session_state.requests}/{settings.session_request_limit}")
    return locals()


@st.cache_resource
def _rag_store(path: str, model_name: str) -> RagStore:
    return RagStore(path, model_name)


def _render_results(prefs: dict, idea: str, rag_store: RagStore | None) -> None:
    if not st.session_state.results:
        return
    st.subheader("Generated captions")
    for index, caption in enumerate(st.session_state.results):
        with st.container(border=True):
            st.markdown(f"**Option {index + 1}**")
            if prefs["platform"] == "YouTube Shorts":
                st.text_input("Title", caption.get("title", ""), key=f"title_{index}", disabled=True)
            st.code(caption["text"], language=None)
            st.caption(" ".join(caption.get("hashtags", [])))
            st.caption(f'{caption.get("character_count", len(caption["text"]))} characters / {caption.get("character_limit", "—")} limit')
            if caption.get("score") is not None:
                st.caption(f"AI quality estimate: {caption['score']}/10 — {caption.get('reason', '')}")
            cols = st.columns(6)
            with cols[0]:
                if st.button("Save", key=f"save_{index}"):
                    save_caption(db_path, caption, platform=prefs["platform"], tone=prefs["tone"], language=prefs["language"], idea=idea)
                    st.success("Saved to history.")
            with cols[1]:
                if st.button("Shorter", key=f"shorter_{index}"):
                    _refine(caption, "Make shorter", prefs)
            with cols[2]:
                if st.button("Funnier", key=f"funnier_{index}"):
                    _refine(caption, "Make funnier", prefs)
            with cols[3]:
                if st.button("Professional", key=f"professional_{index}"):
                    _refine(caption, "Make more professional", prefs)
            with cols[4]:
                if st.button("Different angle", key=f"angle_{index}"):
                    _refine(caption, "Try a different angle", prefs)
            with cols[5]:
                if st.button("Add CTA", key=f"cta_{index}"):
                    _refine(caption, "Add a call to action", prefs)
            if rag_store is not None and st.button("Approve as RAG example", key=f"rag_{index}"):
                try:
                    added = rag_store.add_caption(
                        caption, platform=prefs["platform"], tone=prefs["tone"],
                        language=prefs["language"], length=prefs["length"],
                    )
                    st.success("Added to approved examples." if added else "This example is already stored.")
                except (RagError, ValueError) as exc:
                    st.error(f"Could not add RAG example: {exc}")


def _refine(caption: dict, action: str, prefs: dict) -> None:
    try:
        result = refine_caption(settings, caption, action=action, platform=prefs["platform"], tone=prefs["tone"], language=prefs["language"], emoji_enabled=prefs["emoji_enabled"])
        position = st.session_state.results.index(caption)
        st.session_state.results[position] = result.captions[0]
        st.session_state.provider, st.session_state.model = result.provider, result.model
        st.rerun()
    except ProviderError as exc:
        st.error(str(exc))


def _history() -> None:
    st.subheader("History")
    favourites = st.checkbox("Show favourites only")
    rows = list_captions(db_path, favourites_only=favourites)
    if not rows:
        st.info("No saved captions yet.")
        return
    for row in rows[:30]:
        cols = st.columns([5, 2, 1, 1])
        cols[0].write(f"**{row['platform']}** — {row['text']}\n\n{row['hashtags']}")
        cols[1].caption(row["created_at"])
        if cols[2].button("★" if row["favourite"] else "☆", key=f"fav_{row['id']}"):
            set_favourite(db_path, row["id"], not bool(row["favourite"]))
            st.rerun()
        if cols[3].button("Delete", key=f"delete_{row['id']}"):
            delete_caption(db_path, row["id"])
            st.rerun()


def _brand_voice(rag_store: RagStore | None) -> None:
    with st.expander("Manage brand voice"):
        sample = st.text_area("Add a sample", key="new_brand_voice", max_chars=1000)
        if st.button("Add sample"):
            try:
                add_brand_voice(db_path, sample)
                if rag_store is not None:
                    rag_store.add_brand_guideline(sample)
                st.success("Brand voice sample added.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        for item in list_brand_voice(db_path):
            cols = st.columns([8, 1])
            cols[0].write(item["sample"])
            if cols[1].button("Delete", key=f"brand_delete_{item['id']}"):
                delete_brand_voice(db_path, item["id"])
                st.rerun()


def main() -> None:
    _init_state()
    st.title("Social Media Caption Assistant")
    st.write("Generate platform-aware captions, hashtags, and optional emoji suggestions from an idea or an image.")
    prefs = _preferences()
    rag_store = _rag_store(settings.rag_path, settings.embedding_model) if prefs["rag_enabled"] else None
    image_bytes: bytes | None = None
    idea = ""
    tab_image, tab_idea = st.tabs(["Upload image", "Type an idea"])
    with tab_image:
        upload = st.file_uploader("JPG, PNG, or WEBP (max 10 MB)", type=["jpg", "jpeg", "png", "webp"])
        if upload:
            try:
                image_bytes = process_image(upload.getvalue(), max_bytes=settings.max_image_bytes)
                st.image(image_bytes, use_container_width=True)
            except ImageValidationError as exc:
                st.error(str(exc))
            st.session_state.image_description = st.text_area("Image description (edit if vision is unavailable)", value=st.session_state.image_description)
    with tab_idea:
        idea = st.text_area("What should the caption be about?", max_chars=settings.max_idea_chars, height=120)
    if st.session_state.provider:
        st.caption(f"Last generation: {st.session_state.provider} / {st.session_state.model}")
    ready = bool(settings.ai_configured) and bool(image_bytes or idea.strip() or st.session_state.image_description.strip())
    if not settings.ai_configured:
        st.warning("Configure GROQ_API_KEY or GEMINI_API_KEY in .env or Streamlit secrets to enable AI generation.")
    if st.button("Generate captions", type="primary", disabled=not ready):
        if st.session_state.requests >= settings.session_request_limit:
            st.error("This session has reached its request limit.")
        elif time.monotonic() - st.session_state.last_generation < settings.generation_cooldown_seconds:
            st.warning("Please wait briefly before generating again.")
        else:
            with st.spinner("Generating captions..."):
                try:
                    if image_bytes and not st.session_state.image_description:
                        try:
                            st.session_state.image_description = describe_image(settings, image_bytes)
                        except ProviderError:
                            st.warning("Image understanding failed. Use the manual description to continue.")
                    prompt_args = {
                        "platform": prefs["platform"], "tone": prefs["tone"], "language": prefs["language"],
                        "length": prefs["length"], "variants": prefs["variants"], "idea": idea,
                        "image_description": st.session_state.image_description, "emoji_enabled": prefs["emoji_enabled"],
                        "call_to_action": prefs["call_to_action"], "brand_voice": prefs["brand_voice"],
                        "score_enabled": prefs["score_enabled"],
                    }
                    st.session_state.retrieved_examples = []
                    if rag_store is not None:
                        retrieval_query = " ".join(
                            part for part in (idea, st.session_state.image_description, prefs["brand_voice"]) if part
                        )
                        try:
                            st.session_state.retrieved_examples = rag_store.retrieve(
                                retrieval_query, platform=prefs["platform"], tone=prefs["tone"],
                                language=prefs["language"],
                            )
                            prompt_args["retrieved_context"] = format_retrieved_context(
                                st.session_state.retrieved_examples
                            )
                        except RagError as exc:
                            st.warning(f"RAG is unavailable; continuing without retrieved examples: {exc}")
                    result = generate_captions(settings, prompt_args, image_bytes=image_bytes)
                    st.session_state.results = result.captions
                    st.session_state.provider, st.session_state.model = result.provider, result.model
                    st.session_state.requests += 1
                    st.session_state.last_generation = time.monotonic()
                    st.rerun()
                except (ProviderError, ValueError) as exc:
                    st.error(str(exc))
                    st.caption("Check the provider key, configured model name, quota, and network connection. No credentials are shown here.")
    if prefs["rag_enabled"]:
        if st.session_state.retrieved_examples:
            st.info(f"Retrieved {len(st.session_state.retrieved_examples)} approved reference(s).")
            with st.expander("Inspect retrieved references"):
                for item in st.session_state.retrieved_examples:
                    st.write(item["text"])
        else:
            st.caption("No approved examples matched this request yet.")
    _render_results(prefs, idea, rag_store)
    _history()
    _brand_voice(rag_store)


if __name__ == "__main__":
    main()
