# Social Media Caption Assistant

An AI-powered Streamlit app for platform-specific captions, hashtags, and optional emoji suggestions from an idea or uploaded image.

## Features

- Instagram, LinkedIn, X, and YouTube Shorts rules with Unicode-safe limits.
- Groq-first generation with Gemini vision/text fallback.
- In-memory image validation and preprocessing; images are not stored.
- Caption history, favourites, deletion, and brand-voice samples in SQLite.
- Optional AI quality estimates and refinement actions.
- Provider-free startup with visible configuration status.
- Optional local RAG using ChromaDB and Sentence Transformers. Only explicitly approved examples enter the knowledge base.

## Optional RAG knowledge base

When **Use approved examples (RAG)** is enabled, the app embeds the current idea/image description and retrieves relevant approved caption examples using platform, tone, and language metadata. The retrieved text is added to the existing prompt as style guidance; it is not copied verbatim. If RAG is disabled, empty, or unavailable, the original generation pipeline runs unchanged.

Click **Approve as RAG example** on a generated caption to add it explicitly. Brand-voice samples can still be managed independently in SQLite. The persistent local ChromaDB directory is `rag_data/` by default and is excluded from Git.

RAG configuration is optional:

```text
RAG_PATH=rag_data
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

The first enabled RAG request may download the embedding model locally. If model download, ChromaDB initialization, or retrieval fails, the app reports the issue and continues without RAG. To troubleshoot, check network access for the first model download, confirm the `rag_data` directory is writable, and run `pytest -q`.

## Install and run on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
```

Fill these fields in `.env` yourself:

```text
GEMINI_API_KEY=
GROQ_API_KEY=
GEMINI_MODEL=gemini-3.8-flash
GROQ_MODEL=llama-3.1-8b-instant
RAG_PATH=rag_data
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Never commit `.env`, secrets, uploaded images, or `captions.db`.

## Tests

```powershell
pytest -q
```

Tests do not call either API and do not require keys. Manual smoke tests are available in `scripts/try_vision.py` and `scripts/try_captions.py` after configuring credentials.

## Evaluation

```powershell
python evaluation\run_evaluation.py
python evaluation\summarize.py
```

The evaluation sheet is blind and requires manual 1–5 ratings. The tooling does not claim one prompt is better without measured results.

## Deployment

Push this folder to a GitHub repository, select `app.py` in Streamlit Community Cloud, and add the values from `.streamlit/secrets.toml.example` through the deployment Secrets settings. Use a persistent external database only if requirements change; this project intentionally uses local SQLite.

## Privacy and limitations

When enabled, image and text content is sent to the selected third-party AI provider. Review current provider free-tier data policies before use. Provider model names and response schemas can change, so run the manual smoke tests after changing models. No engagement prediction is provided; quality scores are AI estimates.
