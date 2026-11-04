# Project coding instructions

- Keep UI code in `app.py`; keep provider, prompt, image, post-processing, and persistence logic in their dedicated modules.
- Never log credentials, image bytes, raw provider responses, or private user captions.
- Keep provider calls mockable and never require credentials in unit tests.
- Use parameterized SQLite queries and context-managed connections.
- Preserve Unicode text and enforce platform limits after generation.
- Prefer explicit, user-visible errors over silent fallbacks.
