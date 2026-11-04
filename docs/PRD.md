# Social Media Caption Assistant PRD

The application accepts an image or a 500-character idea, applies platform, tone, language, length, emoji, call-to-action, and brand-voice preferences, and generates 3–5 structured caption options. Groq is the primary text provider and Gemini is used for vision and fallback text generation.

Images are validated, orientation-corrected, resized, compressed in memory, and never persisted. SQLite stores only explicitly saved caption metadata, favourites, and brand-voice samples. The app can run without keys in a configuration-status mode.

Platform constraints are represented in `prompts.py` and enforced again in `postprocess.py`. AI scoring is optional and is labelled as an estimate. Provider failures are surfaced to the user and unit tests mock provider boundaries.
