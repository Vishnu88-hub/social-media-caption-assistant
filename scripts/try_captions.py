"""Manual caption smoke test. Requires configured provider credentials."""

from config import Settings
from llm import generate_captions

result = generate_captions(Settings.load(), {
    "platform": "Instagram", "tone": "casual", "language": "English", "length": "medium",
    "variants": 3, "idea": "A sunrise walk by the beach", "image_description": "",
    "emoji_enabled": True, "call_to_action": "", "brand_voice": "", "score_enabled": False,
})
for caption in result.captions:
    print(caption)
