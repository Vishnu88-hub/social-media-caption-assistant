"""Manual vision smoke test. Requires GEMINI_API_KEY and an image path."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from image_utils import process_image
from llm import describe_image

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()
print(describe_image(Settings.load(), process_image(args.image.read_bytes())))
