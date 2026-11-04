"""Create a blind prompt-comparison CSV from deterministic sample inputs.

Actual provider calls are intentionally opt-in and belong in a local experiment.
"""

from __future__ import annotations

import csv
from pathlib import Path

SAMPLES = ("food", "travel", "selfies", "products", "events")


def create_rating_sheet(output: Path) -> None:
    answer_key = output.with_name("answer_key.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "category", "platform", "caption", "hashtags", "length", "limit_compliance", "relevance_1_5", "creativity_1_5", "tone_match_1_5"])
        writer.writeheader()
        key_rows = []
        item_id = 1
        for category in SAMPLES:
            for version in ("generic", "engineered"):
                writer.writerow({"item_id": item_id, "category": category, "platform": "Instagram", "caption": "", "hashtags": "", "length": "", "limit_compliance": "", "relevance_1_5": "", "creativity_1_5": "", "tone_match_1_5": ""})
                key_rows.append({"item_id": item_id, "prompt_version": version})
                item_id += 1
    with answer_key.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "prompt_version"])
        writer.writeheader()
        writer.writerows(key_rows)


if __name__ == "__main__":
    create_rating_sheet(Path("evaluation/rating_sheet.csv"))
