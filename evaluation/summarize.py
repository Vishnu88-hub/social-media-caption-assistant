"""Summarize manually rated evaluation results without claiming superiority."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def summarize(path: Path, answer_key_path: Path | None = None) -> dict[str, float]:
    answer_key_path = answer_key_path or path.with_name("answer_key.csv")
    with answer_key_path.open(encoding="utf-8", newline="") as handle:
        key = {row["item_id"]: row["prompt_version"] for row in csv.DictReader(handle)}
    totals: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scores = [row[key] for key in ("relevance_1_5", "creativity_1_5", "tone_match_1_5") if row.get(key)]
            if scores:
                totals[key[row["item_id"]]].extend(float(score) for score in scores)
    averages = {version: sum(values) / len(values) for version, values in totals.items() if values}
    _write_svg(averages, path.with_name("comparison.svg"))
    return averages


def _write_svg(averages: dict[str, float], output: Path) -> None:
    bars = []
    for index, (version, average) in enumerate(sorted(averages.items())):
        width = max(1, int(average * 50))
        y = 30 + index * 45
        bars.append(f'<text x="5" y="{y + 15}">{version}</text><rect x="100" y="{y}" width="{width}" height="22" fill="#4f46e5"/><text x="{105 + width}" y="{y + 15}">{average:.2f}</text>')
    output.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="500" height="120">' + "".join(bars) + "</svg>", encoding="utf-8")


if __name__ == "__main__":
    print(summarize(Path("evaluation/rating_sheet.csv")))
