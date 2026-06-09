from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_airqa_dataset(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []

    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def question_paper_ids(example: dict[str, Any], include_reference: bool = False) -> set[str]:
    ids = []
    keys = ["anchor_pdf"]
    if include_reference:
        keys.append("reference_pdf")
    for key in keys:
        value = example.get(key, [])
        if isinstance(value, str):
            ids.append(value)
        elif isinstance(value, list):
            ids.extend(item for item in value if isinstance(item, str))
    return set(ids)


def load_baseline_answers(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    answers = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            uuid = data.get("uuid")
            answer = data.get("answer")
            if uuid and answer is not None:
                answers[str(uuid)] = str(answer)
    return answers
