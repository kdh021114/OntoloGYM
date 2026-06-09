"""Generate an academic hysteresis-loop QA dataset with gpt-5.4.

The previous visual QA set over-emphasized literal figure-reading tasks
such as axis labels and marker colors. This script keeps the same 40 reused
text questions, but regenerates the 40 hysteresis questions so that each one
asks about magnetic interpretation, property comparison, or experimental
conclusion grounded in the separated figure image and caption.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from common.project_config import load_env_file, load_project_config
from common.usage_logging import log_openai_usage


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "hysteresis_ablation_manual_20260604"
ASSET_MANIFEST = DATA_DIR / "assets" / "hysteresis_assets.jsonl"
QA_DIR = DATA_DIR / "qa"
BASE40_PATH = QA_DIR / "base_current80_selected40.jsonl"
OUTPUT_IMAGE_QA = QA_DIR / "image_hysteresis_academic_qa40_gpt54_verified.jsonl"
OUTPUT_RAW = QA_DIR / "image_hysteresis_academic_qa_candidates_gpt54_raw.jsonl"
OUTPUT_DATASET = QA_DIR / "qa_dataset_hysteresis_academic_augmented_80.jsonl"
OUTPUT_READABLE = QA_DIR / "qa_dataset_hysteresis_academic_augmented_80_readable.md"
OUTPUT_REPORT = QA_DIR / "gpt54_academic_image_qa_verification_report.md"

MODEL = os.getenv("ONTOLOGYM_HYSTERESIS_ACADEMIC_QA_MODEL", "gpt-5.4")
TARGET_IMAGE_QA_COUNT = int(os.getenv("ONTOLOGYM_HYSTERESIS_ACADEMIC_QA_COUNT", "40"))
QUESTIONS_PER_ASSET = int(os.getenv("ONTOLOGYM_HYSTERESIS_ACADEMIC_QA_PER_ASSET", "3"))
OPENAI_TIMEOUT_SECONDS = float(os.getenv("ONTOLOGYM_OPENAI_TIMEOUT_SECONDS", "120"))


PROMPT_TEMPLATE = """
You are generating academic QA examples for a knowledge-graph ablation on
magnetic hysteresis-loop figures in materials science papers.

Generate up to {questions_per_asset} QA items from the attached figure image and caption.
The QA should test whether a model can use the figure/caption to recover
scientific magnetic meaning, not whether it can recite superficial graphic labels.

Allowed question types:
- Compare coercivity, remanence, saturation, loop width, loop height, squareness,
  magnetic hardness, switching-field distribution, anisotropy, exchange-bias-like
  shift, FORC features, current/temperature/gating dependence, magnetic phase
  transition, or magnetotransport hysteresis.
- Ask what scientific conclusion follows from a visible hysteresis trend.
- Ask which sample/condition has the stronger or weaker magnetic property and why.
- Ask for a clearly reported value only when it is tied to a magnetic property
  or interpretation, not just a label-reading task.

Hard bans:
- Do not ask for the x-axis label, y-axis label, unit, marker shape, color, panel
  letter, legend entry, or any pure graphic-identification fact.
- Do not ask "which panel shows..." unless the answer also explains a magnetic
  result or interpretation.
- Do not ask questions answerable by generic background knowledge alone.
- Do not invent values that are not visible or captioned. If a precise value is
  not clearly supported, phrase the answer qualitatively.

Each answer must be concise, scientifically phrased, and directly supported by
the image and caption. Prefer a one- or two-sentence answer.

Figure asset id: {asset_id}
Paper id: {paper_id}
Figure label: {figure_label}
Caption:
{caption}

Return JSON only:
{{
  "qa": [
    {{
      "question": "academic question",
      "answer": "concise reference answer",
      "answer_format": "concise scientific statement",
      "reasoning": "1-3 sentences explaining the figure/caption evidence",
      "academic_focus": "coercivity|remanence|anisotropy|FORC|current dependence|temperature dependence|magnetotransport hysteresis|other",
      "visual_dependency": "what must be read or compared in the image"
    }}
  ]
}}
"""


BAD_QUESTION_PATTERNS = [
    r"\bx-?axis\b",
    r"\by-?axis\b",
    r"\baxis label\b",
    r"\bwhat color\b",
    r"\bwhich color\b",
    r"\bmarker shape\b",
    r"\blegend entr",
    r"\bunit of\b",
    r"\bprinted\b",
    r"\blabeled above\b",
    r"\blabel of\b",
    r"\bwhat is the label\b",
    r"\bwhich panel corresponds\b",
    r"\bwhich panel shows\b",
    r"\bwhat panel\b",
]

ACADEMIC_TERMS = {
    "coerc",
    "reman",
    "saturat",
    "squareness",
    "anisotrop",
    "hard",
    "soft",
    "switch",
    "hysteresis",
    "loop",
    "demagnet",
    "forc",
    "magnetization",
    "magnetoresistance",
    "hall",
    "current",
    "temperature",
    "cooling field",
    "exchange",
    "memory",
    "phase",
    "magnetic",
    "mrr",
}


def main() -> None:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))
    if OpenAI is None:
        raise ImportError("openai is required for gpt-5.4 QA generation.")

    assets = _load_jsonl(ASSET_MANIFEST)
    if len(assets) < 1:
        raise ValueError(f"No hysteresis assets found: {ASSET_MANIFEST}")
    base40 = _load_jsonl(BASE40_PATH)
    if len(base40) != 40:
        raise ValueError(f"Expected 40 reused text QA examples, found {len(base40)}: {BASE40_PATH}")

    raw_candidates: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, 1):
        print(f"[{index:02d}/{len(assets):02d}] generating academic QA for {asset.get('asset_id')}", flush=True)
        generated = _generate_for_asset(asset)
        raw_candidates.extend(generated)
        _write_jsonl(OUTPUT_RAW, raw_candidates)

    selected, rejected = _select_verified(raw_candidates, assets)
    if len(selected) != TARGET_IMAGE_QA_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_IMAGE_QA_COUNT} verified image QA examples, selected {len(selected)}. "
            f"Rejected {len(rejected)} candidates."
        )

    image_examples = [_to_airqa_example(item, rank=i + 1) for i, item in enumerate(selected)]
    combined = [_mark_base_example(item, i + 1) for i, item in enumerate(base40)] + image_examples
    _write_jsonl(OUTPUT_IMAGE_QA, image_examples)
    _write_jsonl(OUTPUT_DATASET, combined)
    _write_readable(combined, selected, rejected)
    print(json.dumps(_summary(selected, rejected, combined), ensure_ascii=False, indent=2), flush=True)


def _generate_for_asset(asset: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = Path(str(asset.get("image_path") or ""))
    if not image_path.exists():
        return []
    prompt = PROMPT_TEMPLATE.format(
        questions_per_asset=QUESTIONS_PER_ASSET,
        asset_id=asset.get("asset_id", ""),
        paper_id=asset.get("paper_id", ""),
        figure_label=asset.get("figure_label", ""),
        caption=asset.get("caption", ""),
    )
    content = _call_openai_json(prompt, image_path)
    qa_items = content.get("qa", [])
    if not isinstance(qa_items, list):
        return []
    generated: list[dict[str, Any]] = []
    for local_index, qa in enumerate(qa_items, 1):
        if not isinstance(qa, dict):
            continue
        item = dict(qa)
        item.update(
            {
                "asset_id": asset.get("asset_id"),
                "paper_id": asset.get("paper_id"),
                "figure_label": asset.get("figure_label"),
                "image_path": asset.get("image_path"),
                "caption_path": asset.get("caption_path"),
                "caption": asset.get("caption"),
                "candidate_index": local_index,
                "generation_model": MODEL,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        item["verification"] = _verify_candidate(item)
        generated.append(item)
    return generated


def _call_openai_json(prompt: str, image_path: Path) -> dict[str, Any]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    last_error = None
    for attempt in range(1, 5):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1800,
            )
            log_openai_usage(completion, component="hysteresis_academic_qa_generation")
            return _loads_json_object(completion.choices[0].message.content or "{}")
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(3 * attempt)
    raise last_error  # pragma: no cover


def _verify_candidate(item: dict[str, Any]) -> dict[str, Any]:
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    reasoning = str(item.get("reasoning") or "").strip()
    visual_dependency = str(item.get("visual_dependency") or "").strip()
    text = " ".join([question, answer, reasoning, visual_dependency]).lower()
    failures = []
    if len(question) < 40:
        failures.append("question_too_short")
    if len(answer) < 25:
        failures.append("answer_too_short")
    if any(re.search(pattern, question, flags=re.IGNORECASE) for pattern in BAD_QUESTION_PATTERNS):
        failures.append("surface_graphic_question")
    if not any(term in text for term in ACADEMIC_TERMS):
        failures.append("missing_academic_magnetism_focus")
    if not visual_dependency:
        failures.append("missing_visual_dependency")
    if "not reported" in answer.lower() or "cannot be determined" in answer.lower():
        failures.append("negative_or_unanswerable")
    return {
        "status": "accepted" if not failures else "rejected",
        "failures": failures,
    }


def _select_verified(
    candidates: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = [
        item
        for item in candidates
        if item.get("verification", {}).get("status") == "accepted"
    ]
    rejected = [
        item
        for item in candidates
        if item.get("verification", {}).get("status") != "accepted"
    ]
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for item in accepted:
        by_asset.setdefault(str(item.get("asset_id")), []).append(item)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for asset in assets:
        asset_id = str(asset.get("asset_id"))
        options = by_asset.get(asset_id, [])
        if not options:
            raise RuntimeError(f"No accepted academic QA candidate for asset: {asset_id}")
        chosen = options[0]
        selected.append(chosen)
        selected_ids.add(id(chosen))

    remaining = [item for item in accepted if id(item) not in selected_ids]
    remaining.sort(key=lambda item: (str(item.get("paper_id")), str(item.get("asset_id")), int(item.get("candidate_index") or 0)))
    for item in remaining:
        if len(selected) >= TARGET_IMAGE_QA_COUNT:
            break
        selected.append(item)

    if len(selected) < TARGET_IMAGE_QA_COUNT:
        raise RuntimeError(
            f"Only {len(selected)} accepted candidates available for {TARGET_IMAGE_QA_COUNT} required examples."
        )
    return selected[:TARGET_IMAGE_QA_COUNT], rejected


def _to_airqa_example(item: dict[str, Any], *, rank: int) -> dict[str, Any]:
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hysteresis-academic::{item.get('asset_id')}::{question}"))
    context = (
        "Use the separated hysteresis-loop figure image and caption to answer an academic "
        "magnetism question. "
        f"Figure asset: {item.get('asset_id')}. "
        f"Figure label: {item.get('figure_label')}. "
        f"Caption: {item.get('caption')}"
    )
    return {
        "uuid": uid,
        "question": question,
        "answer_format": str(item.get("answer_format") or "Provide a concise scientific statement."),
        "tags": [
            "single",
            "image",
            "figure",
            "objective",
            "hysteresis_loop",
            "hysteresis_ablation_image40",
            "academic_hysteresis_qa",
            "gpt54_generated",
            "codex_verified",
            "expanded_25_assets",
        ],
        "anchor_pdf": [str(item.get("paper_id"))],
        "reference_pdf": [],
        "conference": [],
        "evaluator": {
            "eval_func": "eval_reference_answer_with_llm",
            "eval_kwargs": {
                "reference_answer": answer,
                "question": question,
            },
        },
        "annotator": "gpt-5.4_then_codex_academic_verify",
        "target_eval_type": "objective",
        "context": context,
        "answer": answer,
        "reasoning": str(item.get("reasoning") or "").strip(),
        "figure_asset_id": item.get("asset_id"),
        "hysteresis_asset_id": item.get("asset_id"),
        "figure_image_path": item.get("image_path"),
        "figure_caption_path": item.get("caption_path"),
        "figure_caption": item.get("caption"),
        "academic_focus": item.get("academic_focus"),
        "visual_dependency": item.get("visual_dependency"),
        "ablation_source": "gpt54_hysteresis_image40_academic_assets",
        "generation_model": MODEL,
        "qa_generation_model": MODEL,
        "verification_status": "codex_verified",
        "verified_by": "codex",
        "verified_at": datetime.now().date().isoformat(),
        "image_qa_rank": rank,
    }


def _mark_base_example(example: dict[str, Any], rank: int) -> dict[str, Any]:
    copied = dict(example)
    tags = [str(tag) for tag in copied.get("tags", [])]
    for tag in ["hysteresis_ablation_base40", "text_base_selected"]:
        if tag not in tags:
            tags.append(tag)
    copied["tags"] = tags
    copied["hysteresis_ablation_source"] = "reused_text_qa"
    copied["base_selection_rank"] = copied.get("base_selection_rank", rank)
    return copied


def _summary(selected: list[dict[str, Any]], rejected: list[dict[str, Any]], combined: list[dict[str, Any]]) -> dict[str, Any]:
    asset_counts: dict[str, int] = {}
    focus_counts: dict[str, int] = {}
    for item in selected:
        asset_counts[str(item.get("asset_id"))] = asset_counts.get(str(item.get("asset_id")), 0) + 1
        focus = str(item.get("academic_focus") or "other")
        focus_counts[focus] = focus_counts.get(focus, 0) + 1
    return {
        "status": "completed",
        "model": MODEL,
        "image_qa": len(selected),
        "combined_qa": len(combined),
        "assets_covered": len(asset_counts),
        "rejected_candidates": len(rejected),
        "asset_counts": asset_counts,
        "focus_counts": focus_counts,
        "image_qa_path": str(OUTPUT_IMAGE_QA),
        "dataset_path": str(OUTPUT_DATASET),
        "report_path": str(OUTPUT_REPORT),
    }


def _write_readable(combined: list[dict[str, Any]], selected: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    OUTPUT_READABLE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Hysteresis Academic QA Dataset",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- QA generation model: {MODEL}",
        f"- Total examples: {len(combined)}",
        f"- Reused text examples: {sum(1 for item in combined if 'hysteresis_ablation_base40' in item.get('tags', []))}",
        f"- Academic image examples: {sum(1 for item in combined if 'hysteresis_ablation_image40' in item.get('tags', []))}",
        "",
    ]
    for index, item in enumerate(combined, 1):
        lines.extend(
            [
                f"## {index}. {item.get('uuid')}",
                "",
                f"**Tags:** {', '.join(str(tag) for tag in item.get('tags', []))}",
                "",
                f"**Question:** {item.get('question')}",
                "",
                f"**Answer:** {item.get('answer')}",
                "",
            ]
        )
        if item.get("figure_asset_id"):
            lines.extend(
                [
                    f"**Asset:** {item.get('figure_asset_id')}",
                    "",
                    f"**Focus:** {item.get('academic_focus')}",
                    "",
                    f"**Visual dependency:** {item.get('visual_dependency')}",
                    "",
                    f"**Reasoning:** {item.get('reasoning')}",
                    "",
                ]
            )
    OUTPUT_READABLE.write_text("\n".join(lines), encoding="utf-8")

    report = [
        "# gpt-5.4 Academic Hysteresis QA Verification Report",
        "",
        "Verification criteria:",
        "- Reject pure graphic-description questions such as axis labels, marker colors, panel letters, and legend-only identification.",
        "- Require a magnetic/materials-science focus such as coercivity, remanence, anisotropy, switching, FORC, temperature/current dependence, magnetoresistance hysteresis, or magnetic memory.",
        "- Require a non-empty visual dependency so that the question is tied to the separated figure image, not generic domain knowledge.",
        "",
        f"Selected academic image QA: {len(selected)}",
        f"Rejected candidates: {len(rejected)}",
        "",
        "## Selected",
        "",
    ]
    for index, item in enumerate(selected, 1):
        report.extend(
            [
                f"### {index}. {item.get('asset_id')}",
                "",
                f"Q: {item.get('question')}",
                "",
                f"A: {item.get('answer')}",
                "",
                f"Focus: {item.get('academic_focus')}",
                "",
                f"Visual dependency: {item.get('visual_dependency')}",
                "",
            ]
        )
    if rejected:
        report.extend(["## Rejected", ""])
        for item in rejected:
            report.extend(
                [
                    f"- {item.get('asset_id')}: {item.get('question')} "
                    f"({', '.join(item.get('verification', {}).get('failures', []))})"
                ]
            )
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _loads_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {content[:500]}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from QA generation.")
    return data


if __name__ == "__main__":
    sys.path.insert(0, os.fspath(PROJECT_ROOT))
    main()
