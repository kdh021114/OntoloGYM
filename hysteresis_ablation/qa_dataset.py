from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


def build_augmented_qa_dataset(config) -> dict[str, Any]:
    """Build the 40 text + 40 hysteresis-image QA dataset for the ablation."""
    dry_run = bool(getattr(config, "HYSTERESIS_QA_DRY_RUN", True))
    image_example_dir = Path(config.HYSTERESIS_QA_IMAGE_EXAMPLE_DIR)
    output_path = Path(config.HYSTERESIS_QA_OUTPUT_DATASET_PATH)
    summary_path = Path(config.HYSTERESIS_QA_SUMMARY_JSON)

    if bool(getattr(config, "HYSTERESIS_QA_GENERATE_IMAGE_EXAMPLES", False)):
        image_generation = _generate_image_examples(config, dry_run=dry_run)
    else:
        image_generation = {"status": "skipped"}

    base_dataset_path = _base_dataset_path(config)
    base_examples = _load_jsonl(base_dataset_path)
    image_examples = _load_examples_from_dir(image_example_dir, required_tag="hysteresis_figure")

    text_count = int(getattr(config, "HYSTERESIS_QA_REUSED_TEXT_COUNT", 40))
    image_count = int(getattr(config, "HYSTERESIS_QA_IMAGE_COUNT", 40))
    selected_text = _select_examples(
        base_examples,
        count=text_count,
        seed=int(getattr(config, "HYSTERESIS_QA_RANDOM_SEED", 42)),
        sample_randomly=bool(getattr(config, "HYSTERESIS_QA_SAMPLE_BASE_RANDOMLY", True)),
    )
    selected_images = _select_examples(
        image_examples,
        count=image_count,
        seed=int(getattr(config, "HYSTERESIS_QA_RANDOM_SEED", 42)) + 1,
        sample_randomly=False,
    )

    combined = []
    combined.extend(_mark_source(example, "reused_text_qa") for example in selected_text)
    combined.extend(_mark_source(example, "hysteresis_image_qa") for example in selected_images)

    summary = {
        "status": "dry_run" if dry_run else "completed",
        "base_dataset_path": str(base_dataset_path),
        "image_example_dir": str(image_example_dir),
        "output_dataset_path": str(output_path),
        "base_examples_available": len(base_examples),
        "image_examples_available": len(image_examples),
        "requested_text_examples": text_count,
        "requested_image_examples": image_count,
        "selected_text_examples": len(selected_text),
        "selected_image_examples": len(selected_images),
        "combined_examples": len(combined),
        "image_generation": image_generation,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path, combined)
    return summary


def _generate_image_examples(config, *, dry_run: bool) -> dict[str, Any]:
    from qa_extractor.extractor.annotator import (
        SingleAnnotator,
        DEFAULT_LLM_MODEL,
        DEFAULT_TEMPERATURE,
        prepare_parsed_json_inputs,
        reload_available_uuids,
        save_annotated_result,
    )

    from .assets import load_asset_manifest

    assets = load_asset_manifest(config.HYSTERESIS_ASSET_MANIFEST_JSONL)
    paper_ids = sorted({str(asset.get("paper_id") or "") for asset in assets if asset.get("image_path")})
    paper_ids = [paper_id for paper_id in paper_ids if paper_id]
    target_count = int(getattr(config, "HYSTERESIS_QA_IMAGE_COUNT", 40))
    if dry_run:
        return {
            "status": "dry_run",
            "paper_ids_with_assets": len(paper_ids),
            "target_image_examples": target_count,
        }
    if not paper_ids:
        return {"status": "skipped", "reason": "no hysteresis assets available"}

    prepare_parsed_json_inputs()
    reload_available_uuids()
    image_example_dir = Path(config.HYSTERESIS_QA_IMAGE_EXAMPLE_DIR)
    image_example_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    failed = 0
    max_failures = max(1, int(getattr(config, "QA_MAX_FAILURES", 8)))
    focus_hint = str(getattr(config, "HYSTERESIS_QA_FOCUS_HINT", "") or "")

    attempt = 0
    while generated < target_count and failed < max_failures:
        pid = paper_ids[attempt % len(paper_ids)]
        attempt += 1
        try:
            result = SingleAnnotator(
                pid=pid,
                model=DEFAULT_LLM_MODEL,
                temperature=DEFAULT_TEMPERATURE,
            ).annotate(
                log_dir=os.fspath(getattr(config, "QA_LOG_DIR")),
                explore_func="hysteresis_image",
                qa_focus="hysteresis_figure_image",
                qa_focus_hint=focus_hint,
                example_dir=os.fspath(image_example_dir),
            )
            if result is None:
                failed += 1
                continue
            save_annotated_result(result)
            generated += 1
            failed = 0
        except Exception as exc:
            print(f"[hysteresis/image_qa] Failed for {pid}. {exc}")
            failed += 1

    return {
        "status": "completed",
        "target_image_examples": target_count,
        "generated_image_examples": generated,
        "failed_streak": failed,
        "attempts": attempt,
    }


def _base_dataset_path(config) -> Path:
    configured = getattr(config, "HYSTERESIS_QA_BASE_DATASET_PATH", None)
    if configured:
        return Path(configured)
    return Path(config.QA_OUTPUT_DATASET_PATH)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(example, dict):
            examples.append(example)
    return examples


def _load_examples_from_dir(path: Path, *, required_tag: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    examples = []
    for example_path in sorted(path.glob("*.json")):
        try:
            example = json.loads(example_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tags = {str(tag) for tag in example.get("tags", [])}
        if required_tag and required_tag not in tags:
            continue
        examples.append(example)
    return examples


def _select_examples(
    examples: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    sample_randomly: bool,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if len(examples) <= count:
        return list(examples)
    if not sample_randomly:
        return list(examples[:count])
    rng = random.Random(seed)
    return rng.sample(examples, count)


def _mark_source(example: dict[str, Any], source: str) -> dict[str, Any]:
    copied = dict(example)
    tags = list(copied.get("tags", []))
    if source not in tags:
        tags.append(source)
    copied["tags"] = tags
    copied["hysteresis_ablation_source"] = source
    return copied


def _write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
