from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None

from common.papers import IMAGE_EXTENSIONS, PaperInput, discover_paper_inputs
from common.usage_logging import log_openai_usage

from .prompts import HYSTERESIS_FIGURE_CLASSIFIER_PROMPT


logger = logging.getLogger(__name__)


DEFAULT_CAPTION_KEYWORDS = (
    "hysteresis",
    "hysteretic",
    "coercivity",
    "coercive",
    "remanence",
    "remanent",
    "demagnetization",
    "magnetization loop",
    "m-h",
    "m-h loop",
    "b-h",
    "j-h",
    "magnetic field",
    "sweeping",
    "sweep direction",
    "field dependence",
    "forc",
    "hysteron",
)


@dataclass(frozen=True)
class FigureCandidate:
    paper_id: str
    figure_index: int
    figure_label: str
    caption: str
    source_json_path: Path
    paper_dir: Path
    original_image_path: Path | None


def load_asset_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    assets = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            asset = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(asset, dict):
            assets.append(asset)
    return assets


def collect_hysteresis_assets(config) -> dict[str, Any]:
    """Select hysteresis-loop figures and copy their images/captions into an ablation folder."""
    output_dir = Path(config.HYSTERESIS_ASSET_OUTPUT_DIR)
    image_dir = Path(config.HYSTERESIS_ASSET_IMAGE_DIR)
    caption_dir = Path(config.HYSTERESIS_ASSET_CAPTION_DIR)
    manifest_path = Path(config.HYSTERESIS_ASSET_MANIFEST_JSONL)
    dry_run = bool(getattr(config, "HYSTERESIS_ASSET_DRY_RUN", True))

    source_dir = Path(config.HYSTERESIS_SOURCE_PAPER_DIR)
    papers = discover_paper_inputs(source_dir)
    candidates = []
    selected = []
    rejected = []

    max_figures = getattr(config, "HYSTERESIS_MAX_FIGURES_TO_SCAN", None)
    for paper in papers:
        for candidate in _iter_figure_candidates(paper):
            candidates.append(candidate)
            decision = _classify_candidate(candidate, config)
            record = _candidate_record(candidate, decision)
            if decision["selected"]:
                selected.append(record)
            else:
                rejected.append(record)
            if max_figures is not None and len(candidates) >= int(max_figures):
                break
        if max_figures is not None and len(candidates) >= int(max_figures):
            break

    copied_assets = []
    if not dry_run:
        image_dir.mkdir(parents=True, exist_ok=True)
        caption_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for record in selected:
                asset = _copy_asset(record, image_dir=image_dir, caption_dir=caption_dir)
                manifest.write(json.dumps(asset, ensure_ascii=False) + "\n")
                copied_assets.append(asset)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "dry_run" if dry_run else "completed",
        "source_dir": str(source_dir),
        "papers": len(papers),
        "figure_candidates": len(candidates),
        "selected_assets": len(selected),
        "copied_assets": len(copied_assets),
        "manifest_path": str(manifest_path),
        "image_dir": str(image_dir),
        "caption_dir": str(caption_dir),
        "selection": {
            "use_caption_heuristic": bool(getattr(config, "HYSTERESIS_USE_CAPTION_HEURISTIC", True)),
            "use_vision_classifier": bool(getattr(config, "HYSTERESIS_USE_VISION_CLASSIFIER", False)),
            "caption_score_threshold": int(getattr(config, "HYSTERESIS_CAPTION_SCORE_THRESHOLD", 2)),
        },
    }
    summary_path = output_dir / ("dry_run_summary.json" if dry_run else "run_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_review_file(output_dir / "selected_assets_preview.json", selected[:200])
    _write_review_file(output_dir / "rejected_assets_preview.json", rejected[:200])
    logger.info("Hysteresis asset collection %s: %s selected.", summary["status"], len(selected))
    return summary


def _iter_figure_candidates(paper: PaperInput) -> list[FigureCandidate]:
    try:
        raw_data = json.loads(paper.json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    candidates = []
    for index, figure in enumerate(_raw_figures(raw_data), start=1):
        if not isinstance(figure, dict):
            continue
        caption = _figure_caption(figure)
        if not caption:
            continue
        image_path = _resolve_figure_image_path(paper, figure, caption)
        label = _figure_label(caption, index)
        candidates.append(
            FigureCandidate(
                paper_id=paper.paper_id,
                figure_index=index,
                figure_label=label,
                caption=caption,
                source_json_path=paper.json_path,
                paper_dir=paper.paper_dir,
                original_image_path=image_path,
            )
        )
    return candidates


def _raw_figures(raw_data: dict[str, Any]) -> list[Any]:
    if isinstance(raw_data.get("figures"), list):
        return raw_data["figures"]
    nested = raw_data.get("info_from_mineru")
    if isinstance(nested, dict) and isinstance(nested.get("figures"), list):
        return nested["figures"]
    return []


def _figure_caption(figure: dict[str, Any]) -> str:
    return _clean(
        figure.get("figure_caption")
        or figure.get("caption")
        or figure.get("title")
        or ""
    )


def _figure_label(caption: str, index: int) -> str:
    match = re.search(r"\b(?:fig(?:ure)?\.?)\s*([sS]?\d+[A-Za-z]?)", caption, re.IGNORECASE)
    if match:
        return f"Figure {match.group(1)}"
    return f"Figure {index}"


def _resolve_figure_image_path(paper: PaperInput, figure: dict[str, Any], caption: str) -> Path | None:
    candidates: list[Path] = []
    for key in ("figure_filename", "figure_path", "image_path", "path", "filename", "file"):
        value = str(figure.get(key) or "").strip()
        if not value:
            continue
        raw_path = Path(value)
        candidates.append(raw_path)
        candidates.append(paper.paper_dir / raw_path)
        candidates.append(paper.json_path.parent / raw_path)
        if paper.figures_dir:
            candidates.append(paper.figures_dir / raw_path.name)

    existing = _first_existing_image(candidates)
    if existing:
        return existing

    caption_keys = _figure_index_candidates(caption)
    for image_path in paper.figure_paths:
        normalized_stem = image_path.stem.lower().replace("-", "_")
        if any(key in normalized_stem for key in caption_keys):
            return image_path
    return None


def _figure_index_candidates(caption: str) -> list[str]:
    matches = re.findall(r"(?:figure|fig\.?)\s*([sS]?\d+[A-Za-z]?)", caption or "", re.IGNORECASE)
    keys = []
    for match in matches:
        normalized = match.lower()
        keys.extend(
            [
                f"figure_{normalized}",
                f"figure{normalized}",
                f"fig_{normalized}",
                f"fig{normalized}",
            ]
        )
    return keys


def _first_existing_image(candidates: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except FileNotFoundError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file() and resolved.suffix.lower() in IMAGE_EXTENSIONS:
            return resolved
    return None


def _classify_candidate(candidate: FigureCandidate, config) -> dict[str, Any]:
    score, matched_keywords = _caption_score(
        candidate.caption,
        getattr(config, "HYSTERESIS_CAPTION_KEYWORDS", DEFAULT_CAPTION_KEYWORDS),
    )
    caption_selected = (
        bool(getattr(config, "HYSTERESIS_USE_CAPTION_HEURISTIC", True))
        and score >= int(getattr(config, "HYSTERESIS_CAPTION_SCORE_THRESHOLD", 2))
    )
    vision_result = None
    if bool(getattr(config, "HYSTERESIS_USE_VISION_CLASSIFIER", False)):
        if candidate.original_image_path is None:
            vision_result = {"is_hysteresis_loop": False, "confidence": 0.0, "reason": "missing image file"}
        else:
            vision_result = _classify_with_vision(candidate, config)

    vision_selected = False
    if isinstance(vision_result, dict):
        vision_selected = bool(vision_result.get("is_hysteresis_loop")) and float(vision_result.get("confidence") or 0.0) >= float(
            getattr(config, "HYSTERESIS_VISION_CONFIDENCE_THRESHOLD", 0.55)
        )

    require_image = bool(getattr(config, "HYSTERESIS_REQUIRE_IMAGE_FILE", True))
    has_required_image = candidate.original_image_path is not None or not require_image
    selected = has_required_image and (caption_selected or vision_selected)
    if bool(getattr(config, "HYSTERESIS_USE_VISION_CLASSIFIER", False)):
        selected = has_required_image and vision_selected
        if bool(getattr(config, "HYSTERESIS_ALLOW_CAPTION_FALLBACK_WITH_VISION", True)):
            selected = has_required_image and (vision_selected or caption_selected)

    return {
        "selected": selected,
        "caption_score": score,
        "matched_keywords": matched_keywords,
        "caption_selected": caption_selected,
        "vision": vision_result,
        "selection_method": _selection_method(caption_selected, vision_selected),
    }


def _caption_score(caption: str, keywords: tuple[str, ...] | list[str]) -> tuple[int, list[str]]:
    normalized = re.sub(r"\s+", " ", caption.lower())
    matched = []
    for keyword in keywords:
        keyword = str(keyword).lower().strip()
        if keyword and keyword in normalized:
            matched.append(keyword)
    return len(matched), matched


def _selection_method(caption_selected: bool, vision_selected: bool) -> str:
    if caption_selected and vision_selected:
        return "caption_heuristic+vision"
    if vision_selected:
        return "vision"
    if caption_selected:
        return "caption_heuristic"
    return "rejected"


def _classify_with_vision(candidate: FigureCandidate, config) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai is required when HYSTERESIS_USE_VISION_CLASSIFIER=True.")
    prompt = HYSTERESIS_FIGURE_CLASSIFIER_PROMPT.format(caption=candidate.caption)
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(candidate.original_image_path)},
        },
    ]
    options = {}
    temperature = getattr(config, "HYSTERESIS_CLASSIFIER_TEMPERATURE", 0.0)
    if temperature is not None:
        options["temperature"] = temperature
    max_completion_tokens = getattr(config, "HYSTERESIS_CLASSIFIER_MAX_COMPLETION_TOKENS", 400)
    if max_completion_tokens is not None:
        options["max_completion_tokens"] = max_completion_tokens

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    last_error = None
    for attempt in range(1, 5):
        try:
            completion = client.chat.completions.create(
                model=getattr(config, "HYSTERESIS_CLASSIFIER_MODEL", "gpt-5.4-mini"),
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                **options,
            )
            log_openai_usage(completion, component="hysteresis_asset_classifier")
            return _loads_json_object(completion.choices[0].message.content or "{}")
        except Exception as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(3 * attempt)
    raise last_error


def _candidate_record(candidate: FigureCandidate, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": _asset_id(candidate),
        "paper_id": candidate.paper_id,
        "figure_index": candidate.figure_index,
        "figure_label": candidate.figure_label,
        "caption": candidate.caption,
        "source_json_path": str(candidate.source_json_path),
        "paper_dir": str(candidate.paper_dir),
        "original_image_path": str(candidate.original_image_path) if candidate.original_image_path else "",
        **decision,
    }


def _copy_asset(record: dict[str, Any], *, image_dir: Path, caption_dir: Path) -> dict[str, Any]:
    asset_id = record["asset_id"]
    image_path = ""
    original_image_path = Path(record["original_image_path"]) if record.get("original_image_path") else None
    if original_image_path and original_image_path.exists():
        target_image = image_dir / f"{asset_id}{original_image_path.suffix.lower()}"
        shutil.copy2(original_image_path, target_image)
        image_path = str(target_image)

    caption_path = caption_dir / f"{asset_id}.txt"
    caption_path.write_text(
        "\n".join(
            [
                f"Paper ID: {record['paper_id']}",
                f"Figure: {record['figure_label']}",
                "",
                "Caption:",
                record["caption"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    asset = dict(record)
    asset["image_path"] = image_path
    asset["caption_path"] = str(caption_path)
    return asset


def _asset_id(candidate: FigureCandidate) -> str:
    base = f"{candidate.paper_id}_{candidate.figure_label}"
    digest = hashlib.sha1(candidate.caption.encode("utf-8")).hexdigest()[:8]
    return f"{_slug(base)}_{digest}"


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:80] or "figure"


def _image_data_url(path: Path | None) -> str:
    if path is None:
        raise FileNotFoundError("No image path was provided.")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


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
    except json.JSONDecodeError:
        return {"is_hysteresis_loop": False, "confidence": 0.0, "reason": "invalid JSON response"}
    return data if isinstance(data, dict) else {}


def _write_review_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
