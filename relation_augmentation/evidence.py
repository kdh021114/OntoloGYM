from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hysteresis_ablation.assets import load_asset_manifest
from hysteresis_ablation.prompts import HYSTERESIS_LOOP_CONTEXT


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    paper_id: str
    source_path: str
    source_type: str
    section: str
    title: str
    page_number: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None

    def source_metadata(self) -> dict[str, Any]:
        metadata = {
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "section": self.section,
            "title": self.title,
            "page_number": self.page_number,
        }
        if self.image_path:
            metadata["image_path"] = self.image_path
        if self.metadata:
            metadata.update(
                {
                    key: value
                    for key, value in self.metadata.items()
                    if key != "prompt_context"
                }
            )
        return metadata


def resolve_input_files(config) -> list[Path]:
    configured = getattr(config, "RELATION_INPUT_FILES", [])
    if configured:
        return [Path(path) for path in configured]
    input_dir = Path(config.RELATION_PROCESSED_DATA_DIR)
    return sorted(input_dir.glob("*.json"))


def load_processed_data(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence_chunks(
    processed_data: dict[str, Any],
    source_path: Path,
    included_sections: list[str],
    excluded_sections: list[str],
    include_tables: bool,
    max_chars_per_chunk: int,
) -> list[EvidenceChunk]:
    paper_id = clean_text(processed_data.get("paper_id")) or source_path.stem
    chunks = []
    for index, section in enumerate(processed_data.get("sections", [])):
        if not isinstance(section, dict):
            continue
        normalized = clean_text(section.get("normalized_section") or section.get("section") or "unknown").lower()
        if not _section_allowed(normalized, included_sections, excluded_sections):
            continue
        text = clean_text(section.get("text"))
        if not text:
            continue
        title = clean_text(section.get("title")) or normalized
        page_number = section.get("page_number")
        for split_index, split_text in enumerate(split_text_for_chunks(text, max_chars_per_chunk)):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{paper_id}:section:{index}:{split_index}",
                    paper_id=paper_id,
                    source_path=str(source_path),
                    source_type="section",
                    section=normalized,
                    title=title,
                    page_number=page_number,
                    text=split_text,
                )
            )

    if include_tables:
        for index, table in enumerate(processed_data.get("tables", [])):
            if not isinstance(table, dict):
                continue
            section = clean_text(table.get("section") or "unknown").lower()
            if not _section_allowed(section, included_sections, excluded_sections):
                continue
            table_text = "\n".join(
                part
                for part in [
                    "Caption: " + clean_text(table.get("caption")) if clean_text(table.get("caption")) else "",
                    clean_text(table.get("markdown")),
                ]
                if part
            )
            if not table_text:
                continue
            for split_index, split_text in enumerate(split_text_for_chunks(table_text, max_chars_per_chunk)):
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{paper_id}:table:{index}:{split_index}",
                        paper_id=paper_id,
                        source_path=str(source_path),
                        source_type="table",
                        section=section,
                        title=clean_text(table.get("caption")) or f"Table {index + 1}",
                        page_number=table.get("page_number"),
                        text=split_text,
                    )
                )
    return chunks


def build_hysteresis_figure_chunks(
    manifest_path: Path,
    *,
    include_captions: bool,
    include_images: bool,
    include_caption_chunks_with_images: bool,
    max_chars_per_chunk: int,
    image_include_caption_context: bool = True,
    max_assets: int | None = None,
) -> list[EvidenceChunk]:
    if not include_captions and not include_images:
        return []

    assets = load_asset_manifest(manifest_path)
    if max_assets is not None:
        assets = assets[: int(max_assets)]

    chunks: list[EvidenceChunk] = []
    for index, asset in enumerate(assets):
        paper_id = clean_text(asset.get("paper_id")) or f"hysteresis_asset_{index + 1}"
        asset_id = clean_text(asset.get("asset_id")) or f"{paper_id}:figure:{index + 1}"
        caption = clean_text(asset.get("caption"))
        figure_label = clean_text(asset.get("figure_label")) or f"Figure {index + 1}"
        image_path = clean_text(asset.get("image_path"))
        caption_path = clean_text(asset.get("caption_path"))
        source_path = caption_path or clean_text(asset.get("source_json_path")) or str(manifest_path)
        page_number = asset.get("page_number")
        base_metadata = {
            "asset_id": asset_id,
            "figure_label": figure_label,
            "caption_path": caption_path,
            "original_image_path": clean_text(asset.get("original_image_path")),
            "selection_method": clean_text(asset.get("selection_method")),
            "prompt_context": HYSTERESIS_LOOP_CONTEXT.strip(),
        }

        should_add_caption_chunk = include_captions and (
            not include_images or include_caption_chunks_with_images
        )
        if should_add_caption_chunk and caption:
            text = _hysteresis_caption_text(figure_label=figure_label, caption=caption)
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{paper_id}:hysteresis_caption:{asset_id}",
                    paper_id=paper_id,
                    source_path=source_path,
                    source_type="hysteresis_figure_caption",
                    section="figure_captions",
                    title=figure_label,
                    page_number=page_number if isinstance(page_number, int) else None,
                    text=_truncate_chunk_text(text, max_chars_per_chunk),
                    metadata=base_metadata,
                )
            )

        if include_images and image_path:
            text = _hysteresis_image_text(
                figure_label=figure_label,
                caption=caption,
                include_caption_context=image_include_caption_context,
            )
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{paper_id}:hysteresis_image:{asset_id}",
                    paper_id=paper_id,
                    source_path=image_path,
                    source_type="hysteresis_figure_image",
                    section="figure_images",
                    title=figure_label,
                    page_number=page_number if isinstance(page_number, int) else None,
                    text=_truncate_chunk_text(text, max_chars_per_chunk),
                    metadata=base_metadata,
                    image_path=image_path,
                )
            )
    return chunks


def split_text_for_chunks(text: str, max_chars: int) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        if len(sentence) > max_chars:
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_candidate_terms(termo_dir: Path, max_terms: int) -> list[str]:
    terms = []
    seen = set()
    for csv_path in sorted(Path(termo_dir).glob("*.terms.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                term = clean_text(row[0])
                key = term.lower()
                if term and key not in seen:
                    seen.add(key)
                    terms.append(term)
                if len(terms) >= max_terms:
                    return terms
    return terms


def select_terms_for_chunk(candidate_terms: list[str], chunk_text: str, max_terms: int) -> list[str]:
    lower_text = chunk_text.lower()
    matches = [term for term in candidate_terms if term.lower() in lower_text]
    return matches[:max_terms]


def quote_in_text(quote: str, text: str) -> bool:
    quote = clean_text(quote).lower()
    text = clean_text(text).lower()
    if not quote:
        return False
    return quote in text


def _section_allowed(section: str, included_sections: list[str], excluded_sections: list[str]) -> bool:
    included = {item.lower() for item in included_sections}
    excluded = {item.lower() for item in excluded_sections}
    if section in excluded:
        return False
    return not included or section in included


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[index:index + max_chars].strip() for index in range(0, len(text), max_chars)]


def _hysteresis_caption_text(*, figure_label: str, caption: str) -> str:
    return "\n".join(
        [
            "Hysteresis-loop figure caption evidence.",
            f"Figure: {figure_label}",
            f"Caption: {caption}",
        ]
    )


def _hysteresis_image_text(*, figure_label: str, caption: str, include_caption_context: bool) -> str:
    lines = [
        "Hysteresis-loop visual evidence. The attached image is the primary evidence for this chunk.",
        "Inspect the figure directly and extract visible panel labels, axes, legends, curve mappings, annotated values, and visual loop comparisons.",
    ]
    if include_caption_context:
        lines.extend(
            [
                "Use the caption only to identify samples, conditions, and figure context.",
                f"Figure: {figure_label}",
                f"Caption: {caption}",
            ]
        )
    else:
        lines.extend(
            [
                "No caption text is provided for this image-only ablation chunk; rely on the attached image and visible in-figure text only.",
                f"Figure: {figure_label}",
            ]
        )
    return "\n".join(lines)


def _truncate_chunk_text(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
