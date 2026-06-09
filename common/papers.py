"""Discovery helpers for parsed-paper input folders and image assets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}


@dataclass(frozen=True)
class PaperInput:
    paper_id: str
    json_path: Path
    paper_dir: Path
    figures_dir: Path | None
    figure_paths: tuple[Path, ...]
    collection: str
    index: int | None
    layout: str

    def asset_payload(self) -> dict:
        payload = {
            "paper_id": self.paper_id,
            "paper_dir": str(self.paper_dir),
            "json_path": str(self.json_path),
            "figures_dir": str(self.figures_dir) if self.figures_dir else "",
            "figure_paths": [str(path) for path in self.figure_paths],
            "collection": self.collection,
            "index": self.index,
            "layout": self.layout,
        }
        return payload


def _collection_and_index(folder_name: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<collection>.+)_(?P<index>\d+)$", folder_name)
    if not match:
        return folder_name, None
    return match.group("collection"), int(match.group("index"))


def _json_priority(folder: Path, json_path: Path) -> tuple[int, str]:
    if json_path.stem == folder.name:
        return (0, json_path.name)
    if json_path.name.endswith(".parsed.json") or json_path.name.endswith("_parsed.json"):
        return (1, json_path.name)
    return (2, json_path.name)


def _figure_paths(figures_dir: Path | None) -> tuple[Path, ...]:
    if figures_dir is None or not figures_dir.exists():
        return ()
    return tuple(
        sorted(
            path
            for path in figures_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def _folder_input(folder: Path) -> PaperInput | None:
    json_files = sorted(folder.glob("*.json"), key=lambda path: _json_priority(folder, path))
    if not json_files:
        return None
    collection, index = _collection_and_index(folder.name)
    figures_dir = folder / "figures"
    return PaperInput(
        paper_id=folder.name,
        json_path=json_files[0],
        paper_dir=folder,
        figures_dir=figures_dir if figures_dir.exists() else None,
        figure_paths=_figure_paths(figures_dir),
        collection=collection,
        index=index,
        layout="folder",
    )


def _flat_input(json_path: Path) -> PaperInput:
    collection, index = _collection_and_index(json_path.stem)
    return PaperInput(
        paper_id=json_path.stem,
        json_path=json_path,
        paper_dir=json_path.parent,
        figures_dir=None,
        figure_paths=(),
        collection=collection,
        index=index,
        layout="flat_json",
    )


def discover_paper_inputs(
    paper_root: str | Path,
    *,
    include_flat_json: bool = True,
) -> list[PaperInput]:
    """Discover parsed JSON inputs, preferring paper folders over legacy flat JSON files."""
    root = Path(paper_root)
    if not root.exists():
        return []

    folder_inputs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        paper_input = _folder_input(child)
        if paper_input is not None:
            folder_inputs.append(paper_input)

    by_id = {paper.paper_id: paper for paper in folder_inputs}
    if include_flat_json:
        for json_path in sorted(root.glob("*.json")):
            if json_path.name.startswith("."):
                continue
            if json_path.stem in by_id:
                continue
            by_id[json_path.stem] = _flat_input(json_path)

    return sorted(by_id.values(), key=lambda paper: paper.paper_id)


def find_paper_input_for_json(json_path: str | Path, paper_root: str | Path | None = None) -> PaperInput:
    path = Path(json_path).resolve()
    if paper_root is not None:
        for paper_input in discover_paper_inputs(paper_root):
            if paper_input.json_path.resolve() == path:
                return paper_input

    parent = path.parent
    if parent.name != Path(paper_root or parent).name:
        folder_input = _folder_input(parent)
        if folder_input is not None and folder_input.json_path.resolve() == path:
            return folder_input
    return _flat_input(path)


def write_paper_manifest(papers: Iterable[PaperInput], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [paper.asset_payload() for paper in papers]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
