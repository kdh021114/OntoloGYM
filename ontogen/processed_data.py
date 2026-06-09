from __future__ import annotations

"""
Normalize parser-specific PDF outputs into one processed_data JSON schema.

This adapter does not run MinerU, Docling, OCR, or any visual model. It only
reads already-produced JSON or existing OntoGen section/table outputs and stores
the explicit structure in a parser-agnostic representation.
"""

import argparse
import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CANONICAL_SECTIONS = [
    "abstract",
    "introduction",
    "methods",
    "experimental",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "ablation",
    "conclusion",
    "other",
]

SECTION_TITLE_ALIASES = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background"],
    "methods": [
        "method",
        "methods",
        "methodology",
        "materials and methods",
        "method and materials",
        "approach",
        "proposed method",
    ],
    "experimental": [
        "experimental",
        "experimental setup",
        "experiment setup",
        "experimental details",
        "implementation details",
    ],
    "experiments": ["experiment", "experiments", "experimental evaluation"],
    "evaluation": ["evaluation", "evaluation setup", "benchmark evaluation"],
    "results": [
        "result",
        "results",
        "results and discussion",
        "experimental results",
        "main results",
    ],
    "discussion": ["discussion", "analysis", "discussion and conclusions"],
    "ablation": ["ablation", "ablation study", "ablation studies", "ablations"],
    "conclusion": ["conclusion", "conclusions", "concluding remarks"],
}

NOISE_SECTION_TITLE_PATTERNS = [
    "references",
    "reference",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "author information",
    "associated content",
    "supporting information",
    "supplemental information",
    "supplementary information",
    "supplementary references",
    "supplementary figures",
    "keywords",
    "for table of contents only",
]

METHOD_SECTION_KEYWORDS = [
    "apparatus",
    "calculation",
    "calculations",
    "characterization",
    "computation",
    "computational",
    "device characterization",
    "experimental section",
    "fabrication",
    "materials",
    "measurement",
    "measurements",
    "method",
    "microscopy",
    "modeling",
    "paramagnetic resonance",
    "preparation",
    "resonance",
    "sample preparation",
    "simulation",
    "spectroscopy",
    "synthesis",
    "x-ray diffraction",
]

RESULT_SECTION_KEYWORDS = [
    "comparison",
    "effect of",
    "influence of",
    "mechanism",
    "properties",
    "results",
    "response",
    "transport",
]


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"th", "td"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"th", "td"} and self._current_cell is not None:
            if self._current_row is not None:
                self._current_row.append(_clean_text("".join(self._current_cell)))
            self._current_cell = None
        elif normalized_tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        logger.warning("JSONL file does not exist: %s", path)
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %s in %s: %s", line_number, path, exc)
    return records


def _normalize_title_text(title: str) -> str:
    normalized = _clean_text(title)
    normalized = re.sub(r"^#+\s*", "", normalized)
    normalized = re.sub(r"\\(?:sub)*section\*?\{(.+?)\}", r"\1", normalized)
    normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", normalized)
    normalized = re.sub(r"^[IVXLC]+\.\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[:.\-]+$", "", normalized)
    return normalized.lower()


def _is_noise_section_title(title: str) -> bool:
    normalized = _normalize_title_text(title)
    return any(normalized == pattern or normalized.startswith(pattern) for pattern in NOISE_SECTION_TITLE_PATTERNS)


def _section_number_root(title: str) -> str | None:
    match = re.match(r"^\s*(\d+)(?:\.\d+)*\.?\s+", _clean_text(title))
    return match.group(1) if match else None


def normalize_section_title(title: str) -> str:
    """
    Map a section title to a canonical label.
    """
    normalized = _normalize_title_text(title)
    if not normalized:
        return "other"
    for canonical, aliases in SECTION_TITLE_ALIASES.items():
        if normalized in aliases:
            return canonical
    for canonical, aliases in SECTION_TITLE_ALIASES.items():
        for alias in aliases:
            if normalized.startswith(alias + " ") or normalized.startswith(alias + ":"):
                return canonical
    if _is_noise_section_title(title):
        return "other"
    if any(keyword in normalized for keyword in METHOD_SECTION_KEYWORDS):
        return "methods"
    if any(keyword in normalized for keyword in RESULT_SECTION_KEYWORDS):
        return "results"
    return "other"


def infer_subsection_labels(sections: list[dict]) -> list[dict]:
    """
    Many parsed JSON files keep subsection titles such as "2.1 ..." as "other".
    If a numbered subsection follows a known top-level section, inherit that
    section label so experimental/result context is not dropped downstream.
    """
    current_number_root = None
    current_section = None
    inheritable = {
        "methods",
        "experimental",
        "experiments",
        "evaluation",
        "results",
        "discussion",
        "ablation",
        "conclusion",
    }
    for section in sections:
        title = section.get("title", "")
        normalized = section.get("normalized_section", "other")
        number_root = _section_number_root(title)
        if normalized in inheritable:
            current_number_root = number_root
            current_section = normalized
            continue
        if normalized != "other" or _is_noise_section_title(title):
            continue
        if number_root and number_root == current_number_root and current_section:
            section["normalized_section"] = current_section
    return sections


def _first_present(record: dict, keys: list[str], default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bbox(value: Any) -> list[float | int]:
    if isinstance(value, list) and len(value) == 4:
        return value
    if isinstance(value, tuple) and len(value) == 4:
        return list(value)
    return []


def _as_page_numbers(record: dict) -> list[int]:
    page_numbers = _first_present(record, ["page_numbers", "pages"], [])
    normalized = []
    for page in _as_list(page_numbers):
        page_number = _as_int_or_none(page)
        if page_number is not None:
            normalized.append(page_number)
    page_number = _as_int_or_none(_first_present(record, ["page_number", "page", "page_idx"], None))
    if page_number is not None and page_number not in normalized:
        normalized.append(page_number)
    return normalized


def _new_processed_data(paper_id: str, source_file: Path | str, parser_backend: str) -> dict:
    return {
        "paper_id": paper_id,
        "source_file": str(source_file) if source_file is not None else "",
        "parser_backend": parser_backend,
        "metadata": {
            "title": "",
            "authors": [],
            "year": None,
        },
        "sections": [],
        "tables": [],
        "figures": [],
        "equations": [],
    }


def _normalize_metadata(raw_metadata: Any) -> dict:
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}
    authors = raw_metadata.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]
    elif not isinstance(authors, list):
        authors = []
    return {
        "title": _clean_text(raw_metadata.get("title", "")),
        "authors": [_clean_text(author) for author in authors if _clean_text(author)],
        "year": _as_int_or_none(raw_metadata.get("year")),
    }


def _get_nested(data: dict, path: list[str], default=None):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _pick_list(data: dict, nested_key: str) -> list:
    mineru_list = _get_nested(data, ["info_from_mineru", nested_key], None)
    if isinstance(mineru_list, list):
        return mineru_list
    top_level = data.get(nested_key)
    if isinstance(top_level, list):
        return top_level
    return []


def table_html_to_markdown(table_html: str) -> str:
    """
    Convert a simple HTML table into TERMO-readable markdown-ish rows.
    """
    if not table_html:
        return ""
    parser = _HTMLTableParser()
    parser.feed(table_html)
    return table_rows_to_markdown(parser.rows)


def table_rows_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    columns = [_clean_text(cell) or f"Column {index + 1}" for index, cell in enumerate(rows[0])]
    if len(columns) < width:
        columns.extend([f"Column {index + 1}" for index in range(len(columns), width)])

    lines = ["Columns: " + " | ".join(columns[:width]), "Rows:"]
    for raw_row in rows[1:]:
        values = [_clean_text(cell) for cell in raw_row]
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        if not any(values):
            continue
        cells = [f"{columns[index]}={values[index]}" for index in range(width)]
        lines.append("- " + "; ".join(cells))
    return "\n".join(lines)


def table_dict_rows_to_markdown(columns: list[str], rows: list[dict]) -> str:
    if not columns and rows:
        columns = list(rows[0].keys())
    columns = [_clean_text(column) for column in columns if _clean_text(column)]
    if not columns:
        return ""
    lines = ["Columns: " + " | ".join(columns), "Rows:"]
    for row in rows:
        cells = [f"{column}={_clean_text(row.get(column, ''))}" for column in columns]
        if any(cell.split("=", 1)[1] for cell in cells):
            lines.append("- " + "; ".join(cells))
    return "\n".join(lines)


def _normalize_section_entry(entry: dict, source: str) -> dict:
    title = _clean_text(_first_present(entry, ["title", "section_title", "heading", "name"], ""))
    page_numbers = _as_page_numbers(entry)
    page_number = page_numbers[0] if page_numbers else None
    normalized_section = _clean_text(
        _first_present(entry, ["normalized_section", "section"], "")
    )
    if normalized_section not in CANONICAL_SECTIONS:
        normalized_section = normalize_section_title(title)
    return {
        "title": title,
        "normalized_section": normalized_section,
        "text": _clean_text(_first_present(entry, ["text", "content"], "")),
        "level": _as_int_or_none(entry.get("level")) or 1,
        "page_number": page_number,
        "page_numbers": page_numbers,
        "source": _clean_text(entry.get("source", source)) or source,
    }


def _section_for_page(page_number: int | None, sections: list[dict]) -> str:
    if page_number is None:
        return "unknown"

    exact_matches = []
    for section in sections:
        section_pages = section.get("page_numbers") or []
        if page_number in section_pages:
            exact_matches.append(section.get("normalized_section", "unknown"))
    exact_matches = [match for match in exact_matches if match and match != "other"]
    if len(set(exact_matches)) == 1:
        return exact_matches[0]

    starts = [
        (section.get("page_number"), section.get("normalized_section", "unknown"))
        for section in sections
        if section.get("page_number") is not None
    ]
    starts = sorted((page, label) for page, label in starts if label and label != "other")
    for index, (start_page, label) in enumerate(starts):
        next_start = starts[index + 1][0] if index + 1 < len(starts) else None
        if next_start is None:
            if page_number == start_page:
                return label
        elif start_page <= page_number < next_start:
            return label
    return "unknown"


def _explicit_or_page_section(entry: dict, sections: list[dict]) -> str:
    explicit = _clean_text(_first_present(entry, ["section", "normalized_section"], ""))
    if explicit:
        normalized = normalize_section_title(explicit)
        return normalized if normalized != "other" else explicit
    return _section_for_page(
        _as_int_or_none(_first_present(entry, ["page_number", "page", "page_idx"], None)),
        sections,
    )


def _normalize_table_entry(entry: dict, sections: list[dict]) -> dict:
    html = _clean_text(_first_present(entry, ["table_html", "html"], ""))
    markdown = _clean_text(_first_present(entry, ["table_markdown", "markdown", "text"], ""))
    if not markdown and html:
        markdown = table_html_to_markdown(html)
    columns = _first_present(entry, ["columns"], [])
    rows = _first_present(entry, ["rows"], [])
    if not markdown and isinstance(columns, list) and isinstance(rows, list):
        markdown = table_dict_rows_to_markdown(columns, rows)

    return {
        "caption": _clean_text(_first_present(entry, ["table_caption", "caption"], "")),
        "html": html,
        "markdown": markdown,
        "bbox": _as_bbox(_first_present(entry, ["table_bbox", "bbox"], [])),
        "page_number": _as_int_or_none(_first_present(entry, ["page_number", "page", "page_idx"], None)),
        "section": _explicit_or_page_section(entry, sections),
    }


def _normalize_figure_entry(entry: dict, sections: list[dict]) -> dict:
    return {
        "caption": _clean_text(_first_present(entry, ["figure_caption", "caption"], "")),
        "bbox": _as_bbox(_first_present(entry, ["figure_bbox", "bbox"], [])),
        "page_number": _as_int_or_none(_first_present(entry, ["page_number", "page", "page_idx"], None)),
        "section": _explicit_or_page_section(entry, sections),
    }


def _normalize_equation_entry(entry: dict, sections: list[dict]) -> dict:
    return {
        "equation_text": _clean_text(
            _first_present(entry, ["equation_text", "latex", "text", "content"], "")
        ),
        "page_number": _as_int_or_none(_first_present(entry, ["page_number", "page", "page_idx"], None)),
        "section": _explicit_or_page_section(entry, sections),
    }


def normalize_airqa_processed_data(input_json: Path, output_json: Path) -> Path:
    """
    Convert AirQA/MinerU-style processed JSON into normalized processed_data.
    """
    input_json = Path(input_json)
    data = _read_json(input_json)
    metadata = data.get("metadata") or _get_nested(data, ["info_from_mineru", "metadata"], {}) or {}
    paper_assets = data.get("paper_assets") if isinstance(data.get("paper_assets"), dict) else {}
    folder_paper_id = input_json.parent.name if (input_json.parent / "figures").exists() else ""
    paper_id = _clean_text(data.get("paper_id") or paper_assets.get("paper_id") or folder_paper_id or input_json.stem)
    source_file = data.get("source_file") or input_json
    processed = _new_processed_data(paper_id, source_file, "airqa_mineru")
    processed["metadata"] = _normalize_metadata(metadata)
    if paper_assets:
        processed["assets"] = paper_assets
    elif (input_json.parent / "figures").exists():
        processed["assets"] = {
            "paper_id": paper_id,
            "paper_dir": str(input_json.parent),
            "json_path": str(input_json),
            "figures_dir": str(input_json.parent / "figures"),
            "figure_paths": [
                str(path)
                for path in sorted((input_json.parent / "figures").iterdir())
                if path.is_file()
            ],
            "layout": "folder",
        }

    processed["sections"] = infer_subsection_labels([
        _normalize_section_entry(entry, source="toc")
        for entry in _pick_list(data, "TOC")
        if isinstance(entry, dict)
    ])
    processed["tables"] = [
        _normalize_table_entry(entry, processed["sections"])
        for entry in _pick_list(data, "tables")
        if isinstance(entry, dict)
    ]
    processed["figures"] = [
        _normalize_figure_entry(entry, processed["sections"])
        for entry in _pick_list(data, "figures")
        if isinstance(entry, dict)
    ]
    processed["equations"] = [
        _normalize_equation_entry(entry, processed["sections"])
        for entry in _pick_list(data, "equations")
        if isinstance(entry, dict)
    ]
    return _write_json(output_json, processed)


def build_processed_data_from_existing_outputs(
    paper_id: str,
    source_file: Path,
    sections_jsonl: Path | None,
    tables_jsonl: Path | None,
    output_json: Path,
) -> Path:
    """
    Convert existing OntoGen section/table JSONL outputs into processed_data.
    """
    processed = _new_processed_data(paper_id, source_file, "existing")

    section_records = _read_jsonl(sections_jsonl)
    processed["sections"] = [
        _normalize_section_entry(
            {
                "title": record.get("heading") or record.get("section", ""),
                "normalized_section": record.get("section", ""),
                "text": record.get("content", ""),
                "level": 1,
                "source": "existing",
            },
            source="existing",
        )
        for record in section_records
    ]

    table_records = _read_jsonl(tables_jsonl)
    processed["tables"] = [
        _normalize_table_entry(
            {
                "caption": record.get("caption", ""),
                "columns": record.get("columns", []),
                "rows": record.get("rows", []),
                "page_number": record.get("page"),
                "section": record.get("section", ""),
            },
            processed["sections"],
        )
        for record in table_records
    ]
    return _write_json(output_json, processed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize PDF parser outputs into processed_data JSON.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    airqa = subparsers.add_parser("airqa", help="Normalize AirQA/MinerU-style JSON.")
    airqa.add_argument("input_json", type=str)
    airqa.add_argument("output_json", type=str)

    existing = subparsers.add_parser("existing", help="Normalize existing section/table JSONL outputs.")
    existing.add_argument("paper_id", type=str)
    existing.add_argument("source_file", type=str)
    existing.add_argument("output_json", type=str)
    existing.add_argument("--sections-jsonl", type=str, default=None)
    existing.add_argument("--tables-jsonl", type=str, default=None)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if args.command == "airqa":
        normalize_airqa_processed_data(Path(args.input_json), Path(args.output_json))
    elif args.command == "existing":
        build_processed_data_from_existing_outputs(
            paper_id=args.paper_id,
            source_file=Path(args.source_file),
            sections_jsonl=Path(args.sections_jsonl) if args.sections_jsonl else None,
            tables_jsonl=Path(args.tables_jsonl) if args.tables_jsonl else None,
            output_json=Path(args.output_json),
        )
