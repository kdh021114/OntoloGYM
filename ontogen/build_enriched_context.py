from __future__ import annotations

"""
Build evidence-aware paper context files from extracted sections and tables.
"""

import argparse
import json
import logging
from pathlib import Path

from extract_sections import CANONICAL_SECTION_ORDER
from extract_tables import TableRecord, table_record_to_markdown
from processed_data import CANONICAL_SECTIONS, _is_noise_section_title

logger = logging.getLogger(__name__)

CONTEXT_ORDER = CANONICAL_SECTION_ORDER + ["conclusion", "tables"]


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        logger.warning("JSONL file does not exist: %s", path)
        return records
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


def _ordered_unique_sections(include_sections: list[str]) -> list[str]:
    requested = {section.lower().strip() for section in include_sections}
    return [section for section in CANONICAL_SECTION_ORDER if section in requested]


def _section_blocks(sections_jsonl: Path, include_sections: list[str]) -> list[str]:
    section_records = _read_jsonl(sections_jsonl)
    requested_sections = _ordered_unique_sections(include_sections)
    section_by_name = {}
    for record in section_records:
        section = str(record.get("section", "")).lower().strip()
        if section not in requested_sections:
            continue
        if section in section_by_name:
            logger.warning("Skipping duplicate section '%s' in %s.", section, sections_jsonl)
            continue
        section_by_name[section] = record

    blocks = []
    for section in requested_sections:
        record = section_by_name.get(section)
        if record is None:
            logger.warning("Section '%s' is missing from %s.", section, sections_jsonl)
            continue
        paper_id = record.get("paper_id") or Path(record.get("source_file", sections_jsonl)).stem
        content = str(record.get("content", "")).strip()
        if not content:
            logger.warning("Section '%s' in %s has no content.", section, sections_jsonl)
            continue
        blocks.append(f"[PAPER={paper_id}][SECTION={section}]\n{content}")
    return blocks


def _table_record_from_dict(record: dict) -> TableRecord:
    return TableRecord(
        source_file=str(record.get("source_file", "")),
        paper_id=str(record.get("paper_id", "")),
        page=int(record.get("page", 0) or 0),
        table_index=int(record.get("table_index", 0) or 0),
        caption=str(record.get("caption", "")),
        columns=list(record.get("columns", []) or []),
        rows=list(record.get("rows", []) or []),
    )


def _table_blocks(tables_jsonl: Path | None, include_tables: bool) -> list[str]:
    if not include_tables:
        return []
    if tables_jsonl is None:
        logger.warning("No table JSONL path was provided.")
        return []
    table_records = _read_jsonl(tables_jsonl)
    if not table_records:
        logger.warning("No table records found in %s.", tables_jsonl)
        return []

    first_record = table_records[0]
    paper_id = first_record.get("paper_id") or Path(tables_jsonl).stem.replace(".tables", "")
    blocks = [f"[PAPER={paper_id}][SECTION=tables]"]
    for record in table_records:
        blocks.append(table_record_to_markdown(_table_record_from_dict(record)))
    return ["\n\n".join(blocks)]


def build_enriched_context(
    sections_jsonl: Path,
    output_path: Path,
    include_sections: list[str],
    tables_jsonl: Path | None = None,
    include_tables: bool = True,
) -> Path:
    """
    Combine section and table provenance into one context file per paper.
    """
    sections_jsonl = Path(sections_jsonl)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = _section_blocks(sections_jsonl, include_sections)
    blocks.extend(_table_blocks(Path(tables_jsonl) if tables_jsonl else None, include_tables))
    output_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")
    return output_path


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Processed data JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ordered_processed_sections(include_sections: list[str]) -> list[str]:
    requested = {section.lower().strip() for section in include_sections}
    return [section for section in CANONICAL_SECTIONS if section in requested]


def _format_page(page_number) -> str:
    return str(page_number) if page_number is not None else "unknown"


def _join_and_trim_blocks(blocks: list[str], max_chars: int | None = None) -> str:
    text = "\n\n".join(blocks).strip()
    if max_chars is None or len(text) <= int(max_chars):
        return text
    max_chars = int(max_chars)
    marker = "\n\n[TRUNCATED] Enriched context was capped for cost control."
    return text[: max(0, max_chars - len(marker))].rstrip() + marker


def _processed_section_blocks(processed_data: dict, include_sections: list[str]) -> list[str]:
    requested_sections = _ordered_processed_sections(include_sections)
    records_by_section = {section: [] for section in requested_sections}
    for record in processed_data.get("sections", []):
        section = str(record.get("normalized_section", "")).lower().strip()
        if section not in records_by_section:
            continue
        if section == "other" and _is_noise_section_title(str(record.get("title", ""))):
            continue
        text = str(record.get("text", "")).strip()
        if not text:
            logger.warning(
                "Processed section '%s' for paper %s has no text.",
                section,
                processed_data.get("paper_id", ""),
            )
            continue
        records_by_section[section].append(record)

    blocks = []
    seen = set()
    paper_id = processed_data.get("paper_id", "")
    for section in requested_sections:
        records = records_by_section.get(section, [])
        if not records:
            logger.warning("Processed section '%s' is missing or empty.", section)
            continue
        for record in records:
            text = str(record.get("text", "")).strip()
            identity = (section, record.get("title", ""), text)
            if identity in seen:
                continue
            seen.add(identity)
            blocks.append(f"[PAPER={paper_id}][SECTION={section}]\n{text}")
    return blocks


def _processed_table_blocks(processed_data: dict, include_tables: bool) -> list[str]:
    if not include_tables:
        return []
    tables = processed_data.get("tables", [])
    if not tables:
        logger.warning("No processed tables found for paper %s.", processed_data.get("paper_id", ""))
        return []

    paper_id = processed_data.get("paper_id", "")
    blocks = [f"[PAPER={paper_id}][SECTION=tables]"]
    for index, table in enumerate(tables, start=1):
        page = _format_page(table.get("page_number"))
        section = table.get("section") or "unknown"
        blocks.append(f"[TABLE paper_id={paper_id} page={page} section={section} table={index}]")
        blocks.append(f"Caption: {table.get('caption', '')}")
        markdown = str(table.get("markdown", "")).strip()
        if markdown:
            blocks.append(markdown)
        elif table.get("html"):
            blocks.append(f"HTML: {table.get('html')}")
    return ["\n".join(blocks)]


def _processed_figure_blocks(processed_data: dict, include_figure_captions: bool) -> list[str]:
    if not include_figure_captions:
        return []
    figures = processed_data.get("figures", [])
    if not figures:
        logger.warning("No processed figure captions found for paper %s.", processed_data.get("paper_id", ""))
        return []

    paper_id = processed_data.get("paper_id", "")
    blocks = [f"[PAPER={paper_id}][SECTION=figure_captions]"]
    for index, figure in enumerate(figures, start=1):
        caption = str(figure.get("caption", "")).strip()
        if not caption:
            continue
        page = _format_page(figure.get("page_number"))
        section = figure.get("section") or "unknown"
        blocks.append(f"[FIGURE paper_id={paper_id} page={page} section={section} figure={index}]")
        blocks.append(f"Caption: {caption}")
    return ["\n".join(blocks)] if len(blocks) > 1 else []


def _processed_equation_blocks(processed_data: dict, include_equations: bool) -> list[str]:
    if not include_equations:
        return []
    equations = processed_data.get("equations", [])
    if not equations:
        logger.warning("No processed equations found for paper %s.", processed_data.get("paper_id", ""))
        return []

    paper_id = processed_data.get("paper_id", "")
    blocks = [f"[PAPER={paper_id}][SECTION=equations]"]
    for index, equation in enumerate(equations, start=1):
        equation_text = str(equation.get("equation_text", "")).strip()
        if not equation_text:
            continue
        page = _format_page(equation.get("page_number"))
        section = equation.get("section") or "unknown"
        blocks.append(f"[EQUATION paper_id={paper_id} page={page} section={section} equation={index}]")
        blocks.append(equation_text)
    return ["\n".join(blocks)] if len(blocks) > 1 else []


def build_enriched_context_from_processed_data(
    processed_data_path: Path,
    output_path: Path,
    include_sections: list[str],
    include_tables: bool = True,
    include_figure_captions: bool = False,
    include_equations: bool = False,
    max_chars: int | None = None,
) -> Path:
    """
    Build enriched context from normalized processed_data JSON.
    """
    processed_data = _read_json(Path(processed_data_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = _processed_section_blocks(processed_data, include_sections)
    blocks.extend(_processed_table_blocks(processed_data, include_tables))
    blocks.extend(_processed_figure_blocks(processed_data, include_figure_captions))
    blocks.extend(_processed_equation_blocks(processed_data, include_equations))
    output_path.write_text(_join_and_trim_blocks(blocks, max_chars) + "\n", encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an evidence-aware enriched context file.")
    parser.add_argument("sections_jsonl", type=str, help="Path to section JSONL.")
    parser.add_argument("output_path", type=str, help="Path to write the enriched text file.")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=CANONICAL_SECTION_ORDER,
        help="Sections to include.",
    )
    parser.add_argument("--tables-jsonl", type=str, default=None, help="Optional table JSONL file.")
    parser.add_argument(
        "--processed-data",
        type=str,
        default=None,
        help="Optional normalized processed_data JSON. If set, section/table JSONL inputs are ignored.",
    )
    parser.add_argument(
        "--include-figure-captions",
        action="store_true",
        help="Include processed figure captions. Images are not interpreted.",
    )
    parser.add_argument(
        "--include-equations",
        action="store_true",
        help="Include processed equation text.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Do not include tables even if --tables-jsonl is provided.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if args.processed_data:
        build_enriched_context_from_processed_data(
            Path(args.processed_data),
            Path(args.output_path),
            args.sections,
            include_tables=not args.no_tables,
            include_figure_captions=args.include_figure_captions,
            include_equations=args.include_equations,
        )
    else:
        build_enriched_context(
            Path(args.sections_jsonl),
            Path(args.output_path),
            args.sections,
            tables_jsonl=Path(args.tables_jsonl) if args.tables_jsonl else None,
            include_tables=not args.no_tables,
        )
