from __future__ import annotations

"""
Extract PDF table evidence into JSONL and markdown files.

This module uses PyMuPDF's page.find_tables() when available. Table extraction is
best-effort because PDF layout quality varies widely; missing tables are treated
as a normal outcome, not a fatal error.
"""

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pymupdf
except ImportError:  # pragma: no cover - depends on the installed PyMuPDF package name.
    try:
        import fitz as pymupdf
    except ImportError:  # pragma: no cover
        pymupdf = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableRecord:
    source_file: str
    paper_id: str
    page: int
    table_index: int
    caption: str
    columns: list[str]
    rows: list[dict[str, str]]


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _unique_columns(raw_columns: list[str], width: int) -> list[str]:
    columns = [_clean_cell(column) for column in raw_columns]
    if len(columns) < width:
        columns.extend([f"Column {index + 1}" for index in range(len(columns), width)])
    columns = columns[:width]

    seen = {}
    unique = []
    for index, column in enumerate(columns):
        name = column or f"Column {index + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name} {count + 1}")
    return unique


def _rows_to_dicts(table_rows: list[list[Any]]) -> tuple[list[str], list[dict[str, str]]]:
    if not table_rows:
        return [], []
    width = max(len(row) for row in table_rows)
    header = [_clean_cell(cell) for cell in table_rows[0]]
    columns = _unique_columns(header, width)

    rows = []
    for raw_row in table_rows[1:]:
        values = [_clean_cell(cell) for cell in raw_row]
        if len(values) < width:
            values.extend([""] * (width - len(values)))
        row = {columns[index]: values[index] for index in range(width)}
        if any(value for value in row.values()):
            rows.append(row)
    return columns, rows


def _candidate_caption_text(page, bbox: tuple[float, float, float, float]) -> list[str]:
    blocks = page.get_text("blocks")
    candidates = []
    x0, y0, x1, y1 = bbox
    del x0, x1
    for block in blocks:
        bx0, by0, bx1, by1, text = block[:5]
        del bx0, bx1
        normalized = " ".join(str(text).split())
        if not normalized:
            continue
        near_above = by1 <= y0 and by1 >= y0 - 120
        near_below = by0 >= y1 and by0 <= y1 + 80
        if near_above or near_below:
            candidates.append(normalized)
    return candidates


def _extract_caption(page, table: Any) -> str:
    bbox = getattr(table, "bbox", None)
    if not bbox:
        return ""
    caption_pattern = re.compile(r"^(Table|TABLE|Tab\.)\s+\d+[.:]?\s+\S.*")
    for candidate in _candidate_caption_text(page, bbox):
        first_line = candidate.splitlines()[0].strip()
        if caption_pattern.match(first_line):
            return first_line
    return ""


def _extract_raw_rows(table: Any) -> list[list[Any]]:
    try:
        rows = table.extract()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Could not extract rows from a detected table: %s", exc)
        return []
    if rows is None:
        return []
    return [list(row or []) for row in rows]


def table_record_to_markdown(record: TableRecord) -> str:
    lines = [
        f"[TABLE paper_id={record.paper_id} page={record.page} table={record.table_index}]",
    ]
    if record.caption:
        lines.append(f"Caption: {record.caption}")
    else:
        lines.append("Caption:")
    lines.append("Columns: " + " | ".join(record.columns))
    lines.append("Rows:")
    for row in record.rows:
        serialized_cells = [f"{column}={row.get(column, '')}" for column in record.columns]
        lines.append("- " + "; ".join(serialized_cells))
    return "\n".join(lines)


def write_table_records(
    records: list[TableRecord],
    output_dir: Path,
    pdf_stem: str,
    output_format: str = "both",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if output_format in {"jsonl", "both"}:
        jsonl_path = output_dir / f"{pdf_stem}.tables.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        written["jsonl"] = jsonl_path

    if output_format in {"markdown", "md", "both"}:
        markdown_path = output_dir / f"{pdf_stem}.tables.md"
        markdown = "\n\n".join(table_record_to_markdown(record) for record in records)
        markdown_path.write_text(markdown, encoding="utf-8")
        written["markdown"] = markdown_path
    return written


def extract_tables(
    pdf_path: Path,
    output_dir: Path,
    output_format: str = "both",
) -> list[TableRecord]:
    """
    Extract tables from a PDF into JSONL and/or markdown files.

    Missing tables are represented by empty output files and an empty list.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if output_format not in {"jsonl", "markdown", "md", "both"}:
        raise ValueError("output_format must be one of: jsonl, markdown, md, both.")
    if pymupdf is None:
        raise ImportError("PyMuPDF is required for table extraction. Install pymupdf or fitz.")

    records = []
    try:
        document = pymupdf.open(pdf_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not open PDF for table extraction: {pdf_path}") from exc

    with document:
        for page_index, page in enumerate(document, start=1):
            if not hasattr(page, "find_tables"):
                logger.warning(
                    "PyMuPDF page.find_tables() is unavailable; no tables extracted from %s.",
                    pdf_path,
                )
                break
            try:
                table_finder = page.find_tables()
                tables = getattr(table_finder, "tables", []) or []
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Table detection failed on page %s of %s: %s", page_index, pdf_path, exc)
                continue

            for table_index, table in enumerate(tables, start=1):
                raw_rows = _extract_raw_rows(table)
                columns, rows = _rows_to_dicts(raw_rows)
                if not columns and not rows:
                    continue
                records.append(
                    TableRecord(
                        source_file=str(pdf_path),
                        paper_id=pdf_path.stem,
                        page=page_index,
                        table_index=table_index,
                        caption=_extract_caption(page, table),
                        columns=columns,
                        rows=rows,
                    )
                )

    write_table_records(records, output_dir, pdf_path.stem, output_format=output_format)
    if not records:
        logger.warning("No tables extracted from %s.", pdf_path)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PDF tables into JSONL and markdown.")
    parser.add_argument("pdf", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where table files are written. Defaults to the PDF directory.",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "markdown", "md", "both"],
        default="both",
        help="Output file format.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    pdf = Path(args.pdf)
    output_dir = Path(args.output_dir) if args.output_dir else pdf.parent
    extract_tables(pdf, output_dir, output_format=args.output_format)
