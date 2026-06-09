"""
Config-driven evidence-aware OntoGen pipeline.

Run from this directory with:
    python run.py

All user-facing settings live in config.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import config
from common.papers import discover_paper_inputs, find_paper_input_for_json, write_paper_manifest
from common.run_context import record_pipeline_run
from build_enriched_context import (
    build_enriched_context,
    build_enriched_context_from_processed_data,
)
from extract_plain_text import process_pdf
from extract_sections import extract_sections
from extract_tables import extract_tables
from generate_categories import generate_categories
from generate_taxonomy import generate_taxonomy
from processed_data import (
    build_processed_data_from_existing_outputs,
    normalize_airqa_processed_data,
)
from run_termo import run_termo_steps_on_file

logger = logging.getLogger(__name__)


def _as_path(path_like) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _ensure_directories() -> None:
    for directory in [
        config.DATA_DIR,
        config.PDF_DIR,
        config.TEXT_DIR,
        config.SECTION_DIR,
        config.TABLE_DIR,
        config.PROCESSED_DATA_DIR,
        config.ENRICHED_DIR,
        config.TERMO_DIR,
        config.CATEGORY_DIR,
        config.TAXONOMY_DIR,
    ]:
        _as_path(directory).mkdir(parents=True, exist_ok=True)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    deduped = []
    for path in paths:
        key = path.resolve() if path.exists() else path.absolute()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_input_files() -> list[Path]:
    configured_files = getattr(config, "PDF_FILES", [])
    if not configured_files:
        pattern = "**/*.pdf" if getattr(config, "RECURSIVE_PDF_DISCOVERY", False) else "*.pdf"
        input_files = list(_as_path(config.PDF_DIR).glob(pattern))
        if getattr(config, "INCLUDE_PARSED_JSON_INPUTS", True):
            paper_inputs = discover_paper_inputs(_as_path(config.PDF_DIR))
            write_paper_manifest(paper_inputs, _as_path(config.RUN_OUTPUT_DIR) / "input_papers.json")
            input_files.extend(paper.json_path for paper in paper_inputs)
        return sorted(_dedupe_paths(input_files))

    input_files = []
    for file_name in configured_files:
        path = _as_path(file_name)
        if not path.is_absolute():
            path = _as_path(config.PDF_DIR) / path
        input_files.append(path)
    return _dedupe_paths(input_files)


def _paper_id_for_input(input_path: Path) -> str:
    if input_path.suffix.lower() == ".json":
        return find_paper_input_for_json(input_path, config.PDF_DIR).paper_id
    return input_path.stem


def _text_path_for_pdf(pdf_path: Path) -> Path:
    return _as_path(config.TEXT_DIR) / f"{_paper_id_for_input(pdf_path)}.processed.{config.TEXT_SOURCE}.txt"


def _sections_jsonl_for_text(text_path: Path) -> Path:
    return _as_path(config.SECTION_DIR) / f"{text_path.stem}.sections.jsonl"


def _tables_jsonl_for_pdf(pdf_path: Path) -> Path:
    return _as_path(config.TABLE_DIR) / f"{_paper_id_for_input(pdf_path)}.tables.jsonl"


def _processed_data_path_for_pdf(pdf_path: Path) -> Path:
    return _as_path(config.PROCESSED_DATA_DIR) / f"{_paper_id_for_input(pdf_path)}.processed_data.json"


def _enriched_path_for_pdf(pdf_path: Path) -> Path:
    return _as_path(config.ENRICHED_DIR) / f"{_paper_id_for_input(pdf_path)}.enriched.txt"


def _should_reuse(path: Path) -> bool:
    return bool(getattr(config, "REUSE_INTERMEDIATE_OUTPUTS", False) and Path(path).exists())


def _expected_termo_outputs(input_path: Path) -> list[Path]:
    output_dir = _as_path(config.TERMO_DIR)
    stem = Path(input_path).stem
    expected = []
    if config.RUN_TERMO_TERMS:
        expected.append(output_dir / f"{stem}.terms.csv")
    if config.RUN_TERMO_ACRONYMS:
        expected.append(output_dir / f"{stem}.acronyms.csv")
    if config.RUN_TERMO_DEFINITIONS:
        expected.append(output_dir / f"{stem}.definitions.csv")
    if config.RUN_TERMO_RELATIONSHIPS:
        expected.append(output_dir / f"{stem}.relationships.csv")
    return expected


def _option_dict(**kwargs) -> dict:
    return {key: value for key, value in kwargs.items() if value is not None}


def _process_pdf_to_text(pdf_path: Path) -> None:
    logger.info("Extracting text from %s with %s.", pdf_path, config.TEXT_SOURCE)
    process_pdf(
        pdf_path,
        threshold_num_repetitions=config.NOUGAT_THRESHOLD_NUM_REPETITIONS,
        output_dir=config.TEXT_DIR,
        text_source=config.TEXT_SOURCE,
        batchsize=config.NOUGAT_BATCHSIZE,
        model=config.NOUGAT_MODEL,
        recompute=config.NOUGAT_RECOMPUTE,
        full_precision=config.NOUGAT_FULL_PRECISION,
        markdown=config.NOUGAT_MARKDOWN,
        no_skipping=config.NOUGAT_NO_SKIPPING,
        pages=config.NOUGAT_PAGES,
    )


def _extract_sections_for_paper(text_path: Path) -> Path | None:
    if not text_path.exists():
        logger.error("Cannot extract sections because text file is missing: %s", text_path)
        return None
    logger.info("Extracting sections from %s.", text_path)
    extract_sections(
        input_path=text_path,
        sections=config.SECTIONS_TO_EXTRACT,
        output_dir=config.SECTION_DIR,
        source_type=config.TEXT_SOURCE,
        strict=config.STRICT_SECTION_EXTRACTION,
    )
    return _sections_jsonl_for_text(text_path)


def _extract_tables_for_paper(pdf_path: Path) -> Path | None:
    if not config.EXTRACT_TABLES:
        return None
    if pdf_path.suffix.lower() != ".pdf":
        logger.info("Skipping table extraction for non-PDF input: %s.", pdf_path.name)
        return None
    logger.info("Extracting tables from %s.", pdf_path)
    extract_tables(
        pdf_path=pdf_path,
        output_dir=config.TABLE_DIR,
        output_format=config.TABLE_OUTPUT_FORMAT,
    )
    return _tables_jsonl_for_pdf(pdf_path)


def _airqa_input_path_for_pdf(pdf_path: Path) -> Path | None:
    airqa_dir = getattr(config, "AIRQA_PROCESSED_DATA_DIR", None)
    if airqa_dir is None:
        return None
    airqa_dir = _as_path(airqa_dir)
    candidates = [
        airqa_dir / f"{pdf_path.stem}.processed_data.json",
        airqa_dir / f"{pdf_path.stem}.processed.json",
        airqa_dir / f"{pdf_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_processed_data_for_paper(pdf_path: Path, text_path: Path) -> Path | None:
    del text_path
    source = getattr(config, "PROCESSED_DATA_SOURCE", "existing")
    if source not in {"existing", "airqa_mineru", "auto"}:
        raise ValueError("PROCESSED_DATA_SOURCE must be 'existing', 'airqa_mineru', or 'auto'.")

    output_path = _processed_data_path_for_pdf(pdf_path)
    if _should_reuse(output_path):
        logger.info("Reusing processed_data at %s.", output_path)
        return output_path
    if pdf_path.suffix.lower() == ".json":
        logger.info("Normalizing parsed JSON input from %s.", pdf_path)
        return normalize_airqa_processed_data(pdf_path, output_path)

    airqa_input = _airqa_input_path_for_pdf(pdf_path)
    if source == "airqa_mineru" or (source == "auto" and airqa_input is not None):
        if airqa_input is None:
            logger.error("No AirQA/MinerU-style JSON found for %s.", pdf_path.name)
            return None
        logger.info("Normalizing AirQA/MinerU-style processed data from %s.", airqa_input)
        return normalize_airqa_processed_data(airqa_input, output_path)

    sections_jsonl = _sections_jsonl_for_text(_text_path_for_pdf(pdf_path))
    tables_jsonl = _tables_jsonl_for_pdf(pdf_path)
    logger.info("Building processed_data from existing section/table outputs at %s.", output_path)
    return build_processed_data_from_existing_outputs(
        paper_id=pdf_path.stem,
        source_file=pdf_path,
        sections_jsonl=sections_jsonl if sections_jsonl.exists() else None,
        tables_jsonl=tables_jsonl if tables_jsonl.exists() else None,
        output_json=output_path,
    )


def _build_context_for_paper(pdf_path: Path, text_path: Path) -> Path | None:
    processed_data_path = _processed_data_path_for_pdf(pdf_path)
    output_path = _enriched_path_for_pdf(pdf_path)
    if _should_reuse(output_path):
        logger.info("Reusing enriched context at %s.", output_path)
        return output_path
    if getattr(config, "USE_PROCESSED_DATA_FOR_ENRICHED_CONTEXT", False) and processed_data_path.exists():
        logger.info("Building enriched context from processed_data at %s.", output_path)
        try:
            return build_enriched_context_from_processed_data(
                processed_data_path=processed_data_path,
                output_path=output_path,
                include_sections=config.INCLUDE_SECTIONS_IN_ENRICHED_CONTEXT,
                include_tables=config.INCLUDE_TABLES_IN_ENRICHED_CONTEXT,
                include_figure_captions=config.INCLUDE_FIGURE_CAPTIONS_IN_ENRICHED_CONTEXT,
                include_equations=config.INCLUDE_EQUATIONS_IN_ENRICHED_CONTEXT,
                max_chars=config.ENRICHED_MAX_CHARS_PER_PAPER,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Could not build enriched context from processed_data for %s: %s. Falling back to section/table JSONL.",
                pdf_path.name,
                exc,
            )

    if getattr(config, "USE_PROCESSED_DATA_FOR_ENRICHED_CONTEXT", False):
        logger.warning(
            "processed_data was requested but is missing for %s; falling back to section/table JSONL.",
            pdf_path.name,
        )

    sections_jsonl = _sections_jsonl_for_text(text_path)
    if not sections_jsonl.exists():
        logger.error("Cannot build enriched context because sections JSONL is missing: %s", sections_jsonl)
        return None
    tables_jsonl = _tables_jsonl_for_pdf(pdf_path)
    if not tables_jsonl.exists():
        tables_jsonl = None
    output_path = _enriched_path_for_pdf(pdf_path)
    logger.info("Building enriched context at %s.", output_path)
    return build_enriched_context(
        sections_jsonl=sections_jsonl,
        output_path=output_path,
        include_sections=config.INCLUDE_SECTIONS_IN_ENRICHED_CONTEXT,
        tables_jsonl=tables_jsonl,
        include_tables=config.INCLUDE_TABLES_IN_ENRICHED_CONTEXT,
    )


def _select_termo_input(pdf_path: Path, text_path: Path) -> Path | None:
    enriched_path = _enriched_path_for_pdf(pdf_path)
    if config.CONTEXT_KIND in {"evidence", "table", "enriched"}:
        if enriched_path.exists():
            return enriched_path
        logger.error("TERMO context_kind=%s requires enriched context: %s", config.CONTEXT_KIND, enriched_path)
        return None
    if text_path.exists():
        return text_path
    logger.error("TERMO input text file is missing: %s", text_path)
    return None


def _run_termo_for_paper(pdf_path: Path, text_path: Path) -> Path | None:
    termo_input = _select_termo_input(pdf_path, text_path)
    if termo_input is None:
        return None
    expected_outputs = _expected_termo_outputs(termo_input)
    if expected_outputs and all(_should_reuse(path) for path in expected_outputs):
        logger.info("Reusing TERMO outputs for %s.", termo_input)
        return termo_input
    logger.info("Running TERMO on %s.", termo_input)
    run_termo_steps_on_file(
        input_path=termo_input,
        output_dir=config.TERMO_DIR,
        terms_model=config.TERMO_TERMS_MODEL,
        acronyms_model=config.TERMO_ACRONYMS_MODEL,
        definitions_model=config.TERMO_DEFINITIONS_MODEL,
        relationships_model=config.TERMO_RELATIONSHIPS_MODEL,
        terms_backend=config.TERMO_TERMS_BACKEND,
        acronyms_backend=config.TERMO_ACRONYMS_BACKEND,
        definitions_backend=config.TERMO_DEFINITIONS_BACKEND,
        relationships_backend=config.TERMO_RELATIONSHIPS_BACKEND,
        context_kind=config.CONTEXT_KIND,
        relationship_mode=config.RELATIONSHIP_MODE,
        include_literal_values=config.INCLUDE_LITERAL_VALUES,
        preserve_table_blocks=config.PRESERVE_TABLE_BLOCKS,
        terms_temperature=config.TERMO_TERMS_TEMPERATURE,
        acronyms_temperature=config.TERMO_ACRONYMS_TEMPERATURE,
        definitions_temperature=config.TERMO_DEFINITIONS_TEMPERATURE,
        relationships_temperature=config.TERMO_RELATIONSHIPS_TEMPERATURE,
        terms_num_ctx=config.TERMO_TERMS_NUM_CTX,
        acronyms_num_ctx=config.TERMO_ACRONYMS_NUM_CTX,
        definitions_num_ctx=config.TERMO_DEFINITIONS_NUM_CTX,
        relationships_num_ctx=config.TERMO_RELATIONSHIPS_NUM_CTX,
        terms_max_completion_tokens=config.TERMO_TERMS_MAX_COMPLETION_TOKENS,
        acronyms_max_completion_tokens=config.TERMO_ACRONYMS_MAX_COMPLETION_TOKENS,
        definitions_max_completion_tokens=config.TERMO_DEFINITIONS_MAX_COMPLETION_TOKENS,
        relationships_max_completion_tokens=config.TERMO_RELATIONSHIPS_MAX_COMPLETION_TOKENS,
        terms_reasoning_effort=config.TERMO_TERMS_REASONING_EFFORT,
        acronyms_reasoning_effort=config.TERMO_ACRONYMS_REASONING_EFFORT,
        definitions_reasoning_effort=config.TERMO_DEFINITIONS_REASONING_EFFORT,
        relationships_reasoning_effort=config.TERMO_RELATIONSHIPS_REASONING_EFFORT,
        terms_base_url=config.TERMO_TERMS_BASE_URL,
        acronyms_base_url=config.TERMO_ACRONYMS_BASE_URL,
        definitions_base_url=config.TERMO_DEFINITIONS_BASE_URL,
        relationships_base_url=config.TERMO_RELATIONSHIPS_BASE_URL,
        max_length_split_terms=config.TERMO_MAX_LENGTH_SPLIT_TERMS,
        max_length_split_acronyms=config.TERMO_MAX_LENGTH_SPLIT_ACRONYMS,
        max_length_split_definitions=config.TERMO_MAX_LENGTH_SPLIT_DEFINITIONS,
        max_length_split_relationships=config.TERMO_MAX_LENGTH_SPLIT_RELATIONSHIPS,
        remove_hallucinated=config.REMOVE_HALLUCINATED_TERMS,
        run_terms=config.RUN_TERMO_TERMS,
        run_acronyms=config.RUN_TERMO_ACRONYMS,
        run_definitions=config.RUN_TERMO_DEFINITIONS,
        run_relationships=config.RUN_TERMO_RELATIONSHIPS,
    )
    return termo_input


def _category_seed_path() -> Path | None:
    configured = getattr(config, "CATEGORY_SEED_FILE", None)
    if configured:
        path = _as_path(configured)
        return path if path.is_absolute() else _as_path(config.PROJECT_ROOT) / path
    candidate = _as_path(config.CATEGORY_DIR) / f"{config.DOMAIN_NAME}_categories_seed.0.txt"
    if candidate.exists():
        return candidate
    seeds = sorted(_as_path(config.CATEGORY_DIR).glob("*_categories_seed.0.txt"))
    return seeds[0] if seeds else None


def _run_category_generation(context_files: list[Path]) -> list[Path]:
    if not context_files:
        logger.error("Cannot generate categories because no context files are available.")
        return []
    logger.info("Generating category seeds from %s context files.", len(context_files))
    return generate_categories(
        txt_files=context_files,
        main_topic=config.DOMAIN_NAME,
        generation_model=config.CATEGORY_GENERATION_MODEL,
        format_model=config.CATEGORY_FORMAT_MODEL,
        synthesis_model=config.CATEGORY_SYNTHESIS_MODEL,
        generation_backend=config.CATEGORY_GENERATION_BACKEND,
        format_backend=config.CATEGORY_FORMAT_BACKEND,
        synthesis_backend=config.CATEGORY_SYNTHESIS_BACKEND,
        generation_base_url=config.CATEGORY_GENERATION_BASE_URL,
        format_base_url=config.CATEGORY_FORMAT_BASE_URL,
        synthesis_base_url=config.CATEGORY_SYNTHESIS_BASE_URL,
        output_dir=config.CATEGORY_DIR,
        evidence_aware=config.EVIDENCE_AWARE_CATEGORIES,
        num_retries_consistency=config.CATEGORY_NUM_RETRIES,
        num_generated_seed=config.CATEGORY_NUM_GENERATED_SEED,
        max_chars_per_file=config.CATEGORY_MAX_CHARS_PER_FILE,
        max_total_chars=config.CATEGORY_MAX_TOTAL_CHARS,
        run_llm_curation=config.CATEGORY_RUN_LLM_CURATION,
        curation_ratio=config.CATEGORY_CURATION_RATIO,
        curation_target_count=config.CATEGORY_CURATION_TARGET_COUNT,
        curation_min_target_count=config.CATEGORY_CURATION_MIN_TARGET_COUNT,
        curation_protected_categories=config.CATEGORY_CURATION_PROTECTED_CATEGORIES,
        curation_model=config.CATEGORY_CURATION_MODEL,
        curation_backend=config.CATEGORY_CURATION_BACKEND,
        curation_base_url=config.CATEGORY_CURATION_BASE_URL,
        generation_model_options=_option_dict(
            temperature=config.CATEGORY_GENERATION_TEMPERATURE,
            num_ctx=config.CATEGORY_GENERATION_NUM_CTX,
            max_completion_tokens=config.CATEGORY_GENERATION_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.CATEGORY_GENERATION_REASONING_EFFORT,
        ),
        format_model_options=_option_dict(
            temperature=config.CATEGORY_FORMAT_TEMPERATURE,
            num_ctx=config.CATEGORY_FORMAT_NUM_CTX,
            max_completion_tokens=config.CATEGORY_FORMAT_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.CATEGORY_FORMAT_REASONING_EFFORT,
        ),
        synthesis_model_options=_option_dict(
            temperature=config.CATEGORY_SYNTHESIS_TEMPERATURE,
            num_ctx=config.CATEGORY_SYNTHESIS_NUM_CTX,
            max_completion_tokens=config.CATEGORY_SYNTHESIS_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.CATEGORY_SYNTHESIS_REASONING_EFFORT,
        ),
        curation_model_options=_option_dict(
            temperature=config.CATEGORY_CURATION_TEMPERATURE,
            num_ctx=config.CATEGORY_CURATION_NUM_CTX,
            max_completion_tokens=config.CATEGORY_CURATION_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.CATEGORY_CURATION_REASONING_EFFORT,
        ),
    )


def _run_taxonomy_generation(context_files: list[Path]) -> None:
    category_seed_file = _category_seed_path()
    if category_seed_file is None or not category_seed_file.exists():
        logger.error("Cannot generate taxonomy because no category seed file was found.")
        return
    if not context_files:
        logger.error("Cannot generate taxonomy because no context files are available.")
        return
    logger.info("Generating taxonomy with category seed %s.", category_seed_file)
    generate_taxonomy(
        category_seed_file=category_seed_file,
        txt_files=context_files,
        model=config.TAXONOMY_MODEL,
        model_params=_option_dict(
            temperature=config.TAXONOMY_TEMPERATURE,
            num_ctx=config.TAXONOMY_NUM_CTX,
            max_completion_tokens=config.TAXONOMY_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.TAXONOMY_REASONING_EFFORT,
        ),
        num_iterations=config.TAXONOMY_NUM_ITERATIONS,
        prompt_include_path=config.TAXONOMY_PROMPT_INCLUDE_PATH,
        root_dir=config.TERMO_DIR,
        output_dir=config.TAXONOMY_DIR,
        backend=config.TAXONOMY_BACKEND,
        base_url=config.TAXONOMY_BASE_URL,
        seed=config.TAXONOMY_SEED,
        sc_retry=config.TAXONOMY_SC_RETRY,
        majority=config.TAXONOMY_MAJORITY,
        max_terms_per_paper=config.TAXONOMY_MAX_TERMS_PER_PAPER,
        max_term_chars=config.TAXONOMY_MAX_TERM_CHARS,
    )


def run_pipeline() -> None:
    _ensure_directories()
    input_files = _resolve_input_files()
    if not input_files:
        logger.warning("No PDF or parsed JSON files configured or found in %s.", config.PDF_DIR)

    context_files = []
    for pdf_path in input_files:
        if not pdf_path.exists():
            logger.error("Configured paper input does not exist: %s", pdf_path)
            continue

        is_pdf = pdf_path.suffix.lower() == ".pdf"
        logger.info("Processing paper input: %s", pdf_path.name)
        text_path = _text_path_for_pdf(pdf_path)
        try:
            if config.RUN_PDF_TO_TEXT and is_pdf:
                _process_pdf_to_text(pdf_path)
            elif config.RUN_PDF_TO_TEXT and not is_pdf:
                logger.info("Skipping PDF-to-text for parsed JSON input: %s.", pdf_path.name)

            if config.RUN_SECTION_EXTRACTION and is_pdf:
                _extract_sections_for_paper(text_path)
            elif config.RUN_SECTION_EXTRACTION and not is_pdf:
                logger.info("Skipping section extraction for parsed JSON input: %s.", pdf_path.name)

            if config.RUN_TABLE_EXTRACTION:
                _extract_tables_for_paper(pdf_path)
            if config.RUN_PROCESSED_DATA_BUILD:
                try:
                    _build_processed_data_for_paper(pdf_path, text_path)
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.warning(
                        "Could not build processed_data for %s: %s. Downstream steps may use the existing section/table JSONL fallback.",
                        pdf_path.name,
                        exc,
                    )
            if config.RUN_BUILD_ENRICHED_CONTEXT:
                _build_context_for_paper(pdf_path, text_path)
            if config.RUN_TERMO:
                termo_input = _run_termo_for_paper(pdf_path, text_path)
                if termo_input is not None:
                    context_files.append(termo_input)
            else:
                fallback_context = _enriched_path_for_pdf(pdf_path)
                if fallback_context.exists():
                    context_files.append(fallback_context)
                elif text_path.exists():
                    context_files.append(text_path)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.exception("Skipping %s after pipeline failure: %s", pdf_path, exc)
            continue

    if config.RUN_CATEGORY_GENERATION:
        generated = _run_category_generation(context_files)
        if generated:
            logger.info("Wrote %s category seed files.", len(generated))

    if config.RUN_TAXONOMY_GENERATION:
        _run_taxonomy_generation(context_files)

    record_pipeline_run(
        config.RUN_OUTPUT_DIR,
        "ontogen",
        status="completed",
        inputs={
            "paper_dir": str(config.PDF_DIR),
            "input_files": [str(path) for path in input_files],
        },
        outputs={
            "processed_data_dir": str(config.PROCESSED_DATA_DIR),
            "enriched_dir": str(config.ENRICHED_DIR),
            "termo_dir": str(config.TERMO_DIR),
            "category_dir": str(config.CATEGORY_DIR),
            "taxonomy_dir": str(config.TAXONOMY_DIR),
            "context_files": [str(path) for path in context_files],
        },
    )
    logger.info("OntoGen pipeline finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    run_pipeline()
