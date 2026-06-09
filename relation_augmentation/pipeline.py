from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from common.project_config import load_env_file, load_project_config
from common.run_context import record_pipeline_run

from .evidence import (
    build_evidence_chunks,
    build_hysteresis_figure_chunks,
    load_candidate_terms,
    load_processed_data,
    quote_in_text,
    resolve_input_files,
    select_terms_for_chunk,
)
from .llm import RelationLLMClient, build_extraction_prompt
from .schema import RelationClaim, RelationSchema


logger = logging.getLogger(__name__)


def run_pipeline() -> dict[str, Any]:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))

    if not getattr(config, "RELATION_RUN_AUGMENTATION", False):
        logger.info("Relation augmentation is disabled. Set RELATION_RUN_AUGMENTATION=True to run it.")
        result = {"status": "disabled"}
        record_pipeline_run(config.RUN_OUTPUT_DIR, "relation_augmentation", status="disabled")
        return result

    output_dir = Path(config.RELATION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = RelationSchema(config.RELATION_ENTITY_TYPES, config.RELATION_TYPES)
    input_files = resolve_input_files(config)
    candidate_terms = load_candidate_terms(
        Path(config.RELATION_TERMO_DIR),
        max_terms=getattr(config, "RELATION_MAX_TERMO_TERMS_TO_LOAD", 10000),
    )
    candidate_term_index = {term.lower() for term in candidate_terms}

    chunks = []
    hysteresis_chunks = []
    if getattr(config, "RELATION_INCLUDE_BASE_TEXT_CHUNKS", True):
        for input_file in input_files:
            processed_data = load_processed_data(input_file)
            file_chunks = build_evidence_chunks(
                processed_data=processed_data,
                source_path=input_file,
                included_sections=config.RELATION_INCLUDED_SECTIONS,
                excluded_sections=config.RELATION_EXCLUDED_SECTIONS,
                include_tables=config.RELATION_INCLUDE_TABLES,
                max_chars_per_chunk=config.RELATION_MAX_CHARS_PER_CHUNK,
            )
            chunks.extend(_select_chunks_for_paper(file_chunks, getattr(config, "RELATION_MAX_CHUNKS_PER_PAPER", None)))

    if getattr(config, "RELATION_INCLUDE_HYSTERESIS_CAPTIONS", False) or getattr(config, "RELATION_INCLUDE_HYSTERESIS_IMAGES", False):
        hysteresis_chunks = build_hysteresis_figure_chunks(
            Path(config.RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL),
            include_captions=getattr(config, "RELATION_INCLUDE_HYSTERESIS_CAPTIONS", False),
            include_images=getattr(config, "RELATION_INCLUDE_HYSTERESIS_IMAGES", False),
            include_caption_chunks_with_images=getattr(
                config,
                "RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES",
                False,
            ),
            image_include_caption_context=getattr(
                config,
                "RELATION_HYSTERESIS_IMAGE_INCLUDE_CAPTION_CONTEXT",
                True,
            ),
            max_chars_per_chunk=config.RELATION_MAX_CHARS_PER_CHUNK,
            max_assets=getattr(config, "RELATION_MAX_HYSTERESIS_ASSETS", None),
        )
        chunks.extend(hysteresis_chunks)

    summary = {
        "status": "dry_run" if getattr(config, "RELATION_DRY_RUN", True) else "completed",
        "input_files": len(input_files),
        "candidate_terms": len(candidate_terms),
        "evidence_chunks": len(chunks),
        "base_text_evidence_enabled": bool(getattr(config, "RELATION_INCLUDE_BASE_TEXT_CHUNKS", True)),
        "base_text_evidence_chunks": len(chunks) - len(hysteresis_chunks),
        "hysteresis_evidence_chunks": len(hysteresis_chunks),
        "relation_types": len(schema.relation_types),
    }

    if getattr(config, "RELATION_DRY_RUN", True):
        summary_path = output_dir / "dry_run_summary.json"
        _write_json(summary_path, summary)
        record_pipeline_run(
            config.RUN_OUTPUT_DIR,
            "relation_augmentation",
            status="dry_run",
            inputs={
                "processed_data_dir": str(config.RELATION_PROCESSED_DATA_DIR),
                "termo_dir": str(config.RELATION_TERMO_DIR),
                "input_files": [str(path) for path in input_files],
            },
            outputs={"summary": str(summary_path)},
            extra=summary,
        )
        logger.info("Relation augmentation dry-run summary written to %s.", summary_path)
        return summary

    llm_client = RelationLLMClient(
        model=config.RELATION_MODEL,
        backend=config.RELATION_BACKEND,
        temperature=config.RELATION_TEMPERATURE,
        max_completion_tokens=config.RELATION_MAX_COMPLETION_TOKENS,
        reasoning_effort=config.RELATION_REASONING_EFFORT,
    )

    accepted_claims = []
    rejected_claims = []
    schema_text = schema.prompt_text()
    max_claims_per_chunk = getattr(config, "RELATION_MAX_CLAIMS_PER_CHUNK", 12)
    if max_claims_per_chunk is not None:
        max_claims_per_chunk = int(max_claims_per_chunk)

    for chunk in chunks:
        terms_for_chunk = select_terms_for_chunk(
            candidate_terms,
            chunk.text,
            max_terms=getattr(config, "RELATION_MAX_CANDIDATE_TERMS", 80),
        )
        prompt = build_extraction_prompt(
            schema_text,
            chunk,
            terms_for_chunk,
            max_claims=max_claims_per_chunk if max_claims_per_chunk is not None else 12,
        )
        raw_claims = llm_client.extract_claims(prompt, image_path=getattr(chunk, "image_path", None))
        chunk_accepted_claims = []
        for raw_claim in raw_claims:
            claim = RelationClaim.from_mapping(raw_claim, source=chunk.source_metadata())
            errors = _validate_claim(claim, schema, chunk, config, candidate_term_index)
            if errors:
                rejected_claims.append({"claim": claim.to_dict(), "errors": errors})
            else:
                chunk_accepted_claims.append(claim)
        kept_claims, dropped_claims = _limit_claims_for_chunk(chunk_accepted_claims, max_claims_per_chunk)
        accepted_claims.extend(kept_claims)
        for dropped_claim in dropped_claims:
            rejected_claims.append(
                {
                    "claim": dropped_claim.to_dict(),
                    "errors": [f"exceeds RELATION_MAX_CLAIMS_PER_CHUNK={max_claims_per_chunk}"],
                }
            )

    if getattr(config, "RELATION_MERGE_DUPLICATES", True):
        accepted_claims = _merge_duplicate_claims(accepted_claims)

    _write_jsonl(Path(config.RELATION_CLAIMS_JSONL), [claim.to_dict() for claim in accepted_claims])
    _write_json(Path(config.RELATION_GRAPH_JSON), build_graph_json(accepted_claims))
    _write_json(output_dir / "rejected_relation_claims.json", {"rejected": rejected_claims})

    summary.update(
        {
            "accepted_claims": len(accepted_claims),
            "rejected_claims": len(rejected_claims),
            "claims_jsonl": str(config.RELATION_CLAIMS_JSONL),
            "graph_json": str(config.RELATION_GRAPH_JSON),
        }
    )
    _write_json(output_dir / "run_summary.json", summary)
    record_pipeline_run(
        config.RUN_OUTPUT_DIR,
        "relation_augmentation",
        status="completed",
        inputs={
            "processed_data_dir": str(config.RELATION_PROCESSED_DATA_DIR),
            "termo_dir": str(config.RELATION_TERMO_DIR),
            "input_files": [str(path) for path in input_files],
        },
        outputs={
            "claims_jsonl": str(config.RELATION_CLAIMS_JSONL),
            "graph_json": str(config.RELATION_GRAPH_JSON),
            "summary": str(output_dir / "run_summary.json"),
        },
        extra=summary,
    )
    logger.info("Relation augmentation finished: %s accepted, %s rejected.", len(accepted_claims), len(rejected_claims))
    return summary


def build_graph_json(claims: list[RelationClaim]) -> dict[str, Any]:
    nodes = {}
    edges = []
    for claim in claims:
        nodes.setdefault(claim.subject, {"id": claim.subject, "type": claim.subject_type})
        nodes.setdefault(claim.object, {"id": claim.object, "type": claim.object_type})
        edges.append(
            {
                "source": claim.subject,
                "target": claim.object,
                "relation": claim.relation,
                "confidence": claim.confidence,
                "evidence_quote": claim.evidence_quote,
                "qualifiers": claim.qualifiers,
                "provenance": claim.source,
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def _select_chunks_for_paper(chunks: list[Any], max_chunks: int | None) -> list[Any]:
    if not max_chunks or len(chunks) <= max_chunks:
        return chunks

    section_priority = {
        "results": 0,
        "discussion": 1,
        "ablation": 1,
        "evaluation": 1,
        "experimental": 2,
        "experiments": 2,
        "methods": 2,
        "conclusion": 3,
    }

    def score(chunk: Any) -> tuple[int, int, int]:
        table_bonus = -1 if chunk.source_type == "table" else 0
        return (
            section_priority.get(chunk.section, 9) + table_bonus,
            0 if chunk.source_type == "table" else 1,
            len(chunk.text),
        )

    selected = sorted(chunks, key=score)[:max_chunks]
    return sorted(selected, key=lambda chunk: chunk.chunk_id)


def _validate_claim(
    claim: RelationClaim,
    schema: RelationSchema,
    chunk: Any,
    config,
    candidate_term_index: set[str],
) -> list[str]:
    errors = schema.validate(claim)
    if claim.confidence < getattr(config, "RELATION_MIN_CONFIDENCE", 0.65):
        errors.append(f"confidence below threshold: {claim.confidence}")
    if getattr(config, "RELATION_REQUIRE_EVIDENCE_QUOTE", True):
        quote_found = quote_in_text(claim.evidence_quote, chunk.text)
        image_quote_allowed = (
            chunk.source_type == "hysteresis_figure_image"
            and not getattr(config, "RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS", False)
            and bool(claim.evidence_quote)
        )
        if not quote_found and not image_quote_allowed:
            errors.append("evidence_quote not found in chunk")
    if getattr(config, "RELATION_REJECT_GENERIC_ENTITIES", True):
        if _is_generic_entity_name(claim.subject, claim.subject_type):
            errors.append("subject is too generic")
        if _is_generic_entity_name(claim.object, claim.object_type):
            errors.append("object is too generic")
    if not getattr(config, "RELATION_ALLOW_ENTITY_OUTSIDE_TERMO", True):
        generated_types = {"Experiment", "Result", "QuantityValue", "Condition", "ExperimentalSetting"}
        if claim.subject_type not in generated_types and claim.subject.lower() not in candidate_term_index:
            errors.append("subject is outside TERMO candidate terms")
        if claim.object_type not in generated_types and claim.object.lower() not in candidate_term_index:
            errors.append("object is outside TERMO candidate terms")
    return errors


def _limit_claims_for_chunk(
    claims: list[RelationClaim],
    max_claims: int | None,
) -> tuple[list[RelationClaim], list[RelationClaim]]:
    if max_claims is None or len(claims) <= max_claims:
        return claims, []
    if max_claims <= 0:
        return [], claims

    ranked = sorted(
        enumerate(claims),
        key=lambda item: (
            -_claim_relation_priority(item[1]),
            -item[1].confidence,
            item[0],
        ),
    )
    kept_indices = {index for index, _ in ranked[:max_claims]}
    kept = [claim for index, claim in enumerate(claims) if index in kept_indices]
    dropped = [claim for index, claim in enumerate(claims) if index not in kept_indices]
    return kept, dropped


def _claim_relation_priority(claim: RelationClaim) -> int:
    priority = {
        "REPORTS_RESULT": 5,
        "HAS_VALUE": 5,
        "INDICATES": 5,
        "HAS_CONDITION": 4,
        "MEASURES_METRIC": 3,
        "HAS_PROPERTY": 3,
        "SHOWS": 3,
        "REPRESENTS": 2,
        "USES_METHOD": 2,
        "USES_MATERIAL": 1,
        "EVIDENCED_BY": 1,
    }
    return priority.get(claim.relation, 0)


def _is_generic_entity_name(value: str, entity_type: str | None = None) -> bool:
    normalized = " ".join(value.lower().split())
    generic_names = {
        "experimental",
        "experiment",
        "experiments",
        "method",
        "methods",
        "measurement",
        "measurements",
        "result",
        "results",
        "discussion",
        "study",
        "sample",
        "samples",
        "analysis",
        "data",
    }
    if normalized in generic_names:
        return True

    if entity_type == "Experiment":
        words = normalized.split()
        generic_event_terms = {
            "analysis",
            "calculation",
            "calculations",
            "characterization",
            "comparison",
            "experiment",
            "experiments",
            "measurement",
            "measurements",
            "preparation",
            "simulation",
            "study",
            "test",
            "tests",
        }
        if len(words) <= 3 and any(term in words for term in generic_event_terms):
            return True
    return False


def _merge_duplicate_claims(claims: list[RelationClaim]) -> list[RelationClaim]:
    merged = {}
    for claim in claims:
        key = claim.dedupe_key()
        previous = merged.get(key)
        if previous is None or claim.confidence > previous.confidence:
            merged[key] = claim
    return list(merged.values())


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
