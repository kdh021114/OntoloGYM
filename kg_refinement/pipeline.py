from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from common.project_config import load_env_file, load_project_config
from common.run_context import record_pipeline_run
from common.usage_logging import log_openai_usage
from relation_augmentation.llm import _loads_json_object
from relation_augmentation.pipeline import _is_generic_entity_name, build_graph_json
from relation_augmentation.schema import RelationClaim, RelationSchema

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None


logger = logging.getLogger(__name__)


REFINE_PROMPT = """
You refine an ontology-based scientific KG using failed QA evidence.

The model answered the question incorrectly. Add only KG relation claims that are directly supported by the gold QA context/reasoning and would help answer similar future questions.
Use only the schema below. Do not add isA/subclass claims here.
Do not copy the question as a relation. Extract compact evidence-grounded claims about experiments, methods, materials, conditions, metrics, or results.

Entity naming rules:
- When the subject_type is Experiment, use a concrete event name made from the gold context, not a section heading.
- Do not use generic labels as entities: "experimental", "experiment", "method", "results", "sample", "study", "measurement".
- Reuse the same concrete Experiment subject across related claims when possible.
- Prefer exact materials, methods, metrics, conditions, results, and values from the context.
- If the context is too short to support a concrete entity and exact evidence quote, skip the claim.

## Relation Schema
{schema}

## Failed QA Case
Paper IDs: {paper_ids}
Question: {question}
Gold answer: {gold_answer}
Model answer: {model_answer}
Reasoning: {reasoning}
Context:
{context}

## Output JSON
Return a JSON object:
{{
  "claims": [
    {{
      "subject": "canonical entity or experiment name",
      "subject_type": "one schema entity type",
      "relation": "one schema relation name",
      "object": "canonical entity/value/result name",
      "object_type": "one schema entity type",
      "evidence_quote": "short quote from the context",
      "confidence": 0.0,
      "qualifiers": {{"optional": "condition, unit, comparator, time, etc."}}
    }}
  ]
}}
If no useful KG patch is supported, return {{"claims": []}}.
"""


def run_pipeline() -> dict[str, Any]:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))

    if not getattr(config, "KG_REFINE_RUN", False):
        logger.info("KG refinement is disabled.")
        record_pipeline_run(config.RUN_OUTPUT_DIR, "kg_refinement", status="disabled")
        return {"status": "disabled"}

    output_dir = Path(config.KG_REFINE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results = _read_json(Path(config.KG_REFINE_INPUT_EVAL_RESULTS_JSON))
    dataset = _load_jsonl(Path(config.KG_REFINE_INPUT_DATASET_PATH))
    examples_by_uuid = {example.get("uuid"): example for example in dataset}
    wrong_records = _wrong_records(
        eval_results.get("records", []),
        threshold=float(config.KG_REFINE_CORRECTNESS_THRESHOLD),
        max_cases=int(config.KG_REFINE_MAX_CASES),
    )

    schema = RelationSchema(config.RELATION_ENTITY_TYPES, config.RELATION_TYPES)
    client = _RefineClient(
        model=config.KG_REFINE_MODEL,
        backend=config.KG_REFINE_BACKEND,
        temperature=config.KG_REFINE_TEMPERATURE,
        max_completion_tokens=config.KG_REFINE_MAX_COMPLETION_TOKENS,
        reasoning_effort=config.KG_REFINE_REASONING_EFFORT,
    )

    accepted: list[RelationClaim] = []
    rejected = []
    for record in wrong_records:
        example = examples_by_uuid.get(record.get("uuid"), {})
        prompt = _build_prompt(schema.prompt_text(), record, example, config)
        raw_claims = client.extract_claims(prompt)
        source = {
            "paper_id": ",".join(example.get("anchor_pdf", []) or []),
            "qa_uuid": record.get("uuid", ""),
            "source_type": "kg_refinement_failed_qa",
        }
        for raw_claim in raw_claims:
            claim = RelationClaim.from_mapping(raw_claim, source=source)
            errors = schema.validate(claim)
            if claim.confidence < float(config.KG_REFINE_MIN_CONFIDENCE):
                errors.append(f"confidence below threshold: {claim.confidence}")
            if _is_generic_entity_name(claim.subject):
                errors.append("subject is too generic")
            if _is_generic_entity_name(claim.object):
                errors.append("object is too generic")
            if errors:
                rejected.append({"claim": claim.to_dict(), "errors": errors})
            else:
                accepted.append(claim)

    accepted = _merge_claims(accepted)
    _write_jsonl(Path(config.KG_REFINE_CLAIMS_JSONL), [claim.to_dict() for claim in accepted])
    _write_json(Path(config.KG_REFINE_GRAPH_JSON), build_graph_json(accepted))
    report = {
        "status": "completed",
        "input_eval_results": str(config.KG_REFINE_INPUT_EVAL_RESULTS_JSON),
        "input_dataset": str(config.KG_REFINE_INPUT_DATASET_PATH),
        "wrong_records": len(wrong_records),
        "accepted_claims": len(accepted),
        "rejected_claims": len(rejected),
        "claims_jsonl": str(config.KG_REFINE_CLAIMS_JSONL),
        "graph_json": str(config.KG_REFINE_GRAPH_JSON),
        "rejected": rejected,
    }
    _write_json(Path(config.KG_REFINE_REPORT_JSON), report)
    record_pipeline_run(
        config.RUN_OUTPUT_DIR,
        "kg_refinement",
        status="completed",
        inputs={
            "eval_results": str(config.KG_REFINE_INPUT_EVAL_RESULTS_JSON),
            "dataset": str(config.KG_REFINE_INPUT_DATASET_PATH),
        },
        outputs={
            "claims_jsonl": str(config.KG_REFINE_CLAIMS_JSONL),
            "graph_json": str(config.KG_REFINE_GRAPH_JSON),
            "report_json": str(config.KG_REFINE_REPORT_JSON),
        },
        extra={"wrong_records": len(wrong_records), "accepted_claims": len(accepted)},
    )
    return report


class _RefineClient:
    def __init__(
        self,
        model: str,
        backend: str,
        temperature: float | None,
        max_completion_tokens: int | None,
        reasoning_effort: str | None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort

    def extract_claims(self, prompt: str) -> list[dict[str, Any]]:
        if self.backend not in {"openai", "oai"}:
            raise ValueError(f"Unsupported KG refinement backend: {self.backend}")
        if OpenAI is None:
            raise ImportError("openai is required for KG refinement.")

        options = {"response_format": {"type": "json_object"}}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            options["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        log_openai_usage(completion, component="kg_refinement")
        data = _loads_json_object(completion.choices[0].message.content or "{}")
        claims = data.get("claims", [])
        return [claim for claim in claims if isinstance(claim, dict)] if isinstance(claims, list) else []


def _build_prompt(schema_text: str, record: dict[str, Any], example: dict[str, Any], config) -> str:
    context = str(example.get("context", ""))[: int(config.KG_REFINE_MAX_CONTEXT_CHARS)]
    return REFINE_PROMPT.format(
        schema=schema_text,
        paper_ids=", ".join(example.get("anchor_pdf", []) or []),
        question=example.get("question", record.get("question", "")),
        gold_answer=example.get("answer", ""),
        model_answer=record.get("answer", ""),
        reasoning=example.get("reasoning", ""),
        context=context,
    )


def _wrong_records(records: list[dict[str, Any]], threshold: float, max_cases: int) -> list[dict[str, Any]]:
    wrong = []
    for record in records:
        score_info = record.get("airqa_score") if isinstance(record.get("airqa_score"), dict) else {}
        score = score_info.get("score")
        if score is None:
            continue
        try:
            is_wrong = float(score) < threshold
        except (TypeError, ValueError):
            is_wrong = True
        if is_wrong:
            wrong.append(record)
        if len(wrong) >= max_cases:
            break
    return wrong


def _merge_claims(claims: list[RelationClaim]) -> list[RelationClaim]:
    merged = {}
    for claim in claims:
        key = claim.dedupe_key()
        previous = merged.get(key)
        if previous is None or claim.confidence > previous.confidence:
            merged[key] = claim
    return list(merged.values())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
