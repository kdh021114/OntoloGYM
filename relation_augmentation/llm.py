from __future__ import annotations

import json
import logging
import os
import re
import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None

from common.usage_logging import log_openai_usage

logger = logging.getLogger(__name__)


def _openai_timeout_seconds() -> float:
    value = os.getenv("ONTOLOGYM_OPENAI_TIMEOUT_SECONDS", "").strip()
    return float(value) if value else 120.0


EXTRACTION_PROMPT_TEMPLATE = """
You extract evidence-grounded knowledge graph relation claims from scientific papers.

Use ONLY the schema below. Do not invent new relation names or entity types.
Do not extract isA/subclass relations.
Extract only compact, high-confidence claims about experimental facts: setup, protocol, used materials or methods, controlled conditions, measured metrics, and reported results or values.
Do not extract general background, applications, broad causal speculation, motivation, challenges, or every detail in the paper.
Do not record a generic capability or topic such as "DFT calculations", "magnetotransport measurements", "sample preparation", or "simulation comparison" unless it is anchored to the concrete material/system, condition, and measured/reported fact in the evidence.
Every claim must be directly supported by the evidence text.
If the source is a hysteresis_figure_image, you must inspect the attached image directly.
If caption text is present in the evidence, use it only as supporting metadata for identifying the figure, samples, and conditions.
For image chunks, prioritize facts that require visual inspection: panel labels, axis labels and
units, legend color/marker mappings, curve-to-sample mappings, annotated numeric values, loop
width/height/squareness comparisons, sweep-direction arrows, and visible second-quadrant or FORC
features. For every claim whose support comes from the attached image rather than literal caption
text, set evidence_quote to exactly "VISIBLE_IN_FIGURE" and add qualifiers.visual_evidence with a
short description of what is visible. Do not create an image-source claim by reading the caption
alone unless the claim is needed to connect a visual observation to the figure identity.
Return at most {max_claims} claims. This is a strict cap, so choose the {max_claims} most important experimental facts in the chunk rather than the first valid facts you notice.
Prefer claims that preserve experimental results and their conditions over routine setup details.

Entity naming rules:
- When the subject_type is Experiment, use a concrete event name made from the evidence, such as "XRR characterization of Pd/Co/Pd/CoO multilayers" or "four-probe magnetotransport measurement of twisted CrSBr bilayer".
- Do not use section headings or generic labels as entities: "experimental", "experiment", "method", "results", "sample", "study", "measurement", "calculation", "comparison".
- Reuse the same concrete Experiment subject across its USES_MATERIAL, USES_METHOD, HAS_CONDITION, MEASURES_METRIC, and REPORTS_RESULT claims.
- Prefer exact material, method, metric, condition, and value names found in the evidence text.
- Include decisive numeric values, units, comparators, and experimental conditions in the object or qualifiers when they are necessary to understand the fact.
- Each triple should remain useful to a downstream QA answerer even if the evidence_quote is hidden; make the subject and object specific enough.
- If a concrete event or entity name cannot be formed, skip the claim.

## Relation Schema
{schema}

## Candidate entities from the existing OntoGen/TERMO output
{candidate_terms}

## Additional extraction guidance
{extra_guidance}

## Evidence
Paper ID: {paper_id}
Source: {source_type}
Section: {section}
Title: {title}

{evidence}

## Output JSON
Return a JSON object with this shape:
{{
  "claims": [
    {{
      "subject": "canonical entity or event name",
      "subject_type": "one schema entity type",
      "relation": "one schema relation name",
      "object": "canonical entity/value/event name",
      "object_type": "one schema entity type",
      "evidence_quote": "short exact quote from the evidence text",
      "confidence": 0.0,
      "qualifiers": {{"optional": "condition, unit, comparator, baseline, time, page, etc."}}
    }}
  ]
}}
If no valid claim is present, return {{"claims": []}}.
"""


class RelationLLMClient:
    def __init__(
        self,
        model: str,
        backend: str,
        temperature: float | None,
        max_completion_tokens: int | None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort

    def extract_claims(self, prompt: str, image_path: str | None = None) -> list[dict[str, Any]]:
        if self.backend not in {"openai", "oai"}:
            raise ValueError(f"Unsupported relation augmentation backend: {self.backend}")
        if OpenAI is None:
            raise ImportError("openai is required for relation augmentation with backend='openai'.")

        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            options["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort

        last_error = None
        for attempt in range(1, 5):
            try:
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    timeout=_openai_timeout_seconds(),
                )
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[_build_user_message(prompt, image_path=image_path)],
                    response_format={"type": "json_object"},
                    **options,
                )
                log_openai_usage(completion, component="relation_augmentation")
                content = completion.choices[0].message.content or "{}"
                break
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    raise
                wait_seconds = 3 * attempt
                logger.warning("Relation augmentation LLM call failed on attempt %s/4: %s", attempt, exc)
                time.sleep(wait_seconds)
        else:
            raise last_error  # pragma: no cover
        data = _loads_json_object(content)
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            return []
        return [claim for claim in claims if isinstance(claim, dict)]


def _loads_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    for candidate in _json_candidates(cleaned):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    logger.warning("Skipping invalid relation JSON response: %s", cleaned[:500])
    return {"claims": []}


def _json_candidates(content: str) -> list[str]:
    candidates = [content]
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])
    repaired = re.sub(r",\s*([}\]])", r"\1", candidates[-1])
    candidates.append(repaired)
    return candidates


def build_extraction_prompt(schema_text: str, chunk, candidate_terms: list[str], max_claims: int = 12) -> str:
    terms_text = "\n".join(f"- {term}" for term in candidate_terms) if candidate_terms else "(none)"
    metadata = getattr(chunk, "metadata", {}) if isinstance(getattr(chunk, "metadata", {}), dict) else {}
    extra_guidance = metadata.get("prompt_context") or "(none)"
    return EXTRACTION_PROMPT_TEMPLATE.format(
        schema=schema_text,
        max_claims=max_claims,
        candidate_terms=terms_text,
        extra_guidance=extra_guidance,
        paper_id=chunk.paper_id,
        source_type=chunk.source_type,
        section=chunk.section,
        title=chunk.title,
        evidence=chunk.text,
    )


def _build_user_message(prompt: str, image_path: str | None = None) -> dict[str, Any]:
    if not image_path:
        return {"role": "user", "content": prompt}
    path = Path(image_path)
    if not path.exists():
        logger.warning("Hysteresis image path does not exist; falling back to text-only prompt: %s", image_path)
        return {"role": "user", "content": prompt}
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            },
        ],
    }
