from __future__ import annotations

import json
import logging
import os
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency.
    OpenAI = None

from common.usage_logging import log_openai_usage


logger = logging.getLogger(__name__)


def _openai_timeout_seconds() -> float:
    value = os.getenv("ONTOLOGYM_OPENAI_TIMEOUT_SECONDS", "").strip()
    return float(value) if value else 120.0


ANSWER_PROMPT = """
You answer AirQA questions using the provided GraphRAG knowledge graph context.

Rules:
- Use the GraphRAG community reports and supporting graph edges first.
- If the GraphRAG context is insufficient and general knowledge is not allowed, say exactly: [INSUFFICIENT_CONTEXT]
- If strict context grounding is enabled, answer only when the retrieved context directly entails the requested fact.
- In strict mode, taxonomy/isA edges may define terms or categories, but they are not evidence for sample-specific results, visual observations, numeric values, trends, comparisons, mechanisms, or experimental conditions.
- In strict mode, do not infer an increase/decrease, higher/lower comparison, easy/hard axis, magnetic state, coercivity/remanence relation, or sample identity from the question wording, general scientific priors, or common patterns. Those details must be explicitly present in a retrieved non-taxonomy edge, evidence quote, qualifier, community report, or node bundle.
- In strict mode, if the retrieved context only contains generic ontology edges such as "coercivity --isA--> Metric" or "magnetic hysteresis loop --isA--> hysteresis loop", say exactly: [INSUFFICIENT_CONTEXT].
- Before giving any non-[INSUFFICIENT_CONTEXT] answer in strict mode, silently check that at least one retrieved context item supports every specific claim in the answer.
- Follow the requested answer format exactly. For short string, number, list, or structured-object answers, return only the requested value/object and no explanation.
- If the requested answer format is JSON or a Python-like list/dict, make the answer parseable and avoid markdown fences.
- If the answer should be a short label or method name, do not add extra words such as "the answer is".
- Do not cite facts that are not supported by the context unless general knowledge is explicitly allowed.

Question:
{question}

Requested answer format:
{answer_format}

General knowledge allowed:
{allow_without_context}

Strict context grounding enabled:
{strict_context_grounding}

GraphRAG knowledge graph context:
{kg_context}

Answer:
"""


PAIRWISE_JUDGE_PROMPT = """
Given the question and answers below, and given the following criteria:
{criterion}

Which of the two answers is better according to the criteria, "ANSWER1" or "ANSWER2"?
Justify your answer and then answer with "Winner=ANSWER1" or "Winner=ANSWER2".
If both answers are equally good, answer with "Winner=None".

QUESTION: {question}

========================================= ANSWER 1 =========================================

{answer1}

========================================= ANSWER 2 =========================================

{answer2}
"""


class OpenAITextClient:
    def __init__(
        self,
        model: str,
        backend: str,
        temperature: float | None,
        max_completion_tokens: int | None,
        reasoning_effort: str | None = None,
        usage_component: str = "qa_evaluation",
    ) -> None:
        if backend not in {"openai", "oai"}:
            raise ValueError(f"Unsupported QA evaluation backend: {backend}")
        if OpenAI is None:
            raise ImportError("openai is required for QA evaluation with backend='openai'.")
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.usage_component = usage_component

    def complete(self, prompt: str) -> str:
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
                    messages=[{"role": "user", "content": prompt}],
                    **options,
                )
                log_openai_usage(completion, component=self.usage_component)
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    break
                wait_seconds = 3 * attempt
                logger.warning("QA evaluation LLM call failed on attempt %s/4: %s", attempt, exc)
                time.sleep(wait_seconds)
        raise last_error


class OpenAIEmbeddingClient:
    def __init__(
        self,
        model: str,
        backend: str,
        dimensions: int | None = None,
        usage_component: str = "qa_evaluation_graphrag_embeddings",
    ) -> None:
        if backend not in {"openai", "oai"}:
            raise ValueError(f"Unsupported embedding backend: {backend}")
        if OpenAI is None:
            raise ImportError("openai is required for embeddings with backend='openai'.")
        self.model = model
        self.dimensions = dimensions
        self.usage_component = usage_component

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "input": [text if text.strip() else " " for text in texts],
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        last_error = None
        for attempt in range(1, 5):
            try:
                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    timeout=_openai_timeout_seconds(),
                )
                response = client.embeddings.create(**payload)
                log_openai_usage(response, component=self.usage_component)
                return [list(item.embedding) for item in response.data]
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    break
                wait_seconds = 3 * attempt
                logger.warning("Embedding call failed on attempt %s/4: %s", attempt, exc)
                time.sleep(wait_seconds)
        raise last_error


def build_answer_prompt(
    question: str,
    answer_format: str,
    kg_context: str,
    allow_without_context: bool,
    strict_context_grounding: bool = False,
) -> str:
    return ANSWER_PROMPT.format(
        question=question,
        answer_format=answer_format,
        kg_context=kg_context,
        allow_without_context="yes" if allow_without_context else "no",
        strict_context_grounding="yes" if strict_context_grounding else "no",
    )


def build_pairwise_judge_prompt(question: str, answer1: str, answer2: str, criterion: str) -> str:
    return PAIRWISE_JUDGE_PROMPT.format(
        question=question,
        answer1=answer1,
        answer2=answer2,
        criterion=criterion,
    )


def parse_pairwise_winner(answer: str) -> str:
    lower = answer.lower()
    answer1 = "winner=answer1" in lower
    answer2 = "winner=answer2" in lower
    if answer1 and not answer2:
        return "ANSWER1"
    if answer2 and not answer1:
        return "ANSWER2"
    return "None"


def json_dumps(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
