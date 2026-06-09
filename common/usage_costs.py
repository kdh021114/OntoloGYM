"""Estimate OpenAI usage cost from OntoloGYM usage logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


USD_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


def estimate_usage_cost(usage_log: str | Path) -> dict[str, Any]:
    usage_log = Path(usage_log)
    summary: dict[str, Any] = {
        "records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "by_model": {},
    }
    if not usage_log.exists():
        return summary

    for line in usage_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        model = str(record.get("model") or "unknown")
        prompt_tokens = int(record.get("prompt_tokens") or 0)
        completion_tokens = int(record.get("completion_tokens") or 0)
        total_tokens = int(record.get("total_tokens") or prompt_tokens + completion_tokens)
        rates = USD_PER_MILLION.get(model, USD_PER_MILLION.get(_base_model_name(model), {"input": 0.0, "output": 0.0}))
        cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000

        summary["records"] += 1
        summary["prompt_tokens"] += prompt_tokens
        summary["completion_tokens"] += completion_tokens
        summary["total_tokens"] += total_tokens
        summary["estimated_cost_usd"] += cost

        model_summary = summary["by_model"].setdefault(
            model,
            {"records": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
        )
        model_summary["records"] += 1
        model_summary["prompt_tokens"] += prompt_tokens
        model_summary["completion_tokens"] += completion_tokens
        model_summary["total_tokens"] += total_tokens
        model_summary["estimated_cost_usd"] += cost

    return summary


def _base_model_name(model: str) -> str:
    for known in USD_PER_MILLION:
        if model.startswith(known):
            return known
    return model
