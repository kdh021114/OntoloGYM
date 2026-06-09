"""Small helper for recording OpenAI token usage from API responses."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USAGE_LOG = ROOT_DIR / "logs" / "openai_usage.jsonl"


def _usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _serializable_usage_details(usage: Any, key: str) -> dict[str, Any] | None:
    details = _usage_value(usage, key)
    if details is None:
        return None
    if isinstance(details, dict):
        return details
    if hasattr(details, "model_dump"):
        return details.model_dump()
    if hasattr(details, "__dict__"):
        return {
            name: value
            for name, value in vars(details).items()
            if not name.startswith("_")
        }
    return None


def log_openai_usage(response: Any, component: str) -> None:
    """Append token usage for an OpenAI response when the SDK returns it."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "model": getattr(response, "model", None),
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "prompt_tokens_details": _serializable_usage_details(usage, "prompt_tokens_details"),
        "completion_tokens_details": _serializable_usage_details(usage, "completion_tokens_details"),
    }

    usage_log = Path(os.getenv("ONTOLOGYM_USAGE_LOG", os.fspath(DEFAULT_USAGE_LOG)))
    usage_log.parent.mkdir(parents=True, exist_ok=True)
    with usage_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
