"""QA extractor settings bridged to ../config.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ONTOLOGYM_ROOT = Path(__file__).resolve().parents[1]
if str(ONTOLOGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(ONTOLOGYM_ROOT))

from common.project_config import load_env_file, load_project_config


_project = load_project_config()
load_env_file(getattr(_project, "ENV_FILE", ONTOLOGYM_ROOT / ".env"))


def _env_or_default(name: str, default):
    value = os.getenv(name)
    return default if value in (None, "") else value


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ONTOLOGYM_ROOT
RUN_OUTPUT_DIR = getattr(_project, "RUN_OUTPUT_DIR", ONTOLOGYM_ROOT / "data")
DATA_DIR = getattr(_project, "QA_DATA_DIR", ONTOLOGYM_ROOT / "data" / "airqa")
PHASE = getattr(_project, "QA_PHASE", "")
PAPER_DIR = getattr(_project, "SHARED_PAPER_DIR", ONTOLOGYM_ROOT / "data" / "papers")
PROCESSED_DATA_DIR = getattr(_project, "QA_PROCESSED_DATA_DIR", DATA_DIR / "processed_data")
METADATA_DIR = getattr(_project, "QA_METADATA_DIR", DATA_DIR / "metadata")
EXAMPLE_DIR = getattr(_project, "QA_EXAMPLE_DIR", DATA_DIR / "examples")
OUTPUT_DATASET_PATH = getattr(_project, "QA_OUTPUT_DATASET_PATH", DATA_DIR / "qa_dataset.jsonl")
LOG_DIR = getattr(_project, "QA_LOG_DIR", ONTOLOGYM_ROOT / "logs" / "qa_extractor")
TMP_DIR = getattr(_project, "QA_TMP_DIR", ONTOLOGYM_ROOT / "tmp" / "qa_extractor")
EVALUATIONS_FILE = ROOT_DIR / "evaluation" / "evaluations.json"

MAX_FAILURES = getattr(_project, "QA_MAX_FAILURES", 50)
EXPLORE_FUNC = getattr(_project, "QA_EXPLORE_FUNC", None)
CONTEXT_MAX_CHARS = int(getattr(_project, "QA_CONTEXT_MAX_CHARS", 900))
REASONING_MAX_CHARS = int(getattr(_project, "QA_REASONING_MAX_CHARS", 500))
REASONING_MAX_STEPS = int(getattr(_project, "QA_REASONING_MAX_STEPS", 3))
ENABLE_QUALITY_FILTER = bool(getattr(_project, "QA_ENABLE_QUALITY_FILTER", True))
ANSWER_LEAKAGE_TOKEN_THRESHOLD = float(getattr(_project, "QA_ANSWER_LEAKAGE_TOKEN_THRESHOLD", 0.72))
ANSWER_LEAKAGE_NUMERIC_THRESHOLD = float(getattr(_project, "QA_ANSWER_LEAKAGE_NUMERIC_THRESHOLD", 0.80))
MIN_SECTION_TEXT_CHARS = int(getattr(_project, "QA_MIN_SECTION_TEXT_CHARS", 260))
MIN_SECTION_WORDS = int(getattr(_project, "QA_MIN_SECTION_WORDS", 45))
REJECT_NEGATIVE_EVIDENCE_QUESTIONS = bool(getattr(_project, "QA_REJECT_NEGATIVE_EVIDENCE_QUESTIONS", True))
NEAR_DUPLICATE_TOKEN_OVERLAP_THRESHOLD = float(getattr(_project, "QA_NEAR_DUPLICATE_TOKEN_OVERLAP_THRESHOLD", 0.64))
NEAR_DUPLICATE_REFERENCE_DIRS = list(getattr(_project, "QA_NEAR_DUPLICATE_REFERENCE_DIRS", []))
MAX_QUESTION_CHARS = int(getattr(_project, "QA_MAX_QUESTION_CHARS", 560))
MAX_ANSWER_FORMAT_CHARS = int(getattr(_project, "QA_MAX_ANSWER_FORMAT_CHARS", 720))
MAX_ANSWER_CHARS = int(getattr(_project, "QA_MAX_ANSWER_CHARS", 720))
GENERATION_MODE = getattr(_project, "QA_GENERATION_MODE", "counts")
PER_PAPER_EXAMPLE_COUNT = int(getattr(_project, "QA_PER_PAPER_EXAMPLE_COUNT", 4))
MAX_PAPERS = getattr(_project, "QA_MAX_PAPERS", None)
RANDOM_SEED = getattr(_project, "QA_RANDOM_SEED", None)
PER_PAPER_FOCUS_PLAN = list(getattr(_project, "QA_PER_PAPER_FOCUS_PLAN", []))
FOCUS_HINTS = dict(getattr(_project, "QA_FOCUS_HINTS", {}))
EXCLUDED_SECTIONS = list(getattr(_project, "QA_EXCLUDED_SECTIONS", []))
INCLUDED_SECTIONS = list(getattr(_project, "QA_INCLUDED_SECTIONS", []))
FOCUS_SECTION_KEYWORDS = dict(getattr(_project, "QA_FOCUS_SECTION_KEYWORDS", {}))
BALANCED_TOTAL_EXAMPLES = getattr(_project, "QA_BALANCED_TOTAL_EXAMPLES", None)
QUESTION_TYPE_TARGET_RATIO = dict(getattr(_project, "QA_QUESTION_TYPE_TARGET_RATIO", {
    "single": 0.5,
    "multi": 0.5,
    "rag": 0.0,
    "comprehensive": 0.0,
}))
EVAL_TYPE_TARGET_RATIO = dict(getattr(_project, "QA_EVAL_TYPE_TARGET_RATIO", {
    "objective": 0.7,
    "subjective": 0.3,
}))
EVAL_BALANCE_MODE = getattr(_project, "QA_EVAL_BALANCE_MODE", "soft")
EVAL_PREFERENCE_HINTS = dict(getattr(_project, "QA_EVAL_PREFERENCE_HINTS", {}))
ENABLE_FIGURE_QA = bool(getattr(_project, "QA_ENABLE_FIGURE_QA", False))
FIGURE_QA_PER_PAPER_COUNT = int(getattr(_project, "QA_FIGURE_QA_PER_PAPER_COUNT", 1))
FIGURE_QA_FOCUS_HINT = getattr(
    _project,
    "QA_FIGURE_QA_FOCUS_HINT",
    "Focus on information visible in the figure and supported by the figure caption.",
)
FIGURE_QA_EXPLORE_FUNC = getattr(_project, "QA_FIGURE_QA_EXPLORE_FUNC", "single_image")
HYSTERESIS_ASSET_MANIFEST_JSONL = getattr(_project, "HYSTERESIS_ASSET_MANIFEST_JSONL", None)
HYSTERESIS_QA_FOCUS_HINT = getattr(
    _project,
    "HYSTERESIS_QA_FOCUS_HINT",
    "Focus on information visible in a hysteresis-loop figure and supported by its caption.",
)

TYPE_EXAMPLE_COUNTS = getattr(_project, "QA_TYPE_EXAMPLE_COUNTS", {
    "single": 1,
    "multi": 0,
    "rag": 0,
    "comprehensive": 0,
})
TYPE_EXAMPLE_COUNTS = {
    str(question_type).lower(): int(count)
    for question_type, count in dict(TYPE_EXAMPLE_COUNTS).items()
}

DEFAULT_LLM_MODEL = _env_or_default("DEFAULT_LLM_MODEL", getattr(_project, "QA_DEFAULT_LLM_MODEL", "gpt-5-mini"))
DEFAULT_TOP_P = float(_env_or_default("DEFAULT_TOP_P", getattr(_project, "QA_DEFAULT_TOP_P", 0.95)))
DEFAULT_TEMPERATURE = float(
    _env_or_default("DEFAULT_TEMPERATURE", getattr(_project, "QA_DEFAULT_TEMPERATURE", 0.0))
)


def ensure_directories() -> None:
    for directory in [
        DATA_DIR,
        PAPER_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        EXAMPLE_DIR,
        LOG_DIR,
        TMP_DIR,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
