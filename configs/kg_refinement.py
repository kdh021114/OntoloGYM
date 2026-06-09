"""오답 해설 기반 KG refinement 파이프라인 설정."""

import os
from pathlib import Path

from configs.common import RUN_OUTPUT_DIR
from configs.qa_extractor import QA_OUTPUT_DATASET_PATH
from configs.qa_evaluation import QA_EVAL_RESULTS_JSON


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


KG_REFINE_RUN = True
_kg_refine_output_override = os.getenv("ONTOLOGYM_KG_REFINE_OUTPUT_DIR", "").strip()
KG_REFINE_OUTPUT_DIR = (
    RUN_OUTPUT_DIR / _kg_refine_output_override
    if _kg_refine_output_override and not os.path.isabs(_kg_refine_output_override)
    else (RUN_OUTPUT_DIR / "kg_refinement" if not _kg_refine_output_override else Path(_kg_refine_output_override))
)
KG_REFINE_CLAIMS_JSONL = KG_REFINE_OUTPUT_DIR / "refined_relation_claims.jsonl"
KG_REFINE_GRAPH_JSON = KG_REFINE_OUTPUT_DIR / "refined_relation_graph.json"
KG_REFINE_REPORT_JSON = KG_REFINE_OUTPUT_DIR / "refinement_report.json"

_kg_refine_eval_override = os.getenv("ONTOLOGYM_KG_REFINE_INPUT_EVAL_RESULTS_JSON", "").strip()
_kg_refine_dataset_override = os.getenv("ONTOLOGYM_KG_REFINE_INPUT_DATASET_PATH", "").strip()
KG_REFINE_INPUT_EVAL_RESULTS_JSON = _kg_refine_eval_override or QA_EVAL_RESULTS_JSON
KG_REFINE_INPUT_DATASET_PATH = _kg_refine_dataset_override or QA_OUTPUT_DATASET_PATH

_kg_refine_model_override = os.getenv("ONTOLOGYM_KG_REFINE_MODEL", os.getenv("ONTOLOGYM_MODEL", "")).strip()
KG_REFINE_MODEL = _kg_refine_model_override or "gpt-5.4-nano"
KG_REFINE_BACKEND = "openai"
KG_REFINE_TEMPERATURE = 0.0
KG_REFINE_MAX_COMPLETION_TOKENS = 2048
KG_REFINE_REASONING_EFFORT = None

# score가 이 값보다 낮은 QA를 오답으로 보고 KG patch 후보를 생성합니다.
KG_REFINE_CORRECTNESS_THRESHOLD = _env_float("ONTOLOGYM_KG_REFINE_CORRECTNESS_THRESHOLD", 0.5)
KG_REFINE_MAX_CASES = _env_int("ONTOLOGYM_KG_REFINE_MAX_CASES", 40)

# refine은 오답 QA의 짧은 context/reasoning만 사용해 비용과 치팅 위험을 줄입니다.
KG_REFINE_MAX_CONTEXT_CHARS = _env_int("ONTOLOGYM_KG_REFINE_MAX_CONTEXT_CHARS", 1500)
KG_REFINE_MIN_CONFIDENCE = _env_float("ONTOLOGYM_KG_REFINE_MIN_CONFIDENCE", 0.55)
