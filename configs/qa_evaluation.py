"""AirQA 데이터셋을 KG context로 풀고 평가하는 파이프라인 설정."""

import os
from pathlib import Path

from configs.common import RUN_OUTPUT_DIR
from configs.ontogen import ONTOGEN_TAXONOMY_DIR, ONTOGEN_TERMO_DIR
from configs.qa_extractor import QA_OUTPUT_DATASET_PATH
from configs.relation_augmentation import RELATION_OUTPUT_DIR


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _env_int_or_none(name: str, default: int | None) -> int | None:
    value = os.getenv(name, "").strip()
    if value == "":
        return default
    if value.lower() in {"none", "null", "off", "false"}:
        return None
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


# =============================================================================
# 1. 실행과 입출력
# =============================================================================

# 이번 실험에서는 KG 답변 성능을 실제로 봐야 하므로 실행합니다.
QA_EVAL_RUN = _env_bool("ONTOLOGYM_QA_EVAL_RUN", True)
QA_EVAL_DRY_RUN = _env_bool("ONTOLOGYM_QA_EVAL_DRY_RUN", False)

_dataset_override = os.getenv("ONTOLOGYM_QA_EVAL_DATASET_PATH", "").strip()
QA_EVAL_DATASET_PATH = Path(_dataset_override) if _dataset_override else QA_OUTPUT_DATASET_PATH
QA_EVAL_PHASE = os.getenv("ONTOLOGYM_QA_EVAL_PHASE", "").strip()
QA_EVAL_DATA_DIR = RUN_OUTPUT_DIR / ("qa_evaluation" if not QA_EVAL_PHASE else f"qa_evaluation_{QA_EVAL_PHASE}")
QA_EVAL_OUTPUT_DIR = QA_EVAL_DATA_DIR / "outputs"
QA_EVAL_ANSWERS_JSONL = QA_EVAL_OUTPUT_DIR / "kg_answers.jsonl"
QA_EVAL_RESULTS_JSON = QA_EVAL_OUTPUT_DIR / "evaluation_results.json"

# KG를 별도 input 폴더에 모아두고 싶을 때 쓰는 경로입니다.
# relation_graph.json, relation_claims.jsonl, *.relationships.csv, taxonomy/*.pkl 등을 넣을 수 있습니다.
QA_EVAL_KG_INPUT_DIR = QA_EVAL_DATA_DIR / "input_kg"
QA_EVAL_KG_CONTEXT_JSON = QA_EVAL_OUTPUT_DIR / "kg_context.json"

# True면 기존 OntoGen/relation augmentation 산출물도 함께 읽습니다.
QA_EVAL_AUTO_INCLUDE_ONTOGEN_OUTPUTS = True
QA_EVAL_EXTRA_KG_DIRS = [
    ONTOGEN_TAXONOMY_DIR,
    ONTOGEN_TERMO_DIR,
    RELATION_OUTPUT_DIR,
]
# OntoGen 논문식 비교를 하려면:
# - KG only: ONTOGEN_TERMO_DIR, RELATION_OUTPUT_DIR만 사용
# - KG + ontology: 위 목록처럼 ONTOGEN_TAXONOMY_DIR의 isA edge까지 함께 사용
_extra_kg_override = os.getenv("ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS", "").strip()
if _extra_kg_override:
    QA_EVAL_EXTRA_KG_DIRS = [Path(path) for path in _extra_kg_override.split(os.pathsep) if path]

# None이면 전체 dataset을 처리합니다.
_max_examples_override = os.getenv("ONTOLOGYM_QA_EVAL_MAX_EXAMPLES", "").strip()
QA_EVAL_MAX_EXAMPLES = int(_max_examples_override) if _max_examples_override else None


# =============================================================================
# 2. 답변 생성 LLM 설정
# =============================================================================

_model_override = os.getenv("ONTOLOGYM_QA_EVAL_MODEL", os.getenv("ONTOLOGYM_MODEL", "")).strip()
QA_EVAL_MODEL = _model_override or "gpt-5.4-nano"
QA_EVAL_BACKEND = "openai"
QA_EVAL_TEMPERATURE = 0.0
QA_EVAL_MAX_COMPLETION_TOKENS = 2048
QA_EVAL_REASONING_EFFORT = None


# =============================================================================
# 3. GraphRAG retrieval 설정
# =============================================================================

# 현재는 community-first retrieval이 노이즈를 만들 수 있어서 supporting edge만 직접 검색합니다.
# community report는 재실험이 필요할 때만 명시적으로 켜세요.
QA_EVAL_GRAPHRAG_CACHE_DIR = RUN_OUTPUT_DIR / "qa_evaluation_graphrag_cache"
QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS = _env_bool("ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS", False)
QA_EVAL_GRAPHRAG_REPORT_MODEL = os.getenv(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_REPORT_MODEL",
    QA_EVAL_MODEL,
).strip() or QA_EVAL_MODEL
QA_EVAL_GRAPHRAG_REPORT_TEMPERATURE = 0.0
QA_EVAL_GRAPHRAG_REPORT_MAX_COMPLETION_TOKENS = 700
QA_EVAL_GRAPHRAG_MAX_COMMUNITIES = 48
QA_EVAL_GRAPHRAG_MAX_NODES_PER_COMMUNITY = int(os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_MAX_NODES_PER_COMMUNITY", "160"))
QA_EVAL_GRAPHRAG_MAX_EDGES_PER_COMMUNITY_REPORT = 80
QA_EVAL_GRAPHRAG_TOP_COMMUNITIES = 0
QA_EVAL_GRAPHRAG_TOP_EDGES = 40
QA_EVAL_MAX_CONTEXT_CHARS = 12000

# 실험/결과 질문에서 isA taxonomy edge가 context를 과도하게 차지하지 않도록 제한합니다.
# 질문이 category/type/ontology 자체를 묻는 경우에는 자동으로 제한이 풀립니다.
QA_EVAL_GRAPHRAG_TAXONOMY_EDGE_FRACTION = _env_float(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_TAXONOMY_EDGE_FRACTION",
    0.18,
)
QA_EVAL_GRAPHRAG_TAXONOMY_COMMUNITY_LIMIT = int(
    os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_TAXONOMY_COMMUNITY_LIMIT", "1")
)

# True면 BM25 lexical search에 embedding semantic search를 섞습니다.
# LLM reranker는 쓰지 않고, question embedding과 KG description embedding의 cosine similarity만 사용합니다.
QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS = _env_bool("ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS", True)
QA_EVAL_GRAPHRAG_EMBEDDING_MODEL = os.getenv(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_MODEL",
    "text-embedding-3-small",
).strip() or "text-embedding-3-small"

# text-embedding-3 계열은 dimensions 축소를 지원합니다. None으로 두면 모델 기본 차원을 씁니다.
QA_EVAL_GRAPHRAG_EMBEDDING_DIMENSIONS = _env_int_or_none(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_DIMENSIONS",
    512,
)
QA_EVAL_GRAPHRAG_EMBEDDING_BATCH_SIZE = int(os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_BATCH_SIZE", "64"))

# Hybrid retrieval 비율입니다. 합이 1이 아니어도 내부에서 정규화됩니다.
QA_EVAL_GRAPHRAG_BM25_WEIGHT = _env_float("ONTOLOGYM_QA_EVAL_GRAPHRAG_BM25_WEIGHT", 0.40)
QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT = _env_float("ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT", 0.60)

# "edge_topk": 기존처럼 edge를 직접 top-k 검색합니다.
# "node_bundle_iterative": BM25 node 후보 1개 + embedding node 후보 1개를 먼저 보여주고,
# answerer가 [INSUFFICIENT_CONTEXT]를 내면 다음 후보를 2개씩 추가합니다.
QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY = os.getenv(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY",
    "node_bundle_iterative",
).strip() or "node_bundle_iterative"

# node bundle retrieval에서는 "후보 node" 개수를 제한합니다. 각 node의 1-hop edge는 bundle로 묶입니다.
QA_EVAL_GRAPHRAG_NODE_BUNDLE_MAX_CANDIDATES = int(
    os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_NODE_BUNDLE_MAX_CANDIDATES", "6")
)
QA_EVAL_GRAPHRAG_NODE_BUNDLE_INITIAL_CANDIDATES = int(
    os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_NODE_BUNDLE_INITIAL_CANDIDATES", "2")
)
QA_EVAL_GRAPHRAG_NODE_BUNDLE_BATCH_SIZE = int(
    os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_NODE_BUNDLE_BATCH_SIZE", "2")
)

# 고차수 node 하나가 context를 전부 잡아먹지 않게 bundle별 edge 수를 제한합니다.
# None으로 두면 max_context_chars가 허용하는 한 1-hop edge를 더 많이 보여줍니다.
QA_EVAL_GRAPHRAG_MAX_EDGES_PER_NODE_BUNDLE = _env_int_or_none(
    "ONTOLOGYM_QA_EVAL_GRAPHRAG_MAX_EDGES_PER_NODE_BUNDLE",
    12,
)

# retrieved node 주변의 isA ancestor path를 몇 단계까지 붙일지 정합니다.
QA_EVAL_GRAPHRAG_TAXONOMY_ANCESTOR_DEPTH = int(
    os.getenv("ONTOLOGYM_QA_EVAL_GRAPHRAG_TAXONOMY_ANCESTOR_DEPTH", "3")
)

# 예전 flat retrieval에서 쓰던 옵션입니다. 현재 GraphRAG는 공정한 corpus-level 평가를 위해
# paper_id를 ranking signal로 쓰지 않고, 출처 표기/provenance 용도로만 남깁니다.
QA_EVAL_PRIORITIZE_QUESTION_PAPERS = False

# AirQA 원칙상 reference_pdf는 정답 출처라 inference 단계에서 숨기는 것이 기본입니다.
# retrieval/comprehensive 실험을 의도적으로 분석할 때만 True로 바꾸세요.
QA_EVAL_INCLUDE_REFERENCE_PAPERS = False

# KG 근거가 부족할 때도 모델이 일반지식으로 답하도록 허용할지 여부입니다.
QA_EVAL_ALLOW_ANSWER_WITHOUT_CONTEXT = False

# True면 answerer가 retrieved KG context에 직접 entail되는 답만 하도록 더 엄격히 요구합니다.
# taxonomy isA edge만 있는 경우에는 실험 결과/수치/증가·감소/비교 판단을 답하지 않아야 합니다.
QA_EVAL_STRICT_CONTEXT_GROUNDING = _env_bool(
    "ONTOLOGYM_QA_EVAL_STRICT_CONTEXT_GROUNDING",
    False,
)


# =============================================================================
# 4. AirQA 정답 evaluator 설정
# =============================================================================

QA_EVAL_RUN_AIRQA_EVALUATOR = _env_bool("ONTOLOGYM_QA_EVAL_RUN_AIRQA_EVALUATOR", False)

# AirQA evaluator 중 일부는 LLM을 다시 호출합니다.
# 비용을 명시적으로 허용하려면 True로 바꿉니다.
QA_EVAL_ALLOW_LLM_EVALUATORS = _env_bool("ONTOLOGYM_QA_EVAL_ALLOW_LLM_EVALUATORS", False)

# nano/mini 답변 성능을 비교할 때 evaluator 모델까지 바뀌면 점수가 흔들립니다.
# 비워두면 기존처럼 QA_EVAL_MODEL을 사용하고, 공정 비교 시에는 고정 judge 모델을 지정하세요.
QA_EVAL_AIRQA_EVALUATOR_MODEL = os.getenv(
    "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL",
    QA_EVAL_MODEL,
).strip() or QA_EVAL_MODEL

# True면 기존 kg_answers.jsonl의 답변은 재생성하지 않고 AirQA score만 다시 계산합니다.
# evaluator prompt/parser를 고친 뒤 재채점할 때 사용합니다.
QA_EVAL_REEVALUATE_EXISTING_SCORES = _env_bool(
    "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES",
    False,
)


# =============================================================================
# 5. OntoGen식 pairwise judge: 선택 사항
# =============================================================================

# baseline 답변 JSONL이 있으면 KG 답변과 비교할 수 있습니다.
# 각 줄은 {"uuid": "...", "answer": "..."} 형식을 권장합니다.
QA_EVAL_BASELINE_ANSWERS_PATH = None
QA_EVAL_RUN_PAIRWISE_JUDGE = False
QA_EVAL_JUDGE_MODEL = os.getenv("ONTOLOGYM_QA_EVAL_JUDGE_MODEL", QA_EVAL_MODEL).strip() or QA_EVAL_MODEL
QA_EVAL_JUDGE_CRITERION = (
    "Directness: How specifically and clearly does the answer address the question?"
)
