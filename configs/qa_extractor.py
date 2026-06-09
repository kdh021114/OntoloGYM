"""AirQA 기반 QA 데이터셋 생성 파이프라인 설정."""

import os

from configs.common import RUN_OUTPUT_DIR


QA_PHASE = os.getenv("ONTOLOGYM_QA_PHASE", "").strip()
QA_DATA_DIR = RUN_OUTPUT_DIR / ("qa_extractor" if not QA_PHASE else f"qa_extractor_{QA_PHASE}")
QA_METADATA_DIR = QA_DATA_DIR / "metadata"
QA_PROCESSED_DATA_DIR = QA_DATA_DIR / "processed_data"
QA_EXAMPLE_DIR = QA_DATA_DIR / "examples"
QA_OUTPUT_DATASET_PATH = QA_DATA_DIR / "qa_dataset.jsonl"
QA_LOG_DIR = QA_DATA_DIR / "logs"
QA_TMP_DIR = RUN_OUTPUT_DIR / "tmp" / "qa_extractor"

# "balanced": 아래 목표 비율에 맞춰 single/multi와 objective/subjective를 함께 조절합니다.
# "per_paper": 각 논문마다 QA_PER_PAPER_EXAMPLE_COUNT개를 생성합니다.
# "counts": 아래 QA_TYPE_EXAMPLE_COUNTS 총량만큼 기존 AirQA 방식으로 생성합니다.
QA_GENERATION_MODE = "balanced"
QA_PER_PAPER_EXAMPLE_COUNT = 4
QA_MAX_PAPERS = None
QA_RANDOM_SEED = int(os.getenv("ONTOLOGYM_QA_RANDOM_SEED", "42"))

# balanced 모드에서 최종 데이터셋의 문제 유형 비율입니다.
# multi는 내부적으로 single 하위문항 2개를 필요로 하므로, 최종 데이터셋 개수보다
# 더 많은 single 후보를 먼저 만든 뒤 multi에 사용된 하위문항은 최종 JSONL에서 제외합니다.
QA_BALANCED_TOTAL_EXAMPLES = 80
QA_QUESTION_TYPE_TARGET_RATIO = {
    "single": 1.0,
    "multi": 0.0,
    "rag": 0.0,
    "comprehensive": 0.0,
}

# balanced 모드에서 최종 데이터셋의 evaluator 타입 비율입니다.
# 이 값은 tracker에게 라벨을 강제하는 설정이 아니라, 후보 생성/선별의 목표 비율입니다.
QA_EVAL_TYPE_TARGET_RATIO = {
    "objective": 0.7,
    "subjective": 0.3,
}

# "soft": objective/subjective에 어울리는 질문을 유도하되 tracker가 자연스러운 evaluator를 고릅니다.
# "off": evaluator 타입 균형을 따로 맞추지 않습니다.
QA_EVAL_BALANCE_MODE = "soft"

QA_EVAL_PREFERENCE_HINTS = {
    "objective": (
        "Prefer a question whose scoring point is naturally objective: an exact number, "
        "short material/method/metric name, boolean, or compact Python list/dict. "
        "Do not make it trivial; it must still require reading the paper context."
    ),
    "subjective": (
        "Prefer a concise explanation, mechanism, or comparison where several phrasings "
        "could be correct. Keep the answer grounded in one or two explicit evidence points."
    ),
}

# 논문별 4문제가 한쪽으로 쏠리지 않도록 반복 적용할 focus plan입니다.
# 현재는 single_text 생성에 적용되며, 각 focus는 prompt와 section 선택에 반영됩니다.
QA_PER_PAPER_FOCUS_PLAN = [
    "experimental_setup",
    "reported_result",
    "isa_concept",
    "mechanism_or_comparison",
]

QA_FOCUS_HINTS = {
    "experimental_setup": (
        "Focus on experimental setup, sample preparation, measurement protocol, "
        "materials, instruments, temperatures, fields, pressures, or other conditions."
    ),
    "reported_result": (
        "Focus on reported results, observed trends, measured values, performance, "
        "or direct comparisons between conditions or samples."
    ),
    "isa_concept": (
        "Focus on a concept, material, method, metric, component, or phenomenon that can "
        "support an ontology-style isA relation. Ask what kind of thing it is or how it is categorized."
    ),
    "mechanism_or_comparison": (
        "Focus on a concise mechanistic explanation, causal interpretation, or comparison "
        "that is explicitly supported by the section."
    ),
}

# 텍스트 기반 QA는 abstract/introduction의 요약 문장보다 실험, 방법, 결과 section에서 만들도록 제한합니다.
# section title 기준 필터이며, figure/table QA에는 적용하지 않습니다.
QA_EXCLUDED_SECTIONS = ["abstract", "introduction", "background", "related work"]
QA_INCLUDED_SECTIONS = [
    "method",
    "materials",
    "experimental",
    "experiment",
    "measurement",
    "fabrication",
    "synthesis",
    "preparation",
    "characterization",
    "evaluation",
    "result",
    "discussion",
    "performance",
    "analysis",
]

# focus별로 우선 고를 section title 키워드입니다. 위 included/excluded 필터를 통과한 section 안에서만 적용됩니다.
QA_FOCUS_SECTION_KEYWORDS = {
    "experimental_setup": [
        "experimental",
        "experiment",
        "method",
        "materials",
        "measurement",
        "sample",
        "fabrication",
        "synthesis",
        "preparation",
    ],
    "reported_result": [
        "result",
        "discussion",
        "performance",
        "characterization",
        "magnetic",
        "electrochemical",
        "evaluation",
    ],
    "isa_concept": [
        "method",
        "materials",
        "experimental",
        "measurement",
        "characterization",
        "result",
        "discussion",
    ],
    "mechanism_or_comparison": [
        "result",
        "discussion",
        "mechanism",
        "analysis",
        "comparison",
        "performance",
    ],
}

# figure 이미지 기반 QA를 생성할지 여부입니다.
# True로 두면 per_paper 모드에서 QA_PER_PAPER_EXAMPLE_COUNT 안의 일부를 figure 이미지 문제로 만듭니다.
# 예: QA_PER_PAPER_EXAMPLE_COUNT=4, QA_FIGURE_QA_PER_PAPER_COUNT=1이면 텍스트 3개 + 이미지 1개입니다.
QA_ENABLE_FIGURE_QA = False
QA_FIGURE_QA_PER_PAPER_COUNT = 1
QA_FIGURE_QA_EXPLORE_FUNC = "single_image"
QA_FIGURE_QA_FOCUS_HINT = (
    "Focus on information that is visible in the figure and supported by the figure caption. "
    "Prefer questions about plotted trends, labels, experimental conditions, comparisons, or measured values."
)

# QA 생성 비용과 분량을 조절하는 기본값입니다.
# multi는 기존 single 문제 2개를 조합해서 만들기 때문에 single을 먼저 충분히 만들어야 합니다.
# 만들지 않을 유형은 개수를 0으로 두면 됩니다.
QA_TYPE_EXAMPLE_COUNTS = {
    "single": 1,
    "multi": 0,
    "rag": 0,  # AirQA 원본의 retrieval 유형입니다.
    "comprehensive": 0,
}
QA_MAX_FAILURES = 8

# 이미 파싱된 JSON만 있을 때는 텍스트 섹션 기반 single QA가 가장 안정적이고 저렴합니다.
QA_EXPLORE_FUNC = "single_text"

# 생성된 예시를 사람이 검수할 때 쓰는 보조 정보입니다.
# AirQA 정식 evaluator는 이 필드를 사용하지 않습니다.
QA_CONTEXT_MAX_CHARS = 900
QA_REASONING_MAX_CHARS = 500
QA_REASONING_MAX_STEPS = 3

# =============================================================================
# QA 품질 필터
# =============================================================================

# True이면 저장 직전에 저품질 문항을 버리고 다시 생성합니다.
# 예: 질문/answer_format에 정답이 노출된 경우, title만 있는 섹션에서 만든 "정보 없음" 문제 등.
QA_ENABLE_QUALITY_FILTER = True

# 답변의 핵심 토큰/숫자가 질문 또는 answer_format에 너무 많이 들어 있으면 정답 누출로 봅니다.
QA_ANSWER_LEAKAGE_TOKEN_THRESHOLD = 0.72
QA_ANSWER_LEAKAGE_NUMERIC_THRESHOLD = 0.80

# title/author만 있는 섹션이나 너무 짧은 섹션은 QA 생성 후보에서 제외합니다.
QA_MIN_SECTION_TEXT_CHARS = 260
QA_MIN_SECTION_WORDS = 45

# "정보가 없다", "수치가 없다" 같은 negative-evidence 문항은 KG 비교에 신호가 약해서 제외합니다.
QA_REJECT_NEGATIVE_EVIDENCE_QUESTIONS = True

# base/holdout처럼 같은 논문에서 반복 생성할 때 기존 질문과 너무 비슷하면 제외합니다.
QA_NEAR_DUPLICATE_TOKEN_OVERLAP_THRESHOLD = float(
    os.getenv("ONTOLOGYM_QA_NEAR_DUPLICATE_TOKEN_OVERLAP_THRESHOLD", "0.64")
)
_duplicate_reference_dirs = os.getenv("ONTOLOGYM_QA_NEAR_DUPLICATE_REFERENCE_DIRS", "").strip()
QA_NEAR_DUPLICATE_REFERENCE_DIRS = [
    path for path in _duplicate_reference_dirs.split(os.pathsep) if path
]

# 너무 긴 문항은 eval 비용을 늘리고 채점 기준도 흐려지므로 저장 직전에 제외합니다.
QA_MAX_QUESTION_CHARS = 560
QA_MAX_ANSWER_FORMAT_CHARS = 720
QA_MAX_ANSWER_CHARS = 720

# QA extractor의 기본 LLM 설정입니다. .env의 같은 이름 값이 있으면 그 값이 우선됩니다.
_model_override = os.getenv("ONTOLOGYM_QA_MODEL", os.getenv("ONTOLOGYM_MODEL", "")).strip()
QA_DEFAULT_LLM_MODEL = _model_override or "gpt-5.4-mini"
QA_DEFAULT_TOP_P = 0.95
QA_DEFAULT_TEMPERATURE = 0.0
