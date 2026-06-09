"""OntoGen 기반 ontology/isA KG 생성 파이프라인 설정."""

import os

from configs.common import RUN_OUTPUT_DIR


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _env_int_or_none(name: str, default: int | None) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    if value.lower() in {"none", "null", "off", "false"}:
        return None
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


# =============================================================================
# 1. OntoGen 입력과 실행 스위치
# =============================================================================

ONTOGEN_DATA_DIR = RUN_OUTPUT_DIR / "ontogen"
ONTOGEN_PROCESSED_DATA_DIR = ONTOGEN_DATA_DIR / "processed_data"
ONTOGEN_ENRICHED_DIR = ONTOGEN_DATA_DIR / "enriched"
ONTOGEN_TERMO_DIR = ONTOGEN_DATA_DIR / "termo"
ONTOGEN_CATEGORY_DIR = ONTOGEN_DATA_DIR / "categories"
ONTOGEN_TAXONOMY_DIR = ONTOGEN_DATA_DIR / "taxonomy"

# 특정 PDF/JSON만 처리하고 싶으면 파일명 또는 절대경로를 넣습니다.
# 비워두면 SHARED_PAPER_DIR 아래의 PDF와 이미 파싱된 JSON을 재귀적으로 찾습니다.
ONTOGEN_PDF_FILES = []
ONTOGEN_RECURSIVE_PDF_DISCOVERY = True
ONTOGEN_INCLUDE_PARSED_JSON_INPUTS = True

# 단일 논문 점검용으로 가장 싼 모델과 최소 반복을 쓰도록 기본 실행 범위를 좁혀 둡니다.
ONTOGEN_RUN_PROCESSED_DATA_BUILD = True
ONTOGEN_RUN_BUILD_ENRICHED_CONTEXT = True
ONTOGEN_RUN_TERMO = True
ONTOGEN_RUN_CATEGORY_GENERATION = True
ONTOGEN_RUN_TAXONOMY_GENERATION = True

# 같은 run 폴더에서 재실행할 때 이미 만든 processed/enriched/TERMO 산출물을 재사용합니다.
# category나 taxonomy 프롬프트만 수정한 뒤 재실험할 때 토큰을 아끼는 용도입니다.
ONTOGEN_REUSE_INTERMEDIATE_OUTPUTS = _env_bool("ONTOLOGYM_ONTOGEN_REUSE_INTERMEDIATE_OUTPUTS", True)

# 이미 파싱된 JSON을 사용할 때 가장 자주 만질 설정입니다.
# "existing": OntoGen 내부 산출물을 사용
# "airqa_mineru": AirQA/MinerU 스타일 JSON을 OntoGen 형식으로 정규화
# "auto": AirQA/MinerU JSON이 있으면 사용하고, 없으면 기존 OntoGen 산출물 사용
ONTOGEN_PROCESSED_DATA_SOURCE = "auto"
ONTOGEN_USE_PROCESSED_DATA_FOR_ENRICHED_CONTEXT = True
ONTOGEN_INCLUDE_FIGURE_CAPTIONS_IN_ENRICHED_CONTEXT = False
ONTOGEN_INCLUDE_EQUATIONS_IN_ENRICHED_CONTEXT = False
# 리뷰 논문처럼 섹션이 매우 길면 nano 실험에서 TERMO 호출이 폭증합니다.
# 논문당 enriched context를 이 길이로 자르고, 잘린 사실을 파일 끝에 표시합니다.
ONTOGEN_ENRICHED_MAX_CHARS_PER_PAPER = 60000

_model_override = os.getenv("ONTOLOGYM_ONTOGEN_MODEL", os.getenv("ONTOLOGYM_MODEL", "")).strip()
_default_model = _model_override or "gpt-5.4-nano"


# =============================================================================
# 2. TERMO: 용어/정의/관계 추출
# =============================================================================

# TERMO 전체를 켰을 때, 아래 세부 단계 중 필요한 것만 실행할 수 있습니다.
ONTOGEN_RUN_TERMO_TERMS = True
ONTOGEN_RUN_TERMO_ACRONYMS = False
ONTOGEN_RUN_TERMO_DEFINITIONS = False
ONTOGEN_RUN_TERMO_RELATIONSHIPS = False

# 단계별 모델을 다르게 둘 수 있습니다.
# 비교적 단순한 용어/약어/정의는 nano, 관계 추출은 mini를 기본값으로 두었습니다.
ONTOGEN_TERMO_TERMS_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_TERMO_TERMS_MODEL", _default_model)
ONTOGEN_TERMO_ACRONYMS_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_TERMO_ACRONYMS_MODEL", _default_model)
ONTOGEN_TERMO_DEFINITIONS_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_TERMO_DEFINITIONS_MODEL", _default_model)
ONTOGEN_TERMO_RELATIONSHIPS_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_TERMO_RELATIONSHIPS_MODEL", _default_model)

ONTOGEN_TERMO_TERMS_BACKEND = "openai"
ONTOGEN_TERMO_ACRONYMS_BACKEND = "openai"
ONTOGEN_TERMO_DEFINITIONS_BACKEND = "openai"
ONTOGEN_TERMO_RELATIONSHIPS_BACKEND = "openai"

# 단계별 최대 출력 토큰입니다. 비용 상한을 잡는 용도라서 노출해 둡니다.
ONTOGEN_TERMO_TERMS_MAX_COMPLETION_TOKENS = 2048
ONTOGEN_TERMO_ACRONYMS_MAX_COMPLETION_TOKENS = 2048
ONTOGEN_TERMO_DEFINITIONS_MAX_COMPLETION_TOKENS = 2048
ONTOGEN_TERMO_RELATIONSHIPS_MAX_COMPLETION_TOKENS = 2048

# 입력 텍스트를 LLM에 나누어 보낼 때의 문자 단위 길이입니다.
ONTOGEN_TERMO_MAX_LENGTH_SPLIT_TERMS = 6000
ONTOGEN_TERMO_MAX_LENGTH_SPLIT_ACRONYMS = 6000
ONTOGEN_TERMO_MAX_LENGTH_SPLIT_DEFINITIONS = 10000
ONTOGEN_TERMO_MAX_LENGTH_SPLIT_RELATIONSHIPS = 10000

# enriched context와 표 기반 evidence를 사용하도록 둔 기본값입니다.
ONTOGEN_CONTEXT_KIND = "enriched"
ONTOGEN_RELATIONSHIP_MODE = "evidence"
ONTOGEN_INCLUDE_LITERAL_VALUES = True


# =============================================================================
# 3. 카테고리 생성
# =============================================================================

ONTOGEN_DOMAIN_NAME = "Scientific Literature"
ONTOGEN_EVIDENCE_AWARE_CATEGORIES = True

# category generation은 seed 생성, 형식 정리, seed 합성으로 나뉩니다.
ONTOGEN_CATEGORY_GENERATION_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_CATEGORY_GENERATION_MODEL", _default_model)
ONTOGEN_CATEGORY_FORMAT_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_CATEGORY_FORMAT_MODEL", _default_model)
ONTOGEN_CATEGORY_SYNTHESIS_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_CATEGORY_SYNTHESIS_MODEL", _default_model)

ONTOGEN_CATEGORY_GENERATION_BACKEND = "openai"
ONTOGEN_CATEGORY_FORMAT_BACKEND = "openai"
ONTOGEN_CATEGORY_SYNTHESIS_BACKEND = "openai"

ONTOGEN_CATEGORY_GENERATION_MAX_COMPLETION_TOKENS = 2048
ONTOGEN_CATEGORY_FORMAT_MAX_COMPLETION_TOKENS = 2048
ONTOGEN_CATEGORY_SYNTHESIS_MAX_COMPLETION_TOKENS = 2048

# 20편 전체 context를 한 번에 넣을 때 비용과 context 길이가 커지는 것을 막는 안전장치입니다.
ONTOGEN_CATEGORY_MAX_CHARS_PER_FILE = _env_int_or_none("ONTOLOGYM_ONTOGEN_CATEGORY_MAX_CHARS_PER_FILE", 6000)
ONTOGEN_CATEGORY_MAX_TOTAL_CHARS = _env_int_or_none("ONTOLOGYM_ONTOGEN_CATEGORY_MAX_TOTAL_CHARS", 60000)

# 값이 클수록 더 다양한 category seed를 만들지만 비용도 함께 늘어납니다.
ONTOGEN_CATEGORY_NUM_RETRIES = _env_int_or_none("ONTOLOGYM_ONTOGEN_CATEGORY_NUM_RETRIES", 1)
ONTOGEN_CATEGORY_NUM_GENERATED_SEED = _env_int_or_none("ONTOLOGYM_ONTOGEN_CATEGORY_NUM_GENERATED_SEED", 1)

# OntoGen 논문은 LLM이 만든 131개 candidate category를 사람이 12개로 줄였습니다.
# 기본값은 그 비율(12/131)을 따라 LLM이 자동으로 최종 category 수를 압축하게 합니다.
ONTOGEN_CATEGORY_RUN_LLM_CURATION = _env_bool("ONTOLOGYM_ONTOGEN_CATEGORY_RUN_LLM_CURATION", True)
ONTOGEN_CATEGORY_CURATION_REFERENCE_SOURCE_COUNT = _env_int_or_none(
    "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_REFERENCE_SOURCE_COUNT",
    131,
)
ONTOGEN_CATEGORY_CURATION_REFERENCE_TARGET_COUNT = _env_int_or_none(
    "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_REFERENCE_TARGET_COUNT",
    12,
)
ONTOGEN_CATEGORY_CURATION_RATIO = (
    ONTOGEN_CATEGORY_CURATION_REFERENCE_TARGET_COUNT / ONTOGEN_CATEGORY_CURATION_REFERENCE_SOURCE_COUNT
)
# 숫자를 직접 고정하고 싶으면 정수를 넣습니다. None이면 위 비율을 사용합니다.
ONTOGEN_CATEGORY_CURATION_TARGET_COUNT = _env_int_or_none("ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_TARGET_COUNT", None)
# 작은 실험셋에서는 위 비율을 그대로 곱하면 category가 1개로 줄어들 수 있습니다.
# 최소 개수를 둬서 Method/Task/Setting/Result 같은 큰 축이 사라지지 않게 합니다.
ONTOGEN_CATEGORY_CURATION_MIN_TARGET_COUNT = _env_int_or_none(
    "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MIN_TARGET_COUNT",
    8,
)
ONTOGEN_CATEGORY_CURATION_PROTECTED_CATEGORIES = [
    "Method",
    "Task",
    "Metric",
    "Experimental Setting",
    "Result",
    "Material / Entity",
    "Component",
]
ONTOGEN_CATEGORY_CURATION_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MODEL", _default_model)
ONTOGEN_CATEGORY_CURATION_BACKEND = "openai"
ONTOGEN_CATEGORY_CURATION_MAX_COMPLETION_TOKENS = 2048


# =============================================================================
# 4. Taxonomy 생성
# =============================================================================

# category 생성을 건너뛰고 직접 seed 파일을 쓰려면 경로를 지정합니다.
ONTOGEN_CATEGORY_SEED_FILE = None

ONTOGEN_TAXONOMY_MODEL = os.getenv("ONTOLOGYM_ONTOGEN_TAXONOMY_MODEL", _default_model)
ONTOGEN_TAXONOMY_BACKEND = "openai"
ONTOGEN_TAXONOMY_MAX_COMPLETION_TOKENS = 2048

# 반복 횟수와 self-consistency 재시도 횟수입니다. 비용에 직접 영향을 줍니다.
ONTOGEN_TAXONOMY_NUM_ITERATIONS = _env_int_or_none("ONTOLOGYM_ONTOGEN_TAXONOMY_NUM_ITERATIONS", 1)
ONTOGEN_TAXONOMY_SC_RETRY = _env_int_or_none("ONTOLOGYM_ONTOGEN_TAXONOMY_SC_RETRY", 1)
ONTOGEN_TAXONOMY_MAX_TERMS_PER_PAPER = _env_int_or_none("ONTOLOGYM_ONTOGEN_TAXONOMY_MAX_TERMS_PER_PAPER", 90)
ONTOGEN_TAXONOMY_MAX_TERM_CHARS = _env_int_or_none("ONTOLOGYM_ONTOGEN_TAXONOMY_MAX_TERM_CHARS", 120)

ONTOGEN_LOG_LEVEL = "INFO"


# =============================================================================
# 5. PDF/텍스트 파싱 관련 설정: 보통은 안 건드려도 됨
# =============================================================================

# 이미 파싱된 JSON을 사용할 예정이면 아래 실행 스위치는 False로 둡니다.
ONTOGEN_RUN_PDF_TO_TEXT = False
ONTOGEN_RUN_SECTION_EXTRACTION = False
ONTOGEN_RUN_TABLE_EXTRACTION = False

# raw PDF에서 텍스트/섹션/표를 직접 뽑아야 할 때만 쓰는 산출물 경로입니다.
ONTOGEN_TEXT_DIR = ONTOGEN_DATA_DIR / "text"
ONTOGEN_SECTION_DIR = ONTOGEN_DATA_DIR / "sections"
ONTOGEN_TABLE_DIR = ONTOGEN_DATA_DIR / "tables"

# "pymupdf"는 가볍고, "nougat"은 별도 설치가 필요하지만 논문 구조를 더 잘 살릴 수 있습니다.
ONTOGEN_TEXT_SOURCE = "pymupdf"

# 섹션 추출을 켰을 때 찾을 논문 섹션명입니다.
ONTOGEN_SECTIONS_TO_EXTRACT = [
    "abstract",
    "introduction",
    "methods",
    "experimental",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "ablation",
]
ONTOGEN_INCLUDE_SECTIONS_IN_ENRICHED_CONTEXT = ONTOGEN_SECTIONS_TO_EXTRACT + [
    "conclusion",
    "other",
]

# 표 추출을 켰을 때의 기본값입니다.
ONTOGEN_EXTRACT_TABLES = True
ONTOGEN_INCLUDE_TABLES_IN_ENRICHED_CONTEXT = True
