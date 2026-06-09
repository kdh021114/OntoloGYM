"""Evidence 기반 relation augmentation 파이프라인 설정."""

import os

from configs.common import RUN_OUTPUT_DIR
from configs.hysteresis_ablation import (
    RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS as DEFAULT_RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS,
    RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL as DEFAULT_RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL,
    RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES as DEFAULT_RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES,
    RELATION_INCLUDE_HYSTERESIS_CAPTIONS as DEFAULT_RELATION_INCLUDE_HYSTERESIS_CAPTIONS,
    RELATION_INCLUDE_HYSTERESIS_IMAGES as DEFAULT_RELATION_INCLUDE_HYSTERESIS_IMAGES,
    RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS as DEFAULT_RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS,
)
from configs.ontogen import ONTOGEN_DATA_DIR, ONTOGEN_PROCESSED_DATA_DIR, ONTOGEN_TERMO_DIR


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
# 1. 실행과 입출력
# =============================================================================

# 이번 실험에서는 실제 relation KG를 만들어야 하므로 실행합니다.
RELATION_RUN_AUGMENTATION = True

RELATION_PHASE = os.getenv("ONTOLOGYM_RELATION_PHASE", "").strip()
RELATION_DATA_DIR = RUN_OUTPUT_DIR / ("relation_augmentation" if not RELATION_PHASE else f"relation_augmentation_{RELATION_PHASE}")
RELATION_PROCESSED_DATA_DIR = ONTOGEN_PROCESSED_DATA_DIR
RELATION_TERMO_DIR = ONTOGEN_TERMO_DIR
RELATION_OUTPUT_DIR = RELATION_DATA_DIR / "outputs"
RELATION_CLAIMS_JSONL = RELATION_OUTPUT_DIR / "relation_claims.jsonl"
RELATION_GRAPH_JSON = RELATION_OUTPUT_DIR / "relation_graph.json"

# 특정 processed_data JSON만 처리하고 싶으면 경로를 넣습니다.
# 비워두면 RELATION_PROCESSED_DATA_DIR 아래의 *.json을 모두 찾습니다.
RELATION_INPUT_FILES = []


# =============================================================================
# 2. LLM 설정
# =============================================================================

_relation_model_override = os.getenv("ONTOLOGYM_RELATION_MODEL", os.getenv("ONTOLOGYM_MODEL", "")).strip()
RELATION_MODEL = _relation_model_override or "gpt-5.4-nano"
RELATION_BACKEND = "openai"
RELATION_TEMPERATURE = 0.1
RELATION_MAX_COMPLETION_TOKENS = 4096
RELATION_REASONING_EFFORT = None

# True면 LLM을 호출하지 않고 evidence chunk 수와 후보 term 로딩만 확인합니다.
RELATION_DRY_RUN = False


# =============================================================================
# 3. Evidence 선택
# =============================================================================

# abstract/introduction은 요약과 배경지식이 섞이기 쉬워 기본 제외합니다.
RELATION_EXCLUDED_SECTIONS = ["abstract", "introduction"]
RELATION_INCLUDED_SECTIONS = [
    "methods",
    "experimental",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "ablation",
    "conclusion",
]
RELATION_INCLUDE_TABLES = True
# False로 두면 일반 논문 본문/table chunk를 relation extraction에 넣지 않습니다.
# Hysteresis ablation처럼 기존 KG 위에 figure caption/image-derived KG만 추가 생성할 때 사용합니다.
RELATION_INCLUDE_BASE_TEXT_CHUNKS = _env_bool("ONTOLOGYM_RELATION_INCLUDE_BASE_TEXT_CHUNKS", True)
# chunk가 너무 잘게 쪼개지면 비용만 늘고 nano 모델의 claim 품질은 크게 좋아지지 않았습니다.
# 한 번에 더 넓은 근거 문맥을 주되, 너무 길면 evidence.py에서 section 단위로 나눕니다.
RELATION_MAX_CHARS_PER_CHUNK = 12000
# 논문 하나에서 relation 추출에 사용할 evidence chunk 최대 개수입니다.
# 숫자로 제한하면 우선순위가 낮은 뒤쪽 section이 빠질 수 있으므로 기본값은 None입니다.
RELATION_MAX_CHUNKS_PER_PAPER = _env_int_or_none("ONTOLOGYM_RELATION_MAX_CHUNKS_PER_PAPER", None)
RELATION_MAX_CANDIDATE_TERMS = _env_int_or_none("ONTOLOGYM_RELATION_MAX_CANDIDATE_TERMS", 80)
# 한 chunk에서 저장할 claim 최대 개수입니다. 프롬프트뿐 아니라 코드에서도 강제합니다.
RELATION_MAX_CLAIMS_PER_CHUNK = _env_int_or_none("ONTOLOGYM_RELATION_MAX_CLAIMS_PER_CHUNK", 12)


# =============================================================================
# 4. Relation schema
# =============================================================================

# isA만으로 부족한 실험 세팅/결과 표현을 보완하는 최소 schema입니다.
# 논문의 모든 내용을 KG로 옮기려 하지 말고, 아래 relation에 명확히 들어맞는 claim만 추출합니다.
# relation의 domain/range는 문자열 하나 또는 문자열 리스트를 둘 다 허용합니다.
RELATION_ENTITY_TYPES = {
    "Material": "논문에서 다루는 물질, 화합물, 소재, 전극, 전해질 등",
    "Component": "소자나 시스템의 구성 요소",
    "Method": "합성법, 처리법, 측정법, 계산법, 분석법",
    "Experiment": "특정 조건에서 수행된 실험 또는 평가 event",
    "ExperimentalSetting": "실험 환경, 장비 설정, protocol 설정",
    "Condition": "온도, 시간, 압력, 전류밀도 등 결과에 영향을 주는 조건",
    "Metric": "측정 지표 또는 성능 지표",
    "Result": "실험에서 보고된 결과 또는 관찰된 결과",
    "QuantityValue": "수치와 단위가 결합된 값",
}

RELATION_TYPES = {
    "USES_MATERIAL": {
        "domain": "Experiment",
        "range": ["Material", "Component"],
        "description": "실험이 특정 물질, 소재, 전극, 전해질, 장치 구성 요소를 사용함",
        "sub_property_of": "prov:used",
    },
    "USES_METHOD": {
        "domain": "Experiment",
        "range": "Method",
        "description": "실험이 특정 방법, 절차, 분석법을 사용함",
        "sub_property_of": "prov:used",
    },
    "HAS_CONDITION": {
        "domain": ["Experiment", "Result"],
        "range": ["ExperimentalSetting", "Condition", "QuantityValue"],
        "description": "실험 또는 결과가 특정 setup, protocol, 온도, 시간, 전류밀도 등의 조건 하에서 성립함",
        "sub_property_of": "sosa:phenomenonTime",
    },
    "MEASURES_METRIC": {
        "domain": "Experiment",
        "range": "Metric",
        "description": "실험이 특정 지표를 측정함",
        "sub_property_of": "sosa:observedProperty",
    },
    "REPORTS_RESULT": {
        "domain": "Experiment",
        "range": ["Result", "QuantityValue"],
        "description": "실험이 특정 결과, 관찰값, 성능 수치 또는 수치-단위 값을 보고함",
        "sub_property_of": "sosa:hasResult",
    },
}


# =============================================================================
# 5. 검증과 병합
# =============================================================================

RELATION_MIN_CONFIDENCE = _env_float("ONTOLOGYM_RELATION_MIN_CONFIDENCE", 0.65)
RELATION_REQUIRE_EVIDENCE_QUOTE = True
RELATION_ALLOW_ENTITY_OUTSIDE_TERMO = True
RELATION_REJECT_GENERIC_ENTITIES = True
RELATION_MERGE_DUPLICATES = True


# =============================================================================
# 6. Hysteresis-loop ablation 옵션
# =============================================================================

# configs/hysteresis_ablation.py의 기본값을 가져오되, 환경 변수로도 바로 override할 수 있게 둡니다.
RELATION_INCLUDE_HYSTERESIS_CAPTIONS = _env_bool(
    "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTIONS",
    DEFAULT_RELATION_INCLUDE_HYSTERESIS_CAPTIONS,
)
RELATION_INCLUDE_HYSTERESIS_IMAGES = _env_bool(
    "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_IMAGES",
    DEFAULT_RELATION_INCLUDE_HYSTERESIS_IMAGES,
)
RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL = DEFAULT_RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL
RELATION_MAX_HYSTERESIS_ASSETS = _env_int_or_none("ONTOLOGYM_RELATION_MAX_HYSTERESIS_ASSETS", None)
RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES = _env_bool(
    "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES",
    DEFAULT_RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES,
)
RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS = _env_bool(
    "ONTOLOGYM_RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS",
    DEFAULT_RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS,
)
RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS = _env_bool(
    "ONTOLOGYM_RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS",
    DEFAULT_RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS,
)

if RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS:
    RELATION_ENTITY_TYPES.update(
        {
            "Figure": "논문 figure 또는 figure panel",
            "Curve": "hysteresis loop, demagnetization curve, FORC, 또는 field-sweep curve",
            "MagneticProperty": "coercivity, remanence, saturation magnetization, squareness, (BH)max 등 자기 특성",
            "Observation": "figure 또는 caption에서 직접 확인되는 시각적 관찰",
        }
    )
    RELATION_TYPES.update(
        {
            "SHOWS": {
                "domain": "Figure",
                "range": ["Curve", "MagneticProperty", "Observation"],
                "description": "figure가 curve, 자기 특성, 또는 관찰 내용을 보여줌",
            },
            "REPRESENTS": {
                "domain": "Curve",
                "range": ["Material", "Experiment", "ExperimentalSetting", "Condition"],
                "description": "curve가 특정 물질, 실험, 조건, 또는 setting을 나타냄",
            },
            "HAS_PROPERTY": {
                "domain": ["Material", "Curve", "Experiment"],
                "range": "MagneticProperty",
                "description": "물질, curve, 또는 실험이 특정 자기 특성을 가짐",
            },
            "HAS_VALUE": {
                "domain": "MagneticProperty",
                "range": "QuantityValue",
                "description": "자기 특성이 figure/caption에 표시된 수치 값을 가짐",
            },
            "INDICATES": {
                "domain": "Observation",
                "range": "Result",
                "description": "시각적 관찰이 특정 결과를 뒷받침함",
            },
            "EVIDENCED_BY": {
                "domain": "Result",
                "range": "Figure",
                "description": "결과가 특정 figure에 의해 뒷받침됨",
            },
        }
    )
