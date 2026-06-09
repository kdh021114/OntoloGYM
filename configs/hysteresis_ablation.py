"""Hysteresis-loop ablation 설정.

기본값은 실제 LLM 실험을 돌리지 않는 dry-run/opt-in 형태입니다.
필요한 ablation 단계만 아래 스위치를 켜서 실행하세요.
"""

import os
from pathlib import Path

from configs.common import RUN_OUTPUT_DIR, SHARED_PAPER_DIR
from hysteresis_ablation.prompts import HYSTERESIS_QA_FOCUS_HINT as DEFAULT_HYSTERESIS_QA_FOCUS_HINT


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


# =============================================================================
# 1. Hysteresis loop figure/caption asset 분리 저장
# =============================================================================

# True면 이미지를 복사하지 않고 후보 개수와 선택 기준만 확인합니다.
HYSTERESIS_ASSET_DRY_RUN = _env_bool("ONTOLOGYM_HYSTERESIS_ASSET_DRY_RUN", True)

# 논문 파싱 JSON과 figures/ 폴더가 들어 있는 입력 폴더입니다.
HYSTERESIS_SOURCE_PAPER_DIR = Path(
    os.getenv("ONTOLOGYM_HYSTERESIS_SOURCE_PAPER_DIR", os.fspath(SHARED_PAPER_DIR))
)

HYSTERESIS_DATA_DIR = RUN_OUTPUT_DIR / "hysteresis_ablation"
HYSTERESIS_ASSET_OUTPUT_DIR = HYSTERESIS_DATA_DIR / "assets"
HYSTERESIS_ASSET_IMAGE_DIR = HYSTERESIS_ASSET_OUTPUT_DIR / "images"
HYSTERESIS_ASSET_CAPTION_DIR = HYSTERESIS_ASSET_OUTPUT_DIR / "captions"
HYSTERESIS_ASSET_MANIFEST_JSONL = HYSTERESIS_ASSET_OUTPUT_DIR / "hysteresis_assets.jsonl"

# Caption keyword heuristic은 무료/빠른 1차 필터입니다.
HYSTERESIS_USE_CAPTION_HEURISTIC = _env_bool("ONTOLOGYM_HYSTERESIS_USE_CAPTION_HEURISTIC", True)
HYSTERESIS_CAPTION_SCORE_THRESHOLD = int(os.getenv("ONTOLOGYM_HYSTERESIS_CAPTION_SCORE_THRESHOLD", "2"))
HYSTERESIS_CAPTION_KEYWORDS = [
    "hysteresis",
    "hysteretic",
    "coercivity",
    "coercive",
    "remanence",
    "remanent",
    "demagnetization",
    "magnetization loop",
    "m-h",
    "b-h",
    "j-h",
    "magnetic field",
    "sweeping",
    "sweep direction",
    "field dependence",
    "forc",
    "hysteron",
]

# True면 이미지 자체를 vision model에 넣어 hysteresis loop 여부를 판별합니다.
# 비용이 발생하므로 기본값은 False입니다.
HYSTERESIS_USE_VISION_CLASSIFIER = _env_bool("ONTOLOGYM_HYSTERESIS_USE_VISION_CLASSIFIER", False)
HYSTERESIS_ALLOW_CAPTION_FALLBACK_WITH_VISION = _env_bool(
    "ONTOLOGYM_HYSTERESIS_ALLOW_CAPTION_FALLBACK_WITH_VISION",
    True,
)
HYSTERESIS_CLASSIFIER_MODEL = os.getenv("ONTOLOGYM_HYSTERESIS_CLASSIFIER_MODEL", "gpt-5.4-mini")
HYSTERESIS_CLASSIFIER_TEMPERATURE = 0.0
HYSTERESIS_CLASSIFIER_MAX_COMPLETION_TOKENS = 400
HYSTERESIS_VISION_CONFIDENCE_THRESHOLD = float(
    os.getenv("ONTOLOGYM_HYSTERESIS_VISION_CONFIDENCE_THRESHOLD", "0.55")
)

# 이미지 파일이 없는 figure caption은 이미지 ablation에 쓰기 어렵기 때문에 기본 제외합니다.
HYSTERESIS_REQUIRE_IMAGE_FILE = _env_bool("ONTOLOGYM_HYSTERESIS_REQUIRE_IMAGE_FILE", True)
HYSTERESIS_MAX_FIGURES_TO_SCAN = _env_int_or_none("ONTOLOGYM_HYSTERESIS_MAX_FIGURES_TO_SCAN", None)


# =============================================================================
# 2. KG relation augmentation에서 figure caption/image 사용
# =============================================================================

# relation_augmentation에서 hysteresis asset manifest를 evidence로 추가할지 여부입니다.
RELATION_INCLUDE_HYSTERESIS_CAPTIONS = _env_bool("ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTIONS", False)
RELATION_INCLUDE_HYSTERESIS_IMAGES = _env_bool("ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_IMAGES", False)
RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL = Path(
    os.getenv("ONTOLOGYM_RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL", os.fspath(HYSTERESIS_ASSET_MANIFEST_JSONL))
)

# image chunk는 caption과 이미지가 함께 들어가므로, 기본적으로 caption-only chunk와 중복 생성하지 않습니다.
RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES = _env_bool(
    "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES",
    False,
)

# False로 두면 hysteresis image chunk에 caption text를 넣지 않습니다.
# image-only ablation처럼 순수 시각 정보만 KG로 만들 때 사용합니다.
RELATION_HYSTERESIS_IMAGE_INCLUDE_CAPTION_CONTEXT = _env_bool(
    "ONTOLOGYM_RELATION_HYSTERESIS_IMAGE_INCLUDE_CAPTION_CONTEXT",
    True,
)

# image에서만 보이는 사실은 exact evidence quote가 없을 수 있습니다.
RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS = _env_bool(
    "ONTOLOGYM_RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS",
    False,
)

# 시각 KG를 더 자세히 만들고 싶을 때만 Figure/Curve/MagneticProperty 관계를 schema에 추가합니다.
RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS = _env_bool(
    "ONTOLOGYM_RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS",
    False,
)


# =============================================================================
# 3. QA dataset 증강: 기존 80개 중 40개 + image QA 40개
# =============================================================================

HYSTERESIS_QA_DRY_RUN = _env_bool("ONTOLOGYM_HYSTERESIS_QA_DRY_RUN", True)
HYSTERESIS_QA_GENERATE_IMAGE_EXAMPLES = _env_bool(
    "ONTOLOGYM_HYSTERESIS_QA_GENERATE_IMAGE_EXAMPLES",
    False,
)
HYSTERESIS_QA_REUSED_TEXT_COUNT = int(os.getenv("ONTOLOGYM_HYSTERESIS_QA_REUSED_TEXT_COUNT", "40"))
HYSTERESIS_QA_IMAGE_COUNT = int(os.getenv("ONTOLOGYM_HYSTERESIS_QA_IMAGE_COUNT", "40"))
HYSTERESIS_QA_RANDOM_SEED = int(os.getenv("ONTOLOGYM_HYSTERESIS_QA_RANDOM_SEED", "42"))
HYSTERESIS_QA_SAMPLE_BASE_RANDOMLY = _env_bool("ONTOLOGYM_HYSTERESIS_QA_SAMPLE_BASE_RANDOMLY", True)

# 비워두면 configs/qa_extractor.py의 QA_OUTPUT_DATASET_PATH를 사용합니다.
HYSTERESIS_QA_BASE_DATASET_PATH = Path(
    os.getenv("ONTOLOGYM_HYSTERESIS_QA_BASE_DATASET_PATH", "")
) if os.getenv("ONTOLOGYM_HYSTERESIS_QA_BASE_DATASET_PATH", "").strip() else None

HYSTERESIS_QA_IMAGE_EXAMPLE_DIR = Path(
    os.getenv("ONTOLOGYM_HYSTERESIS_QA_IMAGE_EXAMPLE_DIR", os.fspath(HYSTERESIS_DATA_DIR / "qa_image_examples"))
)
HYSTERESIS_QA_OUTPUT_DATASET_PATH = Path(
    os.getenv(
        "ONTOLOGYM_HYSTERESIS_QA_OUTPUT_DATASET_PATH",
        os.fspath(HYSTERESIS_DATA_DIR / "qa_dataset_hysteresis_augmented.jsonl"),
    )
)
HYSTERESIS_QA_SUMMARY_JSON = Path(
    os.getenv("ONTOLOGYM_HYSTERESIS_QA_SUMMARY_JSON", os.fspath(HYSTERESIS_DATA_DIR / "qa_dataset_summary.json"))
)
HYSTERESIS_QA_FOCUS_HINT = DEFAULT_HYSTERESIS_QA_FOCUS_HINT


# =============================================================================
# 4. Ablation eval helper
# =============================================================================

# 실제 eval은 비용이 들기 때문에 helper script도 기본 dry-run입니다.
HYSTERESIS_ABLATION_EVAL_DRY_RUN = _env_bool("ONTOLOGYM_HYSTERESIS_ABLATION_EVAL_DRY_RUN", True)
HYSTERESIS_ABLATION_DATASET_PATH = HYSTERESIS_QA_OUTPUT_DATASET_PATH
HYSTERESIS_CAPTION_RELATION_OUTPUT_DIR = RUN_OUTPUT_DIR / "relation_augmentation_hysteresis_caption" / "outputs"
HYSTERESIS_CAPTION_IMAGE_RELATION_OUTPUT_DIR = RUN_OUTPUT_DIR / "relation_augmentation_hysteresis_caption_image" / "outputs"
