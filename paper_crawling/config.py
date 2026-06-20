from pathlib import Path

# ---------------------------
# User Settings
# ---------------------------
# 상위 폴더에서 `python -m paper_crawling` 실행.
# 또는 paper_crawling 폴더 내에서 python run.py 실행.
# 검색 주제, 연도 범위, 다운로드 여부, 랭킹 가중치를 여기서 조절하세요.

# 실행 옵션
# `True`이면 실제 PDF를 받지 않고 후보 검색/랭킹 결과만 저장합니다.
DRY_RUN = False

# 결과 저장 폴더입니다.
OUTPUT_ROOT = Path("paper_crawling/output")

# 터미널 로그 옵션
# 실행 중 수집/랭킹/다운로드 진행 상황을 출력합니다.
PRINT_PROGRESS = True
# `True`이면 OpenAlex page 단위 진행 상황도 함께 출력합니다.
PRINT_RETRIEVAL_PAGE_PROGRESS = True

# 검색 주제
# RETRIEVAL_TERMS는 OpenAlex 1차 검색에 쓰는 넓은 키워드/짧은 구문입니다.
# QUERY와 KEYWORDS는 넓게 모아온 후보를 로컬에서 다시 정렬할 때 사용합니다.
RETRIEVAL_TERMS = [
    "permanent magnet",
    "hard magnetic material",
    "rare earth magnet",
    "NdFeB magnet",
    "coercivity",
    "magnetic anisotropy",
]

# QUERY는 찾고 싶은 주제를 한 문장으로 적습니다.
# KEYWORDS는 rerank에서 더 강하게 보고 싶은 표현입니다.
QUERY = "permanent magnet materials with high coercivity and magnetic anisotropy"
KEYWORDS = [
    "permanent magnet",
    "hard magnetic material",
    "rare-earth magnet",
    "NdFeB",
    "coercivity",
    "magnetic anisotropy",
]
EXCLUDE_KEYWORDS = [
    # "soft magnetic",
]

# 검색 연도 범위
START_YEAR = 2020
END_YEAR = 2026

# 검색할 저널
SOURCES = [
    "advanced_materials",
    "advanced_functional_materials",
    "nature_materials",
    "acs_nano",
    "acs_applied_materials_interfaces",
]

# 후보 수집 및 최종 선택 개수
# source별 raw candidate hard cap입니다.
# 검증된 PDF 후보를 충분히 모으지 못하더라도 이 개수에서 수집을 멈춥니다.
MAX_CANDIDATES_PER_SOURCE = 120
# source별로 실제 PDF 응답이 검증된 후보를 최소 몇 개까지 모을지 정합니다.
# 내부적으로는 이 값의 2배와 80 중 더 큰 값을 minimum pool 크기로 자동 사용합니다.
TARGET_VERIFIED_CANDIDATES_PER_SOURCE = 30
# retrieval term 하나당 OpenAlex에서 최대 몇 개까지 가져올지 정합니다.
MAX_RESULTS_PER_RETRIEVAL_TERM = 40
# 각 저널(source)마다 최종적으로 몇 개의 PDF를 확보할지 정합니다.
TOP_K_PER_SOURCE = 10

# 다운로드 옵션
DOWNLOAD_PDFS = True
DOWNLOAD_OPEN_ACCESS_ONLY = True

# 요청 옵션
# OpenAlex/Crossref 등에 너무 빠르게 요청하지 않도록 약간의 지연을 둡니다.
REQUEST_DELAY_SECONDS = 1.0
EMAIL = "allloook02@yonsei.ac.kr"

# 랭킹 가중치
# 합은 꼭 1일 필요는 없지만, 모두 0이면 안 됩니다.
# BM25는 키워드 회수용, semantic은 의미 기반 재정렬용입니다.
BM25_WEIGHT = 0.4
PHRASE_WEIGHT = 0.15
CITATION_WEIGHT = 0.05
SEMANTIC_WEIGHT = 0.4

# semantic similarity 설정
# 첫 실행 시 Hugging Face에서 모델을 내려받을 수 있습니다.
SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
SEMANTIC_BATCH_SIZE = 32
