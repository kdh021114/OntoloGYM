# paper_crawling

특정 주제와 관련된 논문/공개 PDF를 수집하는 최소 파이프라인입니다.

이 폴더는 아래 흐름을 기준으로 설계했습니다.

1. `OpenAlex`에서 넓은 retrieval term 기준으로 저널별 후보 pool 검색
2. `BM25 + semantic similarity + 구문 매칭 + 인용 수`로 pool 내부 로컬 재정렬
3. `Crossref`와 OpenAlex의 OA 정보를 이용해 PDF URL 보강
4. 합법적으로 접근 가능한 PDF만 다운로드

수집 종료 조건은 단순한 `후보 n개`가 아니라 `실제 PDF 응답이 검증된 후보` 목표치에 맞춥니다.
그래도 부족하면 source별 hard cap에서 멈춥니다.

등록된 OpenAlex 소스:

- `nature`
- `science`
- `nature_energy`
- `ees` (`Energy & Environmental Science`)
- `joule`
- `advanced_materials` (`Advanced Materials`)
- `advanced_functional_materials` (`Advanced Functional Materials`)
- `nature_materials` (`Nature Materials`)
- `acs_nano` (`ACS Nano`)
- `acs_applied_materials_interfaces` (`ACS Applied Materials & Interfaces`)

현재 [config.py](/Users/kdh02/workspace/senior_thesis/paper_crawling/config.py)의 기본 검색 대상은 재료/자성 분야에 맞춰 `advanced_materials`, `advanced_functional_materials`, `nature_materials`, `acs_nano`, `acs_applied_materials_interfaces`로 설정되어 있습니다.

주의:

- Wiley, Nature Portfolio, ACS 저널은 상당수 논문이 유료 접근입니다.
- 기본 설정은 `download_open_access_only=true` 입니다. 먼저 이 모드로 돌리는 것을 권장합니다.

## 왜 이렇게 설계했나

질문 문장 하나만으로 바로 PDF를 크롤링하는 방식보다, 아래 입력을 받는 구조가 훨씬 안정적입니다.

- `query`: 자연어 문장 또는 핵심 문장
- `retrieval_terms`: 넓게 후보를 모으는 검색용 키워드/짧은 구문
- `keywords`: 중요 키워드 리스트
- `exclude_keywords`: 제외어
- `year_range`: 시작/종료 연도
- `sources`: 검색할 저널/학회
- `top_k_per_source`: source별로 실제로 내려받을 상위 개수

즉, `넓은 retrieval term + 정밀 query/keywords + 연도 범위`를 같이 받는 것이 좋습니다.

## BM25 / semantic similarity / 더 좋은 방법

초기 버전은 아래 순서를 권장합니다.

1. `OpenAlex`에서 넓은 키워드로 후보 pool을 넓게 모읍니다.
2. 로컬에서 `BM25`로 제목/초록 기준 재정렬합니다.
3. 임베딩 기반 `semantic similarity`로 query와 제목/초록의 의미적 유사도를 반영합니다.
4. 구문 일치(`solid-state electrolyte` 같은 정확한 phrase)를 추가 가산합니다.
5. 동점권에서는 인용 수 같은 약한 prior를 섞습니다.

이 방식이 좋은 이유:

- 키워드 기반 주제 검색에는 BM25가 여전히 강합니다.
- semantic similarity만 단독으로 쓰면 짧은 query에서 과하게 넓어질 수 있습니다.
- 논문 수집에서는 "후보 회수(recall)"가 먼저이고, semantic similarity는 2차 리랭킹으로 넣는 편이 안전합니다.

더 발전시키고 싶으면:

- 1차 회수: `OpenAlex search`
- 2차 재정렬: 현재 구현된 `BM25`
- 3차 의미 기반 리랭킹: 현재 구현된 `sentence-transformers` 기반 semantic ranker
- 4차 선택적 고급 리랭킹: `SPECTER2`, `bge-m3`, `e5-large` 같은 임베딩 모델 또는 cross-encoder

처음부터 semantic similarity만 메인으로 두기보다는, 현재 구조처럼 `hybrid retrieval`로 시작하는 편이 좋습니다.

## 설치

```bash
pip install -r paper_crawling/requirements.txt
```

## 설정

사용자가 직접 수정해야 하는 파일은 [config.py](/Users/kdh02/workspace/senior_thesis/paper_crawling/config.py) 입니다.

```python
from pathlib import Path

DRY_RUN = False
OUTPUT_ROOT = Path("paper_crawling/output")

RETRIEVAL_TERMS = [
    "permanent magnet",
    "hard magnetic material",
    "rare earth magnet",
    "NdFeB magnet",
    "coercivity",
    "magnetic anisotropy",
]

QUERY = "permanent magnet materials with high coercivity and magnetic anisotropy"
KEYWORDS = [
    "permanent magnet",
    "hard magnetic material",
    "rare-earth magnet",
    "NdFeB",
    "coercivity",
    "magnetic anisotropy",
]
EXCLUDE_KEYWORDS = ["soft magnetic"]

START_YEAR = 2020
END_YEAR = 2026

SOURCES = [
    "advanced_materials",
    "advanced_functional_materials",
    "nature_materials",
    "acs_nano",
    "acs_applied_materials_interfaces",
]
MAX_CANDIDATES_PER_SOURCE = 120
TARGET_VERIFIED_CANDIDATES_PER_SOURCE = 30
MAX_RESULTS_PER_RETRIEVAL_TERM = 40
TOP_K_PER_SOURCE = 10

DOWNLOAD_PDFS = True
DOWNLOAD_OPEN_ACCESS_ONLY = True

REQUEST_DELAY_SECONDS = 1.0
EMAIL = "your_email@example.com"

BM25_WEIGHT = 0.45
PHRASE_WEIGHT = 0.15
CITATION_WEIGHT = 0.05
SEMANTIC_WEIGHT = 0.35

SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 32
```

## 실행

상위 폴더 `senior_thesis`에서 실행할 때:

```bash
python -m paper_crawling
```

`paper_crawling` 폴더 안에서 바로 실행할 때:

```bash
python run.py
```

다운로드 없이 후보만 보고 싶다면 [config.py](/Users/kdh02/workspace/senior_thesis/paper_crawling/config.py) 에서 아래 값을 바꾸면 됩니다.

```python
DRY_RUN = True
```

## 출력 구조

```text
paper_crawling/output/
└── 20260407_140501/
    ├── ranked_candidates.jsonl
    ├── run_summary.json
    ├── advanced_materials/
    │   ├── paper_a.pdf
    │   └── _metadata/
    │       └── paper_a.json
    ├── advanced_functional_materials/
    ├── nature_materials/
    ├── acs_nano/
    └── acs_applied_materials_interfaces/
```

각 source 폴더에는 PDF가 직접 저장되므로, 이후 `paper_parse`에 바로 넘기기 쉽습니다.

예를 들어 `advanced_materials` 결과를 파싱하려면 `paper_parse/config.py`에서 입력 폴더를 지정한 뒤 실행합니다.

```bash
python -m paper_parse
```

## 구현 메모

- 저널 메타데이터 검색은 OpenAlex source ID 기준으로 필터링합니다.
- 1차 검색은 `RETRIEVAL_TERMS`를 넓게 사용하고, 최종 선택은 `QUERY + KEYWORDS` 기반 랭킹으로 결정합니다.
- source 수집은 `TARGET_VERIFIED_CANDIDATES_PER_SOURCE`에 도달할 때까지 계속 진행하고, 너무 커지는 것은 `MAX_CANDIDATES_PER_SOURCE`로 제한합니다.
- 최종 다운로드 목표는 `TOP_K_PER_SOURCE` 기준입니다. source가 5개이고 값이 10이면 총 목표는 50개입니다.
- PDF URL은 OpenAlex OA location과 Crossref `link`를 우선 사용합니다.
- `download_open_access_only=false`로 바꾸면 비-OA PDF URL도 시도하지만, 실제 접근 가능 여부는 기관 인증/출판사 정책에 따라 달라집니다.
- 현재 기본 source들은 모두 OpenAlex 저널 검색 기반입니다.
- semantic similarity는 `sentence-transformers` 임베딩을 사용하며, 첫 실행 시 모델 다운로드가 발생할 수 있습니다.

## 구조

- 사용자 설정 파일:
  - `config.py`
- app 레이어:
  - `src/app/cli.py`
  - `src/app/pipeline.py`
- core 레이어:
  - `src/core/runtime_config.py`
  - `src/core/models.py`
  - `src/core/interfaces.py`
- processing 레이어:
  - `src/processing/candidate_processing.py`
  - `src/processing/downloader.py`
- source 정의와 생성:
  - `src/sources/catalog.py`
  - `src/sources/factory.py`
- source adapter:
  - `src/sources/openalex_journal_source.py`
  - `src/sources/site_pdf_source.py`
- source provider:
  - `src/sources/providers/openalex.py`
  - `src/sources/providers/crossref.py`
  - `src/sources/providers/site_pdf_discovery.py`
- ranker 구현:
  - `src/rankers/bm25_ranker.py`
  - `src/rankers/phrase_match_ranker.py`
  - `src/rankers/citation_ranker.py`
  - `src/rankers/embedders.py`
  - `src/rankers/semantic_similarity_ranker.py`
  - `src/rankers/hybrid_ranker.py`

새 저널/학회를 추가할 때는 보통 `src/sources/catalog.py`에 source 정의 하나만 추가하면 됩니다.
새로운 수집 방식이 필요할 때만 source adapter와 factory 등록을 추가하면 됩니다.

예를 들어 OpenAlex 기반 저널을 하나 더 추가할 때는 `provider_key="openalex_journal"` 인 `SourceDefinition`만 등록하면 됩니다.
