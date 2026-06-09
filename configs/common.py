"""OntoloGYM 공통 경로와 환경 변수 설정."""

import os
from pathlib import Path

from common.run_context import DEFAULT_PIPELINE_ORDER, resolve_run_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# OpenAI API key 등 민감한 값은 이 파일이 아니라 .env에 넣습니다.
ENV_FILE = PROJECT_ROOT / ".env"

# 여러 파이프라인이 함께 참조할 논문 PDF/파싱 JSON 폴더입니다.
_paper_dir_override = os.getenv("ONTOLOGYM_SHARED_PAPER_DIR", "").strip()
SHARED_PAPER_DIR = Path(_paper_dir_override) if _paper_dir_override else PROJECT_ROOT / "data" / "papers"


# =============================================================================
# Run 단위 output 관리
# =============================================================================

# "run": data/<run_id>/ 아래에 모든 파이프라인 산출물을 모읍니다.
# "legacy": 예전처럼 data/airqa, data/ontogen 등 파이프라인별 폴더를 직접 사용합니다.
RUN_OUTPUT_MODE = "run"

# 여러 파이프라인이 같은 산출물 묶음을 이어 쓰기 위한 run 폴더입니다.
RUNS_DIR = PROJECT_ROOT / "data"

# None이면 data/_active_run.txt에 적힌 run을 이어 쓰고, 없으면 새로 만듭니다.
RUN_ID = None

# True로 바꾸면 다음 실행 때 새 run 폴더를 만들고 active run으로 지정합니다.
# 한 번 새 run을 만든 뒤에는 다시 False로 돌려두는 것을 권장합니다.
RUN_CREATE_NEW = False
RUN_ID_PREFIX = "run"

# 전체 실험 흐름의 권장 실행 순서입니다. run_pipeline_sequence.py가 이 순서를 사용합니다.
RUN_PIPELINE_ORDER = DEFAULT_PIPELINE_ORDER

RUN_OUTPUT_DIR = resolve_run_root(
    project_root=PROJECT_ROOT,
    runs_dir=RUNS_DIR,
    paper_dir=SHARED_PAPER_DIR,
    output_mode=RUN_OUTPUT_MODE,
    explicit_run_id=RUN_ID,
    create_new=RUN_CREATE_NEW,
    run_id_prefix=RUN_ID_PREFIX,
    pipeline_order=RUN_PIPELINE_ORDER,
)

# OpenAI usage 로그도 run 폴더에 묶이게 합니다.
USAGE_LOG_PATH = RUN_OUTPUT_DIR / "logs" / "openai_usage.jsonl"
os.environ.setdefault("ONTOLOGYM_USAGE_LOG", os.fspath(USAGE_LOG_PATH))

EXPERIMENT_BUDGET_USD = 70.0
os.environ.setdefault("ONTOLOGYM_BUDGET_USD", str(EXPERIMENT_BUDGET_USD))
