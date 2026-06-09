"""Run KG-grounded evaluation on an AirQA dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from qa_evaluation import run_pipeline


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.ONTOGEN_LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    run_pipeline()
