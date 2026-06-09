"""Run the copied AirQA extractor with OntoloGYM/config.py settings."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
QA_ROOT = PROJECT_ROOT / "qa_extractor"
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from extractor.annotator import run_from_config


if __name__ == "__main__":
    run_from_config()
