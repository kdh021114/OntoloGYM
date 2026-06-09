"""Run the copied OntoGen pipeline with OntoloGYM/config.py settings."""

from __future__ import annotations

import sys
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ONTOGEN_ROOT = PROJECT_ROOT / "ontogen"
if str(ONTOGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(ONTOGEN_ROOT))

from run import run_pipeline
import config


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    run_pipeline()
