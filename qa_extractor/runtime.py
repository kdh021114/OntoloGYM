"""Runtime helpers for the copied AirQA extractor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


QA_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = QA_ROOT / "config.py"
CONFIG_MODULE_NAME = "ontologym_qa_config"


def load_config() -> ModuleType:
    if CONFIG_MODULE_NAME in sys.modules:
        return sys.modules[CONFIG_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(CONFIG_MODULE_NAME, CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load QA config from {CONFIG_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[CONFIG_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
