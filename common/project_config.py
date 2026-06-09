"""Load the root OntoloGYM config and optional .env file."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.py"
CONFIG_MODULE_NAME = "ontologym_project_config"


def load_project_config() -> ModuleType:
    if CONFIG_MODULE_NAME in sys.modules:
        return sys.modules[CONFIG_MODULE_NAME]

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    spec = importlib.util.spec_from_file_location(CONFIG_MODULE_NAME, CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load project config from {CONFIG_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[CONFIG_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_env_file(env_file: str | Path, override: bool = False) -> None:
    path = Path(env_file)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and (override or key not in os.environ):
            os.environ[key] = value
