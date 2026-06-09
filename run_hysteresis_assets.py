"""Collect hysteresis-loop figures/captions into a separate ablation asset folder."""

from __future__ import annotations

import json
import logging

from common.project_config import load_env_file, load_project_config
from hysteresis_ablation.assets import collect_hysteresis_assets


def main() -> None:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))
    log_level = str(getattr(config, "ONTOGEN_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    summary = collect_hysteresis_assets(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
