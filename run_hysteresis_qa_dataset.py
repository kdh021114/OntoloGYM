"""Build the hysteresis-loop ablation QA dataset from config.py."""

from __future__ import annotations

import json
import logging

from common.project_config import load_env_file, load_project_config
from hysteresis_ablation.qa_dataset import build_augmented_qa_dataset


def main() -> None:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))
    log_level = str(getattr(config, "ONTOGEN_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    summary = build_augmented_qa_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
