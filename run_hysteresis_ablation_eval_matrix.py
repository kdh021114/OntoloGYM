"""Prepare or run the three hysteresis ablation QA-evaluation variants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common.project_config import load_env_file, load_project_config


def main() -> None:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))
    dry_run = bool(getattr(config, "HYSTERESIS_ABLATION_EVAL_DRY_RUN", True))
    variants = _variants(config)
    summary = {
        "status": "dry_run" if dry_run else "completed",
        "dataset_path": str(config.HYSTERESIS_ABLATION_DATASET_PATH),
        "variants": variants,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if dry_run:
        return

    for variant in variants:
        env = os.environ.copy()
        env.update(
            {
                "ONTOLOGYM_QA_EVAL_PHASE": variant["phase"],
                "ONTOLOGYM_QA_EVAL_DATASET_PATH": str(config.HYSTERESIS_ABLATION_DATASET_PATH),
                "ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS": os.pathsep.join(variant["kg_dirs"]),
            }
        )
        subprocess.run([sys.executable, "run_qa_evaluation.py"], cwd=Path(__file__).resolve().parent, env=env, check=True)


def _variants(config) -> list[dict[str, Any]]:
    ontogen_dirs = [str(config.ONTOGEN_TAXONOMY_DIR), str(config.ONTOGEN_TERMO_DIR)]
    existing_relation_dir = Path(config.RUN_OUTPUT_DIR) / "relation_augmentation" / "outputs"
    return [
        {
            "name": "existing_kg",
            "phase": "hysteresis_existing_kg",
            "kg_dirs": ontogen_dirs + [str(existing_relation_dir)],
        },
        {
            "name": "caption_augmented_kg",
            "phase": "hysteresis_caption_kg",
            "kg_dirs": ontogen_dirs + [str(config.HYSTERESIS_CAPTION_RELATION_OUTPUT_DIR)],
        },
        {
            "name": "caption_image_augmented_kg",
            "phase": "hysteresis_caption_image_kg",
            "kg_dirs": ontogen_dirs + [str(config.HYSTERESIS_CAPTION_IMAGE_RELATION_OUTPUT_DIR)],
        },
    ]


if __name__ == "__main__":
    main()
