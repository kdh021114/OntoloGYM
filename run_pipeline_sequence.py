"""Run OntoloGYM pipelines in the configured order inside one new run folder."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from common.run_context import DEFAULT_PIPELINE_ORDER, create_new_run


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data"
SHARED_PAPER_DIR = PROJECT_ROOT / "data" / "papers"


COMMANDS = {
    "qa_extractor": [sys.executable, "run_qa_extractor.py"],
    "ontogen": [sys.executable, "run_ontogen.py"],
    "relation_augmentation": [sys.executable, "run_relation_augmentation.py"],
    "qa_evaluation": [sys.executable, "run_qa_evaluation.py"],
    "kg_refinement": [sys.executable, "run_kg_refinement.py"],
    "kg_visualization": [sys.executable, "run_kg_visualization.py"],
}


def main() -> None:
    run_root = create_new_run(
        project_root=PROJECT_ROOT,
        runs_dir=RUNS_DIR,
        paper_dir=SHARED_PAPER_DIR,
        pipeline_order=DEFAULT_PIPELINE_ORDER,
    )
    env = os.environ.copy()
    env["ONTOLOGYM_RUN_ID"] = run_root.name

    import config

    pipeline_order = list(getattr(config, "RUN_PIPELINE_ORDER", DEFAULT_PIPELINE_ORDER))
    print(f"Created run: {run_root}")
    for pipeline_name in pipeline_order:
        command = COMMANDS.get(pipeline_name)
        if command is None:
            print(f"Skipping unknown pipeline: {pipeline_name}")
            continue
        print(f"Running {pipeline_name} ...")
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    print(f"Finished run: {run_root}")


if __name__ == "__main__":
    main()
