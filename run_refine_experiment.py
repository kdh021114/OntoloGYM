"""Run the full QA -> KG -> eval -> KG refine -> holdout comparison experiment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from common.run_context import create_new_run
from common.usage_costs import estimate_usage_cost


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data"
SHARED_PAPER_DIR = PROJECT_ROOT / "data" / "papers"


def _run(script: str, env: dict[str, str]) -> None:
    print(f"Running {script} ...")
    subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env, check=True)


def _print_usage(run_root: Path) -> None:
    summary = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
    print(
        "Usage so far: "
        f"{summary['prompt_tokens']} input, "
        f"{summary['completion_tokens']} output, "
        f"${summary['estimated_cost_usd']:.4f} estimated"
    )


def main() -> None:
    pipeline_order = [
        "qa_extractor",
        "ontogen",
        "relation_augmentation",
        "qa_evaluation",
        "kg_refinement",
        "qa_extractor_refine_holdout",
        "qa_evaluation_holdout_baseline",
        "qa_evaluation_holdout_refined",
        "kg_visualization",
    ]
    run_root = create_new_run(
        project_root=PROJECT_ROOT,
        runs_dir=RUNS_DIR,
        paper_dir=SHARED_PAPER_DIR,
        pipeline_order=pipeline_order,
    )
    base_env = os.environ.copy()
    base_env["ONTOLOGYM_RUN_ID"] = run_root.name

    print(f"Created experiment run: {run_root}")
    _run("run_qa_extractor.py", base_env)
    _print_usage(run_root)
    _run("run_ontogen.py", base_env)
    _print_usage(run_root)
    _run("run_relation_augmentation.py", base_env)
    _print_usage(run_root)
    _run("run_qa_evaluation.py", base_env)
    _print_usage(run_root)
    _run("run_kg_refinement.py", base_env)
    _print_usage(run_root)

    holdout_env = dict(base_env)
    holdout_env["ONTOLOGYM_QA_PHASE"] = "refine_holdout"
    _run("run_qa_extractor.py", holdout_env)
    _print_usage(run_root)

    holdout_dataset = run_root / "qa_extractor_refine_holdout" / "qa_dataset.jsonl"
    baseline_kg_dirs = [
        run_root / "qa_evaluation" / "input_kg",
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    refined_kg_dirs = baseline_kg_dirs + [run_root / "kg_refinement"]

    baseline_eval_env = dict(base_env)
    baseline_eval_env["ONTOLOGYM_QA_EVAL_PHASE"] = "holdout_baseline"
    baseline_eval_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    baseline_eval_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in baseline_kg_dirs)
    _run("run_qa_evaluation.py", baseline_eval_env)
    _print_usage(run_root)

    refined_eval_env = dict(base_env)
    refined_eval_env["ONTOLOGYM_QA_EVAL_PHASE"] = "holdout_refined"
    refined_eval_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    refined_eval_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in refined_kg_dirs)
    _run("run_qa_evaluation.py", refined_eval_env)
    _print_usage(run_root)

    _run("run_kg_visualization.py", base_env)
    print(f"Finished experiment run: {run_root}")


if __name__ == "__main__":
    main()
