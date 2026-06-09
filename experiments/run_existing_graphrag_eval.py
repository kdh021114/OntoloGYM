"""Run GraphRAG QA evaluation phases against an existing OntoloGYM run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.usage_costs import estimate_usage_cost


RUNS_DIR = PROJECT_ROOT / "data"
DEFAULT_RUN_ID = "run_20260526_032051"
DEFAULT_MODEL = "gpt-5.4-nano"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _run_eval(env: dict[str, str], expected_output: Path) -> None:
    if expected_output.exists() and _truthy(os.getenv("ONTOLOGYM_EXISTING_GRAPHRAG_SKIP_COMPLETED", "1")):
        print(f"Skipping existing phase; found {expected_output}", flush=True)
        return
    subprocess.run([sys.executable, "run_qa_evaluation.py"], cwd=PROJECT_ROOT, env=env, check=True)


def _print_usage(run_root: Path) -> None:
    summary = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
    print(
        "Usage so far: "
        f"{summary['prompt_tokens']} input, "
        f"{summary['completion_tokens']} output, "
        f"${summary['estimated_cost_usd']:.4f} estimated",
        flush=True,
    )


def main() -> None:
    run_id = os.getenv("ONTOLOGYM_EXISTING_GRAPHRAG_RUN_ID", DEFAULT_RUN_ID).strip() or DEFAULT_RUN_ID
    run_root = RUNS_DIR / run_id
    if not run_root.exists():
        raise FileNotFoundError(f"Missing run folder: {run_root}")

    base_env = os.environ.copy()
    base_env["ONTOLOGYM_RUN_ID"] = run_id
    if not base_env.get("ONTOLOGYM_MODEL", "").strip():
        base_env["ONTOLOGYM_MODEL"] = DEFAULT_MODEL

    base_dataset = run_root / "qa_extractor" / "qa_dataset.jsonl"
    holdout_dataset = run_root / "qa_extractor_refine_holdout" / "qa_dataset.jsonl"
    baseline_kg_dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    refined_kg_dirs = baseline_kg_dirs + [run_root / "kg_refinement"]

    print(f"Running existing GraphRAG eval for {run_root}", flush=True)

    base_phase_env = dict(base_env)
    base_phase_env["ONTOLOGYM_QA_EVAL_PHASE"] = "graphrag_base"
    base_phase_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(base_dataset)
    base_phase_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in baseline_kg_dirs)
    _run_eval(base_phase_env, run_root / "qa_evaluation_graphrag_base" / "outputs" / "evaluation_results.json")
    _print_usage(run_root)

    holdout_baseline_env = dict(base_env)
    holdout_baseline_env["ONTOLOGYM_QA_EVAL_PHASE"] = "graphrag_holdout_baseline"
    holdout_baseline_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    holdout_baseline_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in baseline_kg_dirs)
    _run_eval(
        holdout_baseline_env,
        run_root / "qa_evaluation_graphrag_holdout_baseline" / "outputs" / "evaluation_results.json",
    )
    _print_usage(run_root)

    holdout_refined_env = dict(base_env)
    holdout_refined_env["ONTOLOGYM_QA_EVAL_PHASE"] = "graphrag_holdout_refined"
    holdout_refined_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    holdout_refined_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in refined_kg_dirs)
    _run_eval(
        holdout_refined_env,
        run_root / "qa_evaluation_graphrag_holdout_refined" / "outputs" / "evaluation_results.json",
    )
    _print_usage(run_root)

    _write_summary()
    print(f"Finished existing GraphRAG eval for {run_root}", flush=True)


def _write_summary() -> None:
    subprocess.run([sys.executable, "experiments/summarize_experiment_results.py"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
