"""Run OntoloGYM experiment while reusing previously generated AirQA datasets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from common.run_context import create_new_run, record_pipeline_run, resolve_run_root
from common.usage_costs import estimate_usage_cost


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data"
SHARED_PAPER_DIR = PROJECT_ROOT / "data" / "papers"
DEFAULT_SOURCE_RUN_ID = "run_20260526_032051"
DEFAULT_MODEL = "gpt-5.4-mini"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _copy_reused_qa(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source QA directory: {source_dir}")
    refresh = _truthy(os.getenv("ONTOLOGYM_REUSE_QA_REFRESH"))
    if target_dir.exists() and refresh:
        shutil.rmtree(target_dir)
    if target_dir.exists():
        return
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns(".DS_Store"))


def _prepare_reused_ontogen_inputs(source_run: Path, run_root: Path) -> Path:
    source_processed_dir = source_run / "ontogen" / "processed_data"
    source_enriched_dir = source_run / "ontogen" / "enriched"
    if not source_processed_dir.exists():
        raise FileNotFoundError(f"Missing source OntoGen processed_data directory: {source_processed_dir}")

    target_processed_dir = run_root / "ontogen" / "processed_data"
    target_enriched_dir = run_root / "ontogen" / "enriched"
    paper_input_dir = run_root / "reused_paper_inputs"

    refresh = _truthy(os.getenv("ONTOLOGYM_REUSE_QA_REFRESH"))
    if refresh:
        for path in [target_processed_dir, target_enriched_dir, paper_input_dir]:
            if path.exists():
                shutil.rmtree(path)

    target_processed_dir.mkdir(parents=True, exist_ok=True)
    target_enriched_dir.mkdir(parents=True, exist_ok=True)
    paper_input_dir.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(source_processed_dir.glob("*.processed_data.json"))
    if not processed_files:
        raise FileNotFoundError(f"No processed_data JSON files found in {source_processed_dir}")

    for source_file in processed_files:
        paper_id = source_file.name.removesuffix(".processed_data.json")
        target_file = target_processed_dir / source_file.name
        if not target_file.exists():
            shutil.copy2(source_file, target_file)

        source_enriched = source_enriched_dir / f"{paper_id}.enriched.txt"
        target_enriched = target_enriched_dir / f"{paper_id}.enriched.txt"
        if source_enriched.exists() and not target_enriched.exists():
            shutil.copy2(source_enriched, target_enriched)

        # OntoGen discovers paper inputs from folder-style JSONs. The file is
        # only used as an input identity because processed/enriched outputs are
        # already placed at the expected reusable paths above.
        paper_dir = paper_input_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        input_json = paper_dir / f"{paper_id}.json"
        if not input_json.exists():
            input_json.write_text(
                '{"reused_processed_data": "'
                + str(target_file).replace("\\", "\\\\").replace('"', '\\"')
                + '"}\n',
                encoding="utf-8",
            )
    return paper_input_dir


def _run(script: str, env: dict[str, str], *, expected_output: Path | None = None) -> None:
    if expected_output is not None and expected_output.exists() and _truthy(os.getenv("ONTOLOGYM_REUSE_QA_SKIP_COMPLETED", "1")):
        print(f"Skipping {script}; found {expected_output}", flush=True)
        return
    print(f"Running {script} ...", flush=True)
    subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env, check=True)


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
    source_run_id = os.getenv("ONTOLOGYM_REUSE_QA_SOURCE_RUN", DEFAULT_SOURCE_RUN_ID).strip() or DEFAULT_SOURCE_RUN_ID
    source_run = RUNS_DIR / source_run_id
    source_base_qa = source_run / "qa_extractor"
    source_holdout_qa = source_run / "qa_extractor_refine_holdout"

    pipeline_order = [
        "qa_extractor_reused",
        "ontogen",
        "relation_augmentation",
        "qa_evaluation",
        "kg_refinement",
        "qa_extractor_refine_holdout_reused",
        "qa_evaluation_holdout_baseline",
        "qa_evaluation_holdout_refined",
        "kg_visualization",
    ]
    env_run_id = os.getenv("ONTOLOGYM_RUN_ID", "").strip()
    if env_run_id:
        run_root = resolve_run_root(
            project_root=PROJECT_ROOT,
            runs_dir=RUNS_DIR,
            paper_dir=SHARED_PAPER_DIR,
            explicit_run_id=env_run_id,
            pipeline_order=pipeline_order,
        )
    else:
        run_root = create_new_run(
            project_root=PROJECT_ROOT,
            runs_dir=RUNS_DIR,
            paper_dir=SHARED_PAPER_DIR,
            run_id_prefix="run_mini",
            pipeline_order=pipeline_order,
        )
    base_env = os.environ.copy()
    base_env["ONTOLOGYM_RUN_ID"] = run_root.name
    if not base_env.get("ONTOLOGYM_MODEL", "").strip():
        base_env["ONTOLOGYM_MODEL"] = DEFAULT_MODEL

    target_base_qa = run_root / "qa_extractor"
    target_holdout_qa = run_root / "qa_extractor_refine_holdout"
    _copy_reused_qa(source_base_qa, target_base_qa)
    _copy_reused_qa(source_holdout_qa, target_holdout_qa)
    reused_paper_input_dir = _prepare_reused_ontogen_inputs(source_run, run_root)
    record_pipeline_run(
        run_root,
        "qa_extractor_reused",
        status="reused",
        inputs={"source_dir": str(source_base_qa)},
        outputs={"dataset_path": str(target_base_qa / "qa_dataset.jsonl")},
        extra={"model": "not_applicable", "examples": _count_jsonl(target_base_qa / "qa_dataset.jsonl")},
    )
    record_pipeline_run(
        run_root,
        "qa_extractor_refine_holdout_reused",
        status="reused",
        inputs={"source_dir": str(source_holdout_qa)},
        outputs={"dataset_path": str(target_holdout_qa / "qa_dataset.jsonl")},
        extra={"model": "not_applicable", "examples": _count_jsonl(target_holdout_qa / "qa_dataset.jsonl")},
    )

    print(f"Created reused-QA experiment run: {run_root}", flush=True)
    print(f"Model override: {base_env['ONTOLOGYM_MODEL']}", flush=True)
    print(f"Reused base QA: {target_base_qa / 'qa_dataset.jsonl'}", flush=True)
    print(f"Reused holdout QA: {target_holdout_qa / 'qa_dataset.jsonl'}", flush=True)
    print(f"Reused OntoGen paper inputs: {reused_paper_input_dir}", flush=True)
    if _truthy(os.getenv("ONTOLOGYM_REUSE_QA_PREPARE_ONLY")):
        print("Prepare-only mode enabled; stopping before API-backed pipeline stages.", flush=True)
        return

    base_env["ONTOLOGYM_SHARED_PAPER_DIR"] = str(reused_paper_input_dir)
    _run("run_ontogen.py", base_env, expected_output=run_root / "ontogen" / "taxonomy" / "tree_0.pkl")
    _print_usage(run_root)
    _run("run_relation_augmentation.py", base_env, expected_output=run_root / "relation_augmentation" / "outputs" / "relation_graph.json")
    _print_usage(run_root)
    _run("run_qa_evaluation.py", base_env, expected_output=run_root / "qa_evaluation" / "outputs" / "evaluation_results.json")
    _print_usage(run_root)
    _run("run_kg_refinement.py", base_env, expected_output=run_root / "kg_refinement" / "refinement_report.json")
    _print_usage(run_root)

    holdout_dataset = target_holdout_qa / "qa_dataset.jsonl"
    baseline_kg_dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    refined_kg_dirs = baseline_kg_dirs + [run_root / "kg_refinement"]

    baseline_eval_env = dict(base_env)
    baseline_eval_env["ONTOLOGYM_QA_EVAL_PHASE"] = "holdout_baseline"
    baseline_eval_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    baseline_eval_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in baseline_kg_dirs)
    _run(
        "run_qa_evaluation.py",
        baseline_eval_env,
        expected_output=run_root / "qa_evaluation_holdout_baseline" / "outputs" / "evaluation_results.json",
    )
    _print_usage(run_root)

    refined_eval_env = dict(base_env)
    refined_eval_env["ONTOLOGYM_QA_EVAL_PHASE"] = "holdout_refined"
    refined_eval_env["ONTOLOGYM_QA_EVAL_DATASET_PATH"] = str(holdout_dataset)
    refined_eval_env["ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS"] = os.pathsep.join(str(path) for path in refined_kg_dirs)
    _run(
        "run_qa_evaluation.py",
        refined_eval_env,
        expected_output=run_root / "qa_evaluation_holdout_refined" / "outputs" / "evaluation_results.json",
    )
    _print_usage(run_root)

    _run("run_kg_visualization.py", base_env, expected_output=run_root / "kg_visualization" / "kg_graph.html")
    _write_summary(run_root)
    print(f"Finished reused-QA experiment run: {run_root}", flush=True)


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_summary(run_root: Path) -> None:
    env = os.environ.copy()
    if run_root.name.startswith("run_mini_"):
        env["ONTOLOGYM_REPORT_MINI_RUN"] = run_root.name
    else:
        env["ONTOLOGYM_REPORT_EXTRA_RUNS"] = run_root.name
    env["ONTOLOGYM_REPORT_OUTPUT"] = str(run_root / "experiment_summary.txt")
    subprocess.run([sys.executable, "summarize_experiment_results.py"], cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
