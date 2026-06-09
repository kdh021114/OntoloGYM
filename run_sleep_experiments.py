"""Run the thesis KG/QA experiments with aggressive artifact reuse.

This script is intentionally config-light: it stitches existing run folders
together and uses environment variables to drive the already separated
pipelines. Re-running it skips completed outputs and only fills missing pieces.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PYTHON = sys.executable

MODEL_RUNS = {
    "nano": DATA_DIR / "run_pre_refine_20260601_014021_current_nano",
    "mini": DATA_DIR / "run_pre_refine_20260601_014021_current_mini",
    "gpt54": DATA_DIR / "run_pre_refine_20260601_014021_current_gpt54",
}

REFINE_QA_ROOT = DATA_DIR / "refine_qa_sets_20260601"
REFINE_DATASETS = [
    REFINE_QA_ROOT / "round_01_current_base80" / "qa_dataset.jsonl",
    REFINE_QA_ROOT / "round_02_old_base80" / "qa_dataset.jsonl",
    REFINE_QA_ROOT / "round_03_old_holdout80" / "qa_dataset.jsonl",
    REFINE_QA_ROOT / "round_04_improved_v3_80" / "qa_dataset.jsonl",
    REFINE_QA_ROOT / "round_05_fair79_plus_1" / "qa_dataset.jsonl",
]
FINAL_HOLDOUT = REFINE_QA_ROOT / "reserved_final_holdout_current80" / "qa_dataset.jsonl"
MANUAL40 = DATA_DIR / "verified_qa_manual_20260601" / "qa_dataset.jsonl"

SUMMARY_JSON = DATA_DIR / "sleep_experiments_20260602_summary.json"
SUMMARY_TXT = DATA_DIR / "sleep_experiments_20260602_summary.txt"

FIXED_ANSWER_MODEL = "gpt-5.4-mini"
FIXED_JUDGE_MODEL = "manual_codex_scoring"
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATE_EVAL_ANSWERS = os.getenv(
    "ONTOLOGYM_SLEEP_EXPERIMENTS_GENERATE_EVAL_ANSWERS",
    "",
).strip().lower() in {"1", "true", "yes", "y", "on"}

USD_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


@dataclass
class StepRecord:
    name: str
    status: str
    run_id: str
    command: str | None = None
    output: str | None = None
    cost_usd: float = 0.0
    seconds: float = 0.0
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "run_id": self.run_id,
            "command": self.command,
            "output": self.output,
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 2),
            "detail": self.detail or {},
        }


def main() -> None:
    _assert_inputs()
    records: list[StepRecord] = []

    print("[1/4] Running main-table and manual-dataset evaluations.")
    records.extend(run_main_and_manual_evals())

    print("[2/4] Running nano iterative refinement ablation.")
    records.extend(run_iterative_refine_ablation())

    print("[3/4] Running KG size ablations.")
    records.extend(run_relation_cap_ablation())
    records.extend(run_taxonomy_size_ablation())

    print("[4/4] Writing summaries.")
    write_summary(records)
    total_cost = sum(record.cost_usd for record in records)
    print(f"Done. New estimated cost: ${total_cost:.4f}")
    print(f"Summary: {SUMMARY_TXT}")


def _assert_inputs() -> None:
    required = [FINAL_HOLDOUT, MANUAL40, *REFINE_DATASETS]
    required.extend(MODEL_RUNS.values())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required experiment inputs:\n" + "\n".join(str(path) for path in missing))


def run_main_and_manual_evals() -> list[StepRecord]:
    records: list[StepRecord] = []
    for model_name, run_root in MODEL_RUNS.items():
        kg_sets = {
            "ontogen_only": ontogen_dirs(run_root),
            "baseline_aug": baseline_dirs(run_root),
            "refined": refined_dirs(run_root, "kg_refinement_schema_v2"),
        }
        for kg_name, kg_dirs in kg_sets.items():
            records.append(
                eval_phase(
                    run_root=run_root,
                    dataset=FINAL_HOLDOUT,
                    phase=f"sleep_holdout_{kg_name}",
                    kg_dirs=kg_dirs,
                    label=f"main:{model_name}:{kg_name}:holdout",
                )
            )
            records.append(
                eval_phase(
                    run_root=run_root,
                    dataset=MANUAL40,
                    phase=f"sleep_manual40_{kg_name}",
                    kg_dirs=kg_dirs,
                    label=f"manual40:{model_name}:{kg_name}",
                )
            )
    return records


def run_iterative_refine_ablation() -> list[StepRecord]:
    records: list[StepRecord] = []
    source_run = MODEL_RUNS["nano"]
    iter_run = DATA_DIR / "run_iter_refine_nano_20260602"
    ensure_base_run(iter_run, source_run)
    first_refine = iter_run / "kg_refinement_iter_r01"
    copy_dir_if_missing(source_run / "kg_refinement_schema_v2", first_refine)

    for index, dataset in enumerate(REFINE_DATASETS, start=1):
        original_phase = f"iter_r{index:02d}_original"
        refined_phase = f"iter_r{index:02d}_refined"
        records.append(
            eval_phase(
                run_root=iter_run,
                dataset=dataset,
                phase=original_phase,
                kg_dirs=baseline_dirs(iter_run),
                label=f"iter-refine:r{index}:original",
            )
        )

        if index >= 2:
            pre_phase = f"iter_r{index:02d}_pre_for_refine"
            pre_eval = eval_phase(
                run_root=iter_run,
                dataset=dataset,
                phase=pre_phase,
                kg_dirs=iter_refined_dirs(iter_run, index - 1),
                label=f"iter-refine:r{index}:pre",
            )
            records.append(pre_eval)
            manual_pre_eval = manual_result_path(iter_run, pre_phase)
            if not manual_pre_eval.exists():
                records.append(
                    StepRecord(
                        name=f"iter-refine:r{index}:patch",
                        status="pending_manual_scoring",
                        run_id=iter_run.name,
                        output=str(manual_pre_eval),
                        detail={
                            "reason": "KG refinement needs manually scored failed QA cases; API evaluator is disabled.",
                            "answer_file": str(iter_run / f"qa_evaluation_{pre_phase}" / "outputs" / "kg_answers.jsonl"),
                        },
                    )
                )
                break
            refine_dir = iter_run / f"kg_refinement_iter_r{index:02d}"
            if not (refine_dir / "refined_relation_graph.json").exists():
                records.append(
                    refine_phase(
                        run_root=iter_run,
                        dataset=dataset,
                        eval_results=manual_pre_eval,
                        output_dir_name=refine_dir.name,
                        label=f"iter-refine:r{index}:patch",
                    )
                )
            else:
                records.append(
                    StepRecord(
                        name=f"iter-refine:r{index}:patch",
                        status="skipped",
                        run_id=iter_run.name,
                        output=str(refine_dir / "refined_relation_graph.json"),
                    )
                )

        records.append(
            eval_phase(
                run_root=iter_run,
                dataset=dataset,
                phase=refined_phase,
                kg_dirs=iter_refined_dirs(iter_run, index),
                label=f"iter-refine:r{index}:refined",
            )
        )
    return records


def run_relation_cap_ablation() -> list[StepRecord]:
    records: list[StepRecord] = []
    source_run = MODEL_RUNS["nano"]
    for cap in [6, 18]:
        run_root = DATA_DIR / f"run_ablation_relcap{cap}_nano_20260602"
        ensure_base_run(run_root, source_run, include_relation=False)
        output_graph = run_root / "relation_augmentation" / "outputs" / "relation_graph.json"
        if not output_graph.exists():
            records.append(
                run_command(
                    name=f"ablation:relation_cap_{cap}:generate",
                    run_root=run_root,
                    script="run_relation_augmentation.py",
                    extra_env={
                        "ONTOLOGYM_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_RELATION_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_RELATION_MAX_CLAIMS_PER_CHUNK": str(cap),
                    },
                    expected_output=output_graph,
                )
            )
        else:
            records.append(
                StepRecord(
                    name=f"ablation:relation_cap_{cap}:generate",
                    status="skipped",
                    run_id=run_root.name,
                    output=str(output_graph),
                    detail=read_relation_summary(run_root),
                )
            )
        records.append(
            eval_phase(
                run_root=run_root,
                dataset=FINAL_HOLDOUT,
                phase=f"ablation_relcap{cap}_holdout",
                kg_dirs=baseline_dirs(run_root),
                label=f"ablation:relation_cap_{cap}:holdout",
            )
        )
    return records


def run_taxonomy_size_ablation() -> list[StepRecord]:
    records: list[StepRecord] = []
    source_run = MODEL_RUNS["nano"]
    variants = {
        "taxsmall": {
            "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_TARGET_COUNT": "7",
            "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MIN_TARGET_COUNT": "7",
            "ONTOLOGYM_ONTOGEN_TAXONOMY_MAX_TERMS_PER_PAPER": "45",
            "ONTOLOGYM_ONTOGEN_CATEGORY_MAX_TOTAL_CHARS": "45000",
        },
        "taxlarge": {
            "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_TARGET_COUNT": "14",
            "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MIN_TARGET_COUNT": "14",
            "ONTOLOGYM_ONTOGEN_TAXONOMY_MAX_TERMS_PER_PAPER": "140",
            "ONTOLOGYM_ONTOGEN_CATEGORY_MAX_TOTAL_CHARS": "80000",
        },
    }
    for variant, env_overrides in variants.items():
        run_root = DATA_DIR / f"run_ablation_{variant}_nano_20260602"
        ensure_base_run(run_root, source_run, include_categories=False, include_taxonomy=False, include_relation=False)
        taxonomy_marker = run_root / "ontogen" / "taxonomy" / "tree_0.pkl"
        if not taxonomy_marker.exists():
            records.append(
                run_command(
                    name=f"ablation:{variant}:ontogen",
                    run_root=run_root,
                    script="run_ontogen.py",
                    extra_env={
                        "ONTOLOGYM_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_TERMO_TERMS_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_CATEGORY_GENERATION_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_CATEGORY_FORMAT_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_CATEGORY_SYNTHESIS_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_ONTOGEN_TAXONOMY_MODEL": "gpt-5.4-nano",
                        **env_overrides,
                    },
                    expected_output=taxonomy_marker,
                )
            )
        else:
            records.append(
                StepRecord(
                    name=f"ablation:{variant}:ontogen",
                    status="skipped",
                    run_id=run_root.name,
                    output=str(taxonomy_marker),
                )
            )

        relation_graph = run_root / "relation_augmentation" / "outputs" / "relation_graph.json"
        if not relation_graph.exists():
            records.append(
                run_command(
                    name=f"ablation:{variant}:relation_cap_12",
                    run_root=run_root,
                    script="run_relation_augmentation.py",
                    extra_env={
                        "ONTOLOGYM_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_RELATION_MODEL": "gpt-5.4-nano",
                        "ONTOLOGYM_RELATION_MAX_CLAIMS_PER_CHUNK": "12",
                    },
                    expected_output=relation_graph,
                )
            )
        else:
            records.append(
                StepRecord(
                    name=f"ablation:{variant}:relation_cap_12",
                    status="skipped",
                    run_id=run_root.name,
                    output=str(relation_graph),
                    detail=read_relation_summary(run_root),
                )
            )

        records.append(
            eval_phase(
                run_root=run_root,
                dataset=FINAL_HOLDOUT,
                phase=f"ablation_{variant}_holdout",
                kg_dirs=baseline_dirs(run_root),
                label=f"ablation:{variant}:holdout",
            )
        )
    return records


def eval_phase(
    *,
    run_root: Path,
    dataset: Path,
    phase: str,
    kg_dirs: list[Path],
    label: str,
) -> StepRecord:
    result_path = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
    answer_path = result_path.with_name("kg_answers.jsonl")
    manual_path = result_path.with_name("evaluation_results.manual.json")
    if manual_path.exists() and not _refresh():
        return StepRecord(
            name=label,
            status="skipped",
            run_id=run_root.name,
            output=str(result_path),
            detail=read_eval_summary(result_path),
        )
    if answer_path.exists() and not _refresh():
        return StepRecord(
            name=label,
            status="pending_manual_scoring",
            run_id=run_root.name,
            output=str(manual_path),
            detail={
                "reason": "Automatic evaluator output is ignored; manual Codex scoring is required.",
                "answers_jsonl": str(answer_path),
                "auto_results_json": str(result_path) if result_path.exists() else None,
            },
        )
    if not GENERATE_EVAL_ANSWERS:
        return StepRecord(
            name=label,
            status="pending_manual_evaluation",
            run_id=run_root.name,
            output=str(manual_path),
            detail={
                "reason": "Evaluation API calls are disabled. Score this phase manually from the dataset and KG artifacts.",
                "dataset": str(dataset),
                "kg_dirs": [str(path) for path in kg_dirs],
            },
        )
    record = run_command(
        name=label,
        run_root=run_root,
        script="run_qa_evaluation.py",
        extra_env={
            "ONTOLOGYM_QA_EVAL_PHASE": phase,
            "ONTOLOGYM_QA_EVAL_DATASET_PATH": str(dataset),
            "ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS": os.pathsep.join(str(path) for path in kg_dirs),
            "ONTOLOGYM_QA_EVAL_MODEL": FIXED_ANSWER_MODEL,
            "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL": FIXED_JUDGE_MODEL,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS": "0",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS": "1",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_BM25_WEIGHT": "0.4",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT": "0.6",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY": "node_bundle_iterative",
            "ONTOLOGYM_QA_EVAL_RUN_AIRQA_EVALUATOR": "0",
            "ONTOLOGYM_QA_EVAL_ALLOW_LLM_EVALUATORS": "0",
            "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES": "0",
        },
        expected_output=result_path,
    )
    if manual_path.exists():
        record.detail = read_eval_summary(result_path)
    elif answer_path.exists():
        record.status = "pending_manual_scoring"
        record.output = str(manual_path)
        record.detail = {
            "reason": "Answers were generated, but evaluation must be scored manually.",
            "answers_jsonl": str(answer_path),
            "auto_results_json": str(result_path) if result_path.exists() else None,
        }
    return record


def refine_phase(
    *,
    run_root: Path,
    dataset: Path,
    eval_results: Path,
    output_dir_name: str,
    label: str,
) -> StepRecord:
    output_graph = run_root / output_dir_name / "refined_relation_graph.json"
    record = run_command(
        name=label,
        run_root=run_root,
        script="run_kg_refinement.py",
        extra_env={
            "ONTOLOGYM_MODEL": "gpt-5.4-nano",
            "ONTOLOGYM_KG_REFINE_MODEL": "gpt-5.4-nano",
            "ONTOLOGYM_KG_REFINE_INPUT_EVAL_RESULTS_JSON": str(eval_results),
            "ONTOLOGYM_KG_REFINE_INPUT_DATASET_PATH": str(dataset),
            "ONTOLOGYM_KG_REFINE_OUTPUT_DIR": output_dir_name,
            "ONTOLOGYM_KG_REFINE_MAX_CASES": "80",
            "ONTOLOGYM_KG_REFINE_MAX_CONTEXT_CHARS": "1500",
        },
        expected_output=output_graph,
    )
    report = run_root / output_dir_name / "refinement_report.json"
    if report.exists():
        record.detail = read_json(report)
    return record


def run_command(
    *,
    name: str,
    run_root: Path,
    script: str,
    extra_env: dict[str, str],
    expected_output: Path,
) -> StepRecord:
    run_root.mkdir(parents=True, exist_ok=True)
    usage_log = run_root / "logs" / "openai_usage.jsonl"
    before_lines = usage_line_count(usage_log)
    start = time.monotonic()
    env = os.environ.copy()
    env.update(
        {
            "ONTOLOGYM_RUN_ID": run_root.name,
            "ONTOLOGYM_USAGE_LOG": str(usage_log),
        }
    )
    env.update(extra_env)
    command = [PYTHON, script]
    print(f"  - {name}: running {' '.join(command)}")
    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
    seconds = time.monotonic() - start
    after_lines = usage_line_count(usage_log)
    cost = usage_cost_delta(usage_log, before_lines, after_lines)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    if not expected_output.exists():
        raise FileNotFoundError(f"{name} finished but expected output is missing: {expected_output}")
    return StepRecord(
        name=name,
        status="completed",
        run_id=run_root.name,
        command=" ".join(command),
        output=str(expected_output),
        cost_usd=cost,
        seconds=seconds,
    )


def ensure_base_run(
    target: Path,
    source: Path,
    *,
    include_categories: bool = True,
    include_taxonomy: bool = True,
    include_relation: bool = True,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "logs").mkdir(exist_ok=True)
    copy_dir_if_missing(source / "ontogen" / "processed_data", target / "ontogen" / "processed_data")
    copy_dir_if_missing(source / "ontogen" / "enriched", target / "ontogen" / "enriched")
    copy_dir_if_missing(source / "ontogen" / "termo", target / "ontogen" / "termo")
    if include_categories:
        copy_dir_if_missing(source / "ontogen" / "categories", target / "ontogen" / "categories")
    if include_taxonomy:
        copy_dir_if_missing(source / "ontogen" / "taxonomy", target / "ontogen" / "taxonomy")
    if include_relation:
        copy_dir_if_missing(source / "relation_augmentation", target / "relation_augmentation")


def copy_dir_if_missing(source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.exists():
        raise FileNotFoundError(f"Cannot copy missing source directory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def ontogen_dirs(run_root: Path) -> list[Path]:
    return [run_root / "ontogen" / "taxonomy", run_root / "ontogen" / "termo"]


def baseline_dirs(run_root: Path) -> list[Path]:
    return [*ontogen_dirs(run_root), run_root / "relation_augmentation" / "outputs"]


def refined_dirs(run_root: Path, refine_dir_name: str) -> list[Path]:
    return [*baseline_dirs(run_root), run_root / refine_dir_name]


def iter_refined_dirs(run_root: Path, round_index: int) -> list[Path]:
    dirs = baseline_dirs(run_root)
    dirs.extend(run_root / f"kg_refinement_iter_r{index:02d}" for index in range(1, round_index + 1))
    return dirs


def usage_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def usage_cost_delta(path: Path, start_line: int, end_line: int) -> float:
    if not path.exists() or end_line <= start_line:
        return 0.0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(usage_record_cost(json.loads(line)) for line in lines[start_line:end_line])


def usage_record_cost(record: dict[str, Any]) -> float:
    model = str(record.get("model") or "")
    base = base_model_name(model)
    rates = USD_PER_MILLION.get(base)
    if not rates:
        return 0.0
    prompt_tokens = int(record.get("prompt_tokens") or 0)
    completion_tokens = int(record.get("completion_tokens") or 0)
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


def base_model_name(model: str) -> str:
    for known in USD_PER_MILLION:
        if model.startswith(known):
            return known
    return model


def read_eval_summary(path: Path) -> dict[str, Any]:
    manual_path = path.with_name("evaluation_results.manual.json")
    has_manual = manual_path.exists()
    if has_manual:
        path = manual_path
    data = read_json(path) if path.exists() else {}
    records = data.get("records", [])
    wrong = 0
    if has_manual:
        for record in records:
            score_info = record.get("airqa_score") if isinstance(record.get("airqa_score"), dict) else {}
            score = score_info.get("score")
            if score is not None and float(score) < 0.5:
                wrong += 1
    return {
        "dataset_path": data.get("dataset_path"),
        "examples": data.get("examples"),
        "average_score": data.get("average_score") if has_manual else None,
        "scored_examples": data.get("scored_examples") if has_manual else 0,
        "wrong_lt_0_5": wrong if has_manual else None,
        "manual_scoring": "complete" if has_manual else "missing",
        "kg_facts": data.get("kg_facts"),
        "kg_retrieval_mode": data.get("kg_retrieval_mode"),
        "kg_retrieval_strategy": data.get("kg_retrieval_strategy"),
    }


def manual_result_path(run_root: Path, phase: str) -> Path:
    return run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.manual.json"


def read_relation_summary(run_root: Path) -> dict[str, Any]:
    summary = run_root / "relation_augmentation" / "outputs" / "run_summary.json"
    if summary.exists():
        return read_json(summary)
    graph = run_root / "relation_augmentation" / "outputs" / "relation_graph.json"
    if not graph.exists():
        return {}
    data = read_json(graph)
    return {"nodes": len(data.get("nodes", [])), "edges": len(data.get("edges", []))}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh() -> bool:
    return os.getenv("ONTOLOGYM_SLEEP_EXPERIMENTS_REFRESH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def write_summary(records: list[StepRecord]) -> None:
    data = {
        "created_at_local_note": "2026-06-02",
        "fixed_answer_model": FIXED_ANSWER_MODEL,
        "fixed_judge_model": FIXED_JUDGE_MODEL,
        "manual40_dataset": str(MANUAL40),
        "final_holdout_dataset": str(FINAL_HOLDOUT),
        "refine_datasets": [str(path) for path in REFINE_DATASETS],
        "total_new_estimated_cost_usd": round(sum(record.cost_usd for record in records), 6),
        "records": [record.to_dict() for record in records],
        "tables": build_tables(records),
    }
    SUMMARY_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_TXT.write_text(render_text_summary(data), encoding="utf-8")


def build_tables(records: list[StepRecord]) -> dict[str, list[dict[str, Any]]]:
    rows = []
    for record in records:
        detail = record.detail or {}
        if "average_score" in detail:
            rows.append(
                {
                    "name": record.name,
                    "run_id": record.run_id,
                    "score": detail.get("average_score"),
                    "examples": detail.get("examples"),
                    "kg_facts": detail.get("kg_facts"),
                    "cost_usd": round(record.cost_usd, 6),
                    "status": record.status,
                }
            )
    return {"eval_scores": rows}


def render_text_summary(data: dict[str, Any]) -> str:
    lines = [
        "OntoloGYM sleep experiments summary",
        "",
        f"Fixed answerer: {data['fixed_answer_model']}",
        f"Fixed judge: {data['fixed_judge_model']}",
        f"New estimated cost: ${data['total_new_estimated_cost_usd']:.4f}",
        "",
        "Evaluation scores",
    ]
    for row in data["tables"]["eval_scores"]:
        score = row["score"]
        score_text = "NA" if score is None else f"{float(score):.4f}"
        lines.append(
            f"- {row['name']}: score={score_text}, examples={row['examples']}, "
            f"kg_facts={row['kg_facts']}, status={row['status']}, new_cost=${row['cost_usd']:.4f}"
        )
    lines.append("")
    lines.append("All step records are in the JSON summary.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
