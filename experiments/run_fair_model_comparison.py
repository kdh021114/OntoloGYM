"""Run a controlled KG-generation comparison across OpenAI models.

This script is intentionally narrower than run_reuse_qa_experiment.py:
it regenerates KG artifacts with the same current pipeline settings while
changing only the KG-generation model. QA answering/reporting and AirQA's
LLM evaluator are fixed so the comparison is about KG quality, not answerer
or judge differences.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.run_context import initialize_manifest, record_pipeline_run
from common.usage_costs import estimate_usage_cost


DATA_DIR = PROJECT_ROOT / "data"
SHARED_PAPER_DIR = DATA_DIR / "papers"

DEFAULT_SOURCE_RUN_ID = "run_20260526_032051"
DEFAULT_DATASET_PATH = (
    DATA_DIR
    / "run_qa_improved_20260530_014522"
    / "qa_extractor_improved_v4"
    / "qa_dataset.jsonl"
)
DEFAULT_MODELS = [
    ("nano", "gpt-5.4-nano"),
    ("mini", "gpt-5.4-mini"),
    ("gpt54", "gpt-5.4"),
]

MODEL_ENV_KEYS = [
    "ONTOLOGYM_MODEL",
    "ONTOLOGYM_ONTOGEN_MODEL",
    "ONTOLOGYM_ONTOGEN_TERMO_TERMS_MODEL",
    "ONTOLOGYM_ONTOGEN_TERMO_ACRONYMS_MODEL",
    "ONTOLOGYM_ONTOGEN_TERMO_DEFINITIONS_MODEL",
    "ONTOLOGYM_ONTOGEN_TERMO_RELATIONSHIPS_MODEL",
    "ONTOLOGYM_ONTOGEN_CATEGORY_GENERATION_MODEL",
    "ONTOLOGYM_ONTOGEN_CATEGORY_FORMAT_MODEL",
    "ONTOLOGYM_ONTOGEN_CATEGORY_SYNTHESIS_MODEL",
    "ONTOLOGYM_ONTOGEN_CATEGORY_CURATION_MODEL",
    "ONTOLOGYM_ONTOGEN_TAXONOMY_MODEL",
    "ONTOLOGYM_RELATION_MODEL",
]

SNAPSHOT_FILES = [
    "config.py",
    "configs/common.py",
    "configs/ontogen.py",
    "configs/relation_augmentation.py",
    "configs/qa_evaluation.py",
    "ontogen/run.py",
    "ontogen/run_termo.py",
    "ontogen/generate_categories.py",
    "ontogen/generate_taxonomy.py",
    "ontogen/termo/prompt.py",
    "relation_augmentation/llm.py",
    "relation_augmentation/schema.py",
    "qa_evaluation/graphrag.py",
    "qa_evaluation/pipeline.py",
    "qa_extractor/evaluation/llm_functions.py",
]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_models(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return DEFAULT_MODELS
    parsed: list[tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, model = item.split(":", 1)
        else:
            model = item
            label = model.replace(".", "").replace("-", "_")
        parsed.append((label.strip(), model.strip()))
    return parsed or DEFAULT_MODELS


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _copy_reused_qa(dataset_path: Path, run_root: Path) -> Path:
    target_dir = run_root / "qa_extractor"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "qa_dataset.jsonl"
    shutil.copy2(dataset_path, target_path)
    record_pipeline_run(
        run_root,
        "qa_dataset_reused",
        status="reused",
        inputs={"dataset_path": str(dataset_path)},
        outputs={"dataset_path": str(target_path)},
        extra={"examples": _count_jsonl(target_path)},
    )
    return target_path


def _prepare_reused_ontogen_inputs(source_run: Path, run_root: Path) -> Path:
    source_processed_dir = source_run / "ontogen" / "processed_data"
    source_enriched_dir = source_run / "ontogen" / "enriched"
    if not source_processed_dir.exists():
        raise FileNotFoundError(f"Missing source processed_data directory: {source_processed_dir}")

    target_processed_dir = run_root / "ontogen" / "processed_data"
    target_enriched_dir = run_root / "ontogen" / "enriched"
    paper_input_dir = run_root / "reused_paper_inputs"

    for path in [target_processed_dir, target_enriched_dir, paper_input_dir]:
        path.mkdir(parents=True, exist_ok=True)

    processed_files = sorted(source_processed_dir.glob("*.processed_data.json"))
    if not processed_files:
        raise FileNotFoundError(f"No processed_data JSON files found in {source_processed_dir}")

    for source_file in processed_files:
        paper_id = source_file.name.removesuffix(".processed_data.json")
        target_file = target_processed_dir / source_file.name
        shutil.copy2(source_file, target_file)

        source_enriched = source_enriched_dir / f"{paper_id}.enriched.txt"
        if source_enriched.exists():
            shutil.copy2(source_enriched, target_enriched_dir / source_enriched.name)

        paper_dir = paper_input_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        input_json = paper_dir / f"{paper_id}.json"
        input_json.write_text(
            json.dumps({"reused_processed_data": str(target_file)}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    record_pipeline_run(
        run_root,
        "ontogen_inputs_reused",
        status="reused",
        inputs={
            "source_processed_dir": str(source_processed_dir),
            "source_enriched_dir": str(source_enriched_dir),
        },
        outputs={
            "processed_data_dir": str(target_processed_dir),
            "enriched_dir": str(target_enriched_dir),
            "paper_input_dir": str(paper_input_dir),
        },
        extra={"papers": len(processed_files)},
    )
    return paper_input_dir


def _source_snapshot() -> dict[str, Any]:
    files = {}
    for rel_path in SNAPSHOT_FILES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            files[rel_path] = {"exists": False}
            continue
        data = path.read_bytes()
        files[rel_path] = {
            "exists": True,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    return {"files": files}


def _write_snapshot(run_root: Path, *, label: str, model: str, dataset_path: Path, source_run: Path) -> None:
    snapshot = {
        "label": label,
        "kg_generation_model": model,
        "dataset_path": str(dataset_path),
        "source_run": str(source_run),
        "source_snapshot": _source_snapshot(),
    }
    path = run_root / "fair_comparison_snapshot.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _base_env(run_root: Path, model: str, paper_input_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ONTOLOGYM_RUN_ID"] = run_root.name
    env["ONTOLOGYM_SHARED_PAPER_DIR"] = str(paper_input_dir)
    env["ONTOLOGYM_ONTOGEN_REUSE_INTERMEDIATE_OUTPUTS"] = "1"
    env["ONTOLOGYM_QA_EVAL_DRY_RUN"] = "0"
    env["ONTOLOGYM_REUSE_QA_SKIP_COMPLETED"] = "0"
    for key in MODEL_ENV_KEYS:
        env[key] = model
    return env


def _eval_env(
    base_env: dict[str, str],
    *,
    dataset_path: Path,
    phase: str,
    answer_model: str,
    judge_model: str,
    run_root: Path,
) -> dict[str, str]:
    kg_dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    env = dict(base_env)
    env.update(
        {
            "ONTOLOGYM_QA_EVAL_PHASE": phase,
            "ONTOLOGYM_QA_EVAL_DATASET_PATH": str(dataset_path),
            "ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS": os.pathsep.join(str(path) for path in kg_dirs),
            "ONTOLOGYM_QA_EVAL_MODEL": answer_model,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_REPORT_MODEL": answer_model,
            "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL": judge_model,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS": "1",
            "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES": "0",
        }
    )
    return env


def _run(script: str, env: dict[str, str]) -> None:
    print(f"Running {script} for {env['ONTOLOGYM_RUN_ID']} ...", flush=True)
    subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env, check=True)


def _run_unless_complete(script: str, env: dict[str, str], completion_path: Path) -> None:
    if _truthy(os.getenv("ONTOLOGYM_FAIR_SKIP_COMPLETED", "1")) and completion_path.exists():
        print(
            f"Skipping {script} for {env['ONTOLOGYM_RUN_ID']} "
            f"(found {completion_path}).",
            flush=True,
        )
        return
    _run(script, env)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _kg_stats(run_root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    graph = _read_json(run_root / "kg_visualization" / "kg_graph.json")
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    stats["nodes"] = len(nodes)
    stats["edges"] = len(edges)
    stats["relations"] = dict(Counter(edge.get("relation") for edge in edges))
    stats["provenance"] = dict(Counter(edge.get("provenance") for edge in edges))

    rel_graph = _read_json(run_root / "relation_augmentation" / "outputs" / "relation_graph.json")
    rel_edges = rel_graph.get("edges", [])
    stats["relation_graph_edges"] = len(rel_edges)
    stats["relation_graph_relations"] = dict(Counter(edge.get("relation") for edge in rel_edges))
    return stats


def _eval_stats(run_root: Path, phase: str) -> dict[str, Any]:
    results = _read_json(run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json")
    return {
        "average_score": results.get("average_score"),
        "evaluated_examples": results.get("evaluated_examples"),
        "scored_examples": results.get("scored_examples"),
        "kg_facts": results.get("kg_facts"),
        "kg_communities": results.get("kg_communities"),
        "results_json": str(run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"),
    }


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Fair Model Comparison Summary",
        f"group_id: {summary['group_id']}",
        f"dataset: {summary['dataset_path']}",
        f"answer_model: {summary['answer_model']}",
        f"judge_model: {summary['judge_model']}",
        "",
    ]
    for item in summary["runs"]:
        lines.append(f"{item['label']} ({item['model']}): {item['run_id']}")
        eval_stats = item.get("eval", {})
        kg_stats = item.get("kg", {})
        usage = item.get("usage", {})
        lines.append(
            "  "
            f"score={eval_stats.get('average_score')}, "
            f"examples={eval_stats.get('evaluated_examples')}, "
            f"kg_facts={eval_stats.get('kg_facts')}, "
            f"nodes={kg_stats.get('nodes')}, edges={kg_stats.get('edges')}"
        )
        lines.append(
            "  "
            f"usage=input {usage.get('prompt_tokens', 0)}, "
            f"output {usage.get('completion_tokens', 0)}, "
            f"cost ${usage.get('estimated_cost_usd', 0.0):.4f}"
        )
    lines.append("")
    lines.append(f"total_estimated_cost_usd: {summary['total_estimated_cost_usd']:.4f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    source_run_id = os.getenv("ONTOLOGYM_FAIR_SOURCE_RUN", DEFAULT_SOURCE_RUN_ID).strip() or DEFAULT_SOURCE_RUN_ID
    source_run = DATA_DIR / source_run_id
    dataset_path = Path(os.getenv("ONTOLOGYM_FAIR_DATASET_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    answer_model = os.getenv("ONTOLOGYM_FAIR_ANSWER_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    judge_model = os.getenv("ONTOLOGYM_FAIR_JUDGE_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    phase = os.getenv("ONTOLOGYM_FAIR_EVAL_PHASE", "fair_fixed_answer_mini_judge_gpt54").strip()
    models = _parse_models(os.getenv("ONTOLOGYM_FAIR_MODELS"))
    budget = float(os.getenv("ONTOLOGYM_FAIR_BUDGET_USD", "30"))

    group_id = os.getenv("ONTOLOGYM_FAIR_GROUP_ID", "").strip()
    if not group_id:
        group_id = "fair_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline_order = [
        "qa_dataset_reused",
        "ontogen_inputs_reused",
        "ontogen",
        "relation_augmentation",
        "kg_visualization",
        f"qa_evaluation_{phase}",
    ]

    summary: dict[str, Any] = {
        "group_id": group_id,
        "source_run": str(source_run),
        "dataset_path": str(dataset_path),
        "answer_model": answer_model,
        "judge_model": judge_model,
        "phase": phase,
        "runs": [],
        "total_estimated_cost_usd": 0.0,
    }

    for label, model in models:
        if summary["total_estimated_cost_usd"] > budget:
            print(f"Budget already exceeded (${summary['total_estimated_cost_usd']:.2f}); stopping before {label}.")
            break

        run_id = f"run_{group_id}_{label}"
        run_root = DATA_DIR / run_id
        if run_root.exists() and _truthy(os.getenv("ONTOLOGYM_FAIR_REFRESH", "0")):
            shutil.rmtree(run_root)
        initialize_manifest(run_root, project_root=PROJECT_ROOT, paper_dir=SHARED_PAPER_DIR, pipeline_order=pipeline_order)

        reused_dataset = _copy_reused_qa(dataset_path, run_root)
        paper_input_dir = _prepare_reused_ontogen_inputs(source_run, run_root)
        _write_snapshot(run_root, label=label, model=model, dataset_path=dataset_path, source_run=source_run)

        env = _base_env(run_root, model, paper_input_dir)
        _run_unless_complete("run_ontogen.py", env, run_root / "ontogen" / "taxonomy" / "tree_0.pkl")
        _run_unless_complete(
            "run_relation_augmentation.py",
            env,
            run_root / "relation_augmentation" / "outputs" / "relation_graph.json",
        )
        _run_unless_complete("run_kg_visualization.py", env, run_root / "kg_visualization" / "kg_graph.json")

        eval_env = _eval_env(
            env,
            dataset_path=reused_dataset,
            phase=phase,
            answer_model=answer_model,
            judge_model=judge_model,
            run_root=run_root,
        )
        _run_unless_complete(
            "run_qa_evaluation.py",
            eval_env,
            run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json",
        )

        usage = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
        summary["total_estimated_cost_usd"] += usage["estimated_cost_usd"]
        item = {
            "label": label,
            "model": model,
            "run_id": run_id,
            "run_root": str(run_root),
            "kg": _kg_stats(run_root),
            "eval": _eval_stats(run_root, phase),
            "usage": usage,
        }
        summary["runs"].append(item)
        print(_format_summary({**summary, "runs": [item]}), flush=True)

    out_json = DATA_DIR / f"{group_id}_summary.json"
    out_txt = DATA_DIR / f"{group_id}_summary.txt"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_txt.write_text(_format_summary(summary), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_txt}", flush=True)


if __name__ == "__main__":
    main()
