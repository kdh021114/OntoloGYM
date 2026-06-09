"""Compare relation-augmentation models with a fixed OntoGen baseline.

This experiment keeps OntoGen outputs constant so taxonomy/TERMO size does not
become a hidden variable. Each model only regenerates relation augmentation, then
the same GraphRAG QA evaluation setup is used for every run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from common.run_context import initialize_manifest, record_pipeline_run
from common.usage_costs import estimate_usage_cost


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_BASELINE_RUN_ID = "run_fair_20260530_103255_gpt54"
DEFAULT_DATASET_PATH = (
    DATA_DIR
    / "run_qa_balanced_mini_20260530_135000"
    / "qa_extractor_balanced_v1"
    / "qa_dataset_single_only.jsonl"
)
DEFAULT_MODELS = [
    ("nano", "gpt-5.4-nano"),
    ("mini", "gpt-5.4-mini"),
    ("gpt54", "gpt-5.4"),
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


def _copy_dataset(dataset_path: Path, run_root: Path) -> Path:
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


def _copy_baseline_ontogen(baseline_run: Path, run_root: Path) -> None:
    source_ontogen = baseline_run / "ontogen"
    if not (source_ontogen / "taxonomy" / "tree_0.pkl").exists():
        raise FileNotFoundError(f"Missing baseline taxonomy: {source_ontogen / 'taxonomy' / 'tree_0.pkl'}")
    if not (source_ontogen / "termo").exists():
        raise FileNotFoundError(f"Missing baseline TERMO directory: {source_ontogen / 'termo'}")
    if not (source_ontogen / "processed_data").exists():
        raise FileNotFoundError(f"Missing baseline processed_data directory: {source_ontogen / 'processed_data'}")

    target_ontogen = run_root / "ontogen"
    if target_ontogen.exists():
        shutil.rmtree(target_ontogen)
    shutil.copytree(source_ontogen, target_ontogen)
    record_pipeline_run(
        run_root,
        "ontogen_baseline_reused",
        status="reused",
        inputs={"baseline_ontogen_dir": str(source_ontogen)},
        outputs={"ontogen_dir": str(target_ontogen)},
        extra={
            "taxonomy_tree": str(target_ontogen / "taxonomy" / "tree_0.pkl"),
            "termo_dir": str(target_ontogen / "termo"),
            "processed_data_dir": str(target_ontogen / "processed_data"),
        },
    )


def _base_env(run_root: Path, *, relation_model: str) -> dict[str, str]:
    env = os.environ.copy()
    env["ONTOLOGYM_RUN_ID"] = run_root.name
    env["ONTOLOGYM_USAGE_LOG"] = str(run_root / "logs" / "openai_usage.jsonl")
    env["ONTOLOGYM_MODEL"] = relation_model
    env["ONTOLOGYM_RELATION_MODEL"] = relation_model
    env["ONTOLOGYM_REUSE_QA_SKIP_COMPLETED"] = "0"
    return env


def _eval_env(
    base_env: dict[str, str],
    *,
    dataset_path: Path,
    run_root: Path,
    phase: str,
    answer_model: str,
    judge_model: str,
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
            "ONTOLOGYM_QA_EVAL_DRY_RUN": "0",
            "ONTOLOGYM_QA_EVAL_MODEL": answer_model,
            "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL": judge_model,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS": "0",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS": "1",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_BM25_WEIGHT": "0.4",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT": "0.6",
            "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES": "0",
        }
    )
    return env


def _run(script: str, env: dict[str, str]) -> None:
    print(f"Running {script} for {env['ONTOLOGYM_RUN_ID']} ...", flush=True)
    subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env, check=True)


def _run_unless_complete(script: str, env: dict[str, str], completion_path: Path) -> None:
    if completion_path.exists() and _truthy(os.getenv("ONTOLOGYM_AUG_COMPARE_SKIP_COMPLETED", "1")):
        print(f"Skipping {script}; found {completion_path}", flush=True)
        return
    _run(script, env)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _relation_stats(run_root: Path) -> dict[str, Any]:
    graph = _read_json(run_root / "relation_augmentation" / "outputs" / "relation_graph.json")
    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "relations": dict(Counter(edge.get("relation") for edge in edges)),
    }


def _eval_stats(run_root: Path, phase: str) -> dict[str, Any]:
    path = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
    results = _read_json(path)
    return {
        "average_score": results.get("average_score"),
        "evaluated_examples": results.get("evaluated_examples"),
        "scored_examples": results.get("scored_examples"),
        "kg_facts": results.get("kg_facts"),
        "kg_communities": results.get("kg_communities"),
        "results_json": str(path),
    }


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Augmentation Model Comparison Summary",
        f"group_id: {summary['group_id']}",
        f"baseline_ontogen_run: {summary['baseline_ontogen_run']}",
        f"dataset: {summary['dataset_path']}",
        f"answer_model: {summary['answer_model']}",
        f"judge_model: {summary['judge_model']}",
        "",
    ]
    for item in summary["runs"]:
        eval_stats = item.get("eval", {})
        rel_stats = item.get("relation_augmentation", {})
        lines.append(f"{item['label']} ({item['model']}): {item['run_id']}")
        lines.append(
            "  "
            f"score={eval_stats.get('average_score')}, "
            f"examples={eval_stats.get('evaluated_examples')}, "
            f"kg_facts={eval_stats.get('kg_facts')}, "
            f"relation_edges={rel_stats.get('edges')}"
        )
        lines.append(f"  estimated_cost_usd=${item.get('estimated_cost_usd', 0.0):.4f}")
    lines.append("")
    lines.append(f"total_estimated_cost_usd=${summary['total_estimated_cost_usd']:.4f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    baseline_run_id = os.getenv("ONTOLOGYM_AUG_BASELINE_RUN", DEFAULT_BASELINE_RUN_ID).strip() or DEFAULT_BASELINE_RUN_ID
    baseline_run = DATA_DIR / baseline_run_id
    if not baseline_run.exists():
        raise FileNotFoundError(f"Missing baseline run: {baseline_run}")

    dataset_path = Path(os.getenv("ONTOLOGYM_AUG_DATASET_PATH", str(DEFAULT_DATASET_PATH))).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    answer_model = os.getenv("ONTOLOGYM_AUG_ANSWER_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    judge_model = os.getenv("ONTOLOGYM_AUG_JUDGE_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    phase = os.getenv("ONTOLOGYM_AUG_EVAL_PHASE", "baseline_ontogen_aug_only_single40").strip()
    models = _parse_models(os.getenv("ONTOLOGYM_AUG_MODELS"))
    budget = float(os.getenv("ONTOLOGYM_AUG_BUDGET_USD", "30"))

    group_id = os.getenv("ONTOLOGYM_AUG_GROUP_ID", "").strip()
    if not group_id:
        group_id = "aug_baseline_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    pipeline_order = [
        "qa_dataset_reused",
        "ontogen_baseline_reused",
        "relation_augmentation",
        f"qa_evaluation_{phase}",
    ]
    summary: dict[str, Any] = {
        "group_id": group_id,
        "baseline_ontogen_run": baseline_run_id,
        "baseline_ontogen_dir": str(baseline_run / "ontogen"),
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
        if run_root.exists() and _truthy(os.getenv("ONTOLOGYM_AUG_REFRESH", "0")):
            shutil.rmtree(run_root)
        initialize_manifest(
            run_root,
            project_root=PROJECT_ROOT,
            paper_dir=baseline_run / "reused_paper_inputs",
            pipeline_order=pipeline_order,
        )

        reused_dataset = _copy_dataset(dataset_path, run_root)
        if not (run_root / "ontogen" / "taxonomy" / "tree_0.pkl").exists():
            _copy_baseline_ontogen(baseline_run, run_root)

        env = _base_env(run_root, relation_model=model)
        _run_unless_complete(
            "run_relation_augmentation.py",
            env,
            run_root / "relation_augmentation" / "outputs" / "relation_graph.json",
        )

        eval_env = _eval_env(
            env,
            dataset_path=reused_dataset,
            run_root=run_root,
            phase=phase,
            answer_model=answer_model,
            judge_model=judge_model,
        )
        _run_unless_complete(
            "run_qa_evaluation.py",
            eval_env,
            run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json",
        )

        usage = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
        item = {
            "label": label,
            "model": model,
            "run_id": run_id,
            "run_root": str(run_root),
            "relation_augmentation": _relation_stats(run_root),
            "eval": _eval_stats(run_root, phase),
            "estimated_cost_usd": usage["estimated_cost_usd"],
        }
        summary["total_estimated_cost_usd"] += item["estimated_cost_usd"]
        summary["runs"].append(item)
        print(_format_summary({**summary, "runs": [item], "total_estimated_cost_usd": item["estimated_cost_usd"]}), flush=True)

    out_json = DATA_DIR / f"{group_id}_summary.json"
    out_txt = DATA_DIR / f"{group_id}_summary.txt"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_txt.write_text(_format_summary(summary), encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_txt}", flush=True)


if __name__ == "__main__":
    main()
