"""Run fixed-model GraphRAG evaluation across the main experiment runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.usage_costs import USD_PER_MILLION


DATA_DIR = PROJECT_ROOT / "data"
FIXED_MODEL = os.getenv("ONTOLOGYM_FIXED_EVAL_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
_default_phase_prefix = f"fixed_graphrag_global_{FIXED_MODEL.replace('.', '').replace('-', '')}"
PHASE_PREFIX = os.getenv("ONTOLOGYM_FIXED_EVAL_PHASE_PREFIX", _default_phase_prefix).strip() or _default_phase_prefix

RUNS = [
    ("nano", "run_20260526_032051"),
    ("mini", "run_mini_20260528_082107"),
    ("gpt54", "run_gpt54_20260529_161017"),
]

PHASES = [
    ("base", "qa_extractor/qa_dataset.jsonl", False),
    ("holdout_baseline", "qa_extractor_refine_holdout/qa_dataset.jsonl", False),
    ("holdout_refined", "qa_extractor_refine_holdout/qa_dataset.jsonl", True),
]


def main() -> None:
    results: dict[str, Any] = {
        "fixed_model": FIXED_MODEL,
        "phase_prefix": PHASE_PREFIX,
        "runs": {},
    }
    for label, run_id in RUNS:
        run_root = DATA_DIR / run_id
        results["runs"][label] = _evaluate_run(label, run_id, run_root)

    summary_txt = _format_summary(results)
    summary_json = DATA_DIR / f"{PHASE_PREFIX}_summary.json"
    summary_path = DATA_DIR / f"{PHASE_PREFIX}_summary.txt"
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(summary_txt + "\n", encoding="utf-8")
    print(summary_txt)
    print(f"\nWrote {summary_path}")


def _evaluate_run(label: str, run_id: str, run_root: Path) -> dict[str, Any]:
    if not run_root.exists():
        return {"status": "missing", "run_id": run_id}

    usage_log = run_root / "logs" / "openai_usage.jsonl"
    start_line = _line_count(usage_log)
    run_result: dict[str, Any] = {
        "status": "completed",
        "run_id": run_id,
        "run_root": str(run_root),
        "phases": {},
    }

    for phase_name, dataset_relpath, include_refined in PHASES:
        phase = f"{PHASE_PREFIX}_{phase_name}"
        dataset_path = run_root / dataset_relpath
        output_path = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
        kg_dirs = _kg_dirs(run_root, include_refined=include_refined)
        _validate_inputs(dataset_path, kg_dirs)
        if not output_path.exists():
            _run_eval_subprocess(
                run_id=run_id,
                phase=phase,
                dataset_path=dataset_path,
                kg_dirs=kg_dirs,
            )
        phase_result = _read_json(output_path)
        run_result["phases"][phase_name] = {
            "phase": phase,
            "results_json": str(output_path),
            "average_score": phase_result.get("average_score"),
            "evaluated_examples": phase_result.get("evaluated_examples"),
            "kg_facts": phase_result.get("kg_facts"),
            "kg_communities": phase_result.get("kg_communities"),
            "kg_context_mode": phase_result.get("kg_context_mode"),
        }

    run_result["fixed_eval_usage"] = _usage_since(usage_log, start_line)
    return run_result


def _kg_dirs(run_root: Path, *, include_refined: bool) -> list[Path]:
    dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    if include_refined:
        dirs.append(run_root / "kg_refinement")
    return dirs


def _validate_inputs(dataset_path: Path, kg_dirs: list[Path]) -> None:
    missing = [str(dataset_path)] if not dataset_path.exists() else []
    missing.extend(str(path) for path in kg_dirs if not path.exists())
    if missing:
        raise FileNotFoundError("Missing fixed-eval inputs:\n" + "\n".join(missing))


def _run_eval_subprocess(run_id: str, phase: str, dataset_path: Path, kg_dirs: list[Path]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "ONTOLOGYM_RUN_ID": run_id,
            "ONTOLOGYM_QA_EVAL_PHASE": phase,
            "ONTOLOGYM_QA_EVAL_DATASET_PATH": os.fspath(dataset_path),
            "ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS": os.pathsep.join(os.fspath(path) for path in kg_dirs),
            "ONTOLOGYM_QA_EVAL_MODEL": FIXED_MODEL,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_REPORT_MODEL": FIXED_MODEL,
            "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL": FIXED_MODEL,
            "ONTOLOGYM_QA_EVAL_DRY_RUN": "0",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS": "1",
            "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES": "0",
        }
    )
    print(f"Running fixed GraphRAG eval: run={run_id}, phase={phase}, model={FIXED_MODEL}", flush=True)
    subprocess.run([sys.executable, "run_qa_evaluation.py"], cwd=PROJECT_ROOT, env=env, check=True)


def _format_summary(results: dict[str, Any]) -> str:
    lines = [
        "Fixed-Model GraphRAG Evaluation Summary",
        f"Fixed model: {results['fixed_model']}",
        "",
    ]
    for label, run in results["runs"].items():
        lines.append(f"{label}: {run.get('run_id')}")
        if run.get("status") != "completed":
            lines.append(f"  status={run.get('status')}")
            continue
        for phase_name, phase in run["phases"].items():
            lines.append(
                "  "
                f"{phase_name}: score={_fmt_score(phase.get('average_score'))}, "
                f"examples={phase.get('evaluated_examples')}, "
                f"kg_edges={phase.get('kg_facts')}, "
                f"communities={phase.get('kg_communities')}, "
                f"mode={phase.get('kg_context_mode')}"
            )
        usage = run["fixed_eval_usage"]
        lines.append(
            "  fixed_eval_usage: "
            f"records={usage['records']}, "
            f"input={usage['prompt_tokens']}, "
            f"output={usage['completion_tokens']}, "
            f"cost=${usage['estimated_cost_usd']:.4f}"
        )
        lines.append("")

    for phase_name, _, _ in PHASES:
        scores = []
        for label in ("nano", "mini", "gpt54"):
            value = results["runs"].get(label, {}).get("phases", {}).get(phase_name, {}).get("average_score")
            scores.append((label, _as_float(value)))
        ordered = all(score is not None for _, score in scores) and scores[0][1] < scores[1][1] < scores[2][1]
        joined = ", ".join(f"{label}={_fmt_score(score)}" for label, score in scores)
        lines.append(f"order_check/{phase_name}: {joined}; nano<mini<gpt54={ordered}")
    return "\n".join(lines).rstrip()


def _usage_since(usage_log: Path, start_line: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "by_model": {},
    }
    if not usage_log.exists():
        return summary
    lines = usage_log.read_text(encoding="utf-8").splitlines()[start_line:]
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        model = str(record.get("model") or "unknown")
        prompt_tokens = int(record.get("prompt_tokens") or 0)
        completion_tokens = int(record.get("completion_tokens") or 0)
        total_tokens = int(record.get("total_tokens") or prompt_tokens + completion_tokens)
        rates = USD_PER_MILLION.get(model, USD_PER_MILLION.get(_base_model_name(model), {"input": 0.0, "output": 0.0}))
        cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000
        summary["records"] += 1
        summary["prompt_tokens"] += prompt_tokens
        summary["completion_tokens"] += completion_tokens
        summary["total_tokens"] += total_tokens
        summary["estimated_cost_usd"] += cost
        by_model = summary["by_model"].setdefault(
            model,
            {"records": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
        )
        by_model["records"] += 1
        by_model["prompt_tokens"] += prompt_tokens
        by_model["completion_tokens"] += completion_tokens
        by_model["total_tokens"] += total_tokens
        by_model["estimated_cost_usd"] += cost
    return summary


def _base_model_name(model: str) -> str:
    for known in USD_PER_MILLION:
        if model.startswith(known):
            return known
    return model


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_score(value: Any) -> str:
    score = _as_float(value)
    return "none" if score is None else f"{score:.4f}"


if __name__ == "__main__":
    main()
