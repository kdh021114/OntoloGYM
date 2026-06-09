"""Summarize OntoloGYM experiment outputs into a text report."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from common.usage_costs import estimate_usage_cost


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_NANO_RUN_ID = "run_20260526_032051"


def main() -> None:
    nano_run = DATA_DIR / (os.getenv("ONTOLOGYM_REPORT_NANO_RUN", DEFAULT_NANO_RUN_ID).strip() or DEFAULT_NANO_RUN_ID)
    mini_run = _resolve_mini_run()
    comparison_runs: list[tuple[str, Path]] = []
    if mini_run:
        comparison_runs.append(("Mini reused-QA run", mini_run))
    comparison_runs.extend(_resolve_extra_runs())

    output_override = os.getenv("ONTOLOGYM_REPORT_OUTPUT", "").strip()
    output_path = Path(output_override).expanduser() if output_override else None
    if output_path is None:
        output_path = (comparison_runs[-1][1] if comparison_runs else nano_run) / "experiment_summary.txt"

    lines = []
    lines.append("OntoloGYM Experiment Summary")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.extend(_summarize_run("Nano reference run", nano_run))
    for title, run_root in comparison_runs:
        lines.append("")
        lines.extend(_summarize_run(title, run_root))
    lines.append("")
    if comparison_runs:
        lines.extend(_compare_runs(nano_run, comparison_runs))
    else:
        lines.append("Mini reused-QA run: not found yet.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output_path)


def _resolve_mini_run() -> Path | None:
    explicit = os.getenv("ONTOLOGYM_REPORT_MINI_RUN", "").strip()
    if explicit:
        path = DATA_DIR / explicit
        return path if path.exists() else None
    candidates = sorted(
        [path for path in DATA_DIR.glob("run_mini_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_extra_runs() -> list[tuple[str, Path]]:
    raw = os.getenv("ONTOLOGYM_REPORT_EXTRA_RUNS", "").strip()
    if not raw:
        return []
    runs: list[tuple[str, Path]] = []
    seen = {(_resolve_mini_run() or Path()).name}
    for item in raw.split(os.pathsep):
        run_id = item.strip()
        if not run_id or run_id in seen:
            continue
        run_root = DATA_DIR / run_id
        if not run_root.exists():
            continue
        runs.append((_title_for_run(run_root), run_root))
        seen.add(run_id)
    return runs


def _title_for_run(run_root: Path) -> str:
    usage = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
    model_names = " ".join(usage.get("by_model", {}).keys()).lower()
    run_name = run_root.name.lower()
    if "gpt54" in run_name or "gpt-5.4" in run_name:
        return "GPT-5.4 reused-QA run"
    if "mini" in run_name or "gpt-5.4-mini" in model_names:
        return "Mini reused-QA run"
    if "nano" in run_name or "gpt-5.4-nano" in model_names:
        return "Nano reused-QA run"
    if "gpt-5.4" in model_names:
        return "GPT-5.4 reused-QA run"
    return f"{run_root.name} reused-QA run"


def _summarize_run(title: str, run_root: Path) -> list[str]:
    lines = [title, f"Run: {run_root}"]
    if not run_root.exists():
        lines.append("Status: missing")
        return lines

    qa_base = run_root / "qa_extractor" / "qa_dataset.jsonl"
    qa_holdout = run_root / "qa_extractor_refine_holdout" / "qa_dataset.jsonl"
    lines.append(f"Base QA examples: {_count_jsonl(qa_base)}")
    lines.append(f"Holdout QA examples: {_count_jsonl(qa_holdout)}")

    legacy_phases = [
        ("legacy/base", "qa_evaluation"),
        ("legacy/holdout baseline", "qa_evaluation_holdout_baseline"),
        ("legacy/holdout refined", "qa_evaluation_holdout_refined"),
    ]
    for label, directory in legacy_phases:
        line = _legacy_phase_line(label, run_root / directory / "outputs" / "evaluation_results.json")
        if line:
            lines.append(line)

    graphrag_phases = [
        ("graphrag/base", ["qa_evaluation_graphrag_base", "qa_evaluation"]),
        ("graphrag/holdout baseline", ["qa_evaluation_graphrag_holdout_baseline", "qa_evaluation_holdout_baseline"]),
        ("graphrag/holdout refined", ["qa_evaluation_graphrag_holdout_refined", "qa_evaluation_holdout_refined"]),
    ]
    for label, directories in graphrag_phases:
        lines.append(_phase_line_from_candidates(label, run_root, directories))

    relation_summary = _read_json(run_root / "relation_augmentation" / "outputs" / "run_summary.json")
    if relation_summary:
        lines.append(
            "Relation augmentation: "
            f"accepted={relation_summary.get('accepted_claims')}, "
            f"rejected={relation_summary.get('rejected_claims')}"
        )
    refinement = _read_json(run_root / "kg_refinement" / "refinement_report.json")
    if refinement:
        lines.append(
            "KG refinement: "
            f"wrong_records={refinement.get('wrong_records')}, "
            f"accepted={refinement.get('accepted_claims')}, "
            f"rejected={refinement.get('rejected_claims')}"
        )

    usage = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
    lines.append(
        "Usage: "
        f"records={usage['records']}, "
        f"input={usage['prompt_tokens']}, "
        f"output={usage['completion_tokens']}, "
        f"estimated_cost_usd={usage['estimated_cost_usd']:.4f}"
    )
    return lines


def _compare_runs(nano_run: Path, comparison_runs: list[tuple[str, Path]]) -> list[str]:
    lines = ["Comparison vs Nano"]
    comparisons = [
        ("graphrag/base", ["qa_evaluation_graphrag_base", "qa_evaluation"]),
        ("graphrag/holdout baseline", ["qa_evaluation_graphrag_holdout_baseline", "qa_evaluation_holdout_baseline"]),
        ("graphrag/holdout refined", ["qa_evaluation_graphrag_holdout_refined", "qa_evaluation_holdout_refined"]),
    ]
    for label, directories in comparisons:
        nano_score = _score_from_candidates(nano_run, directories)
        if nano_score is None:
            lines.append(f"{label}: comparison pending")
            continue
        parts = [f"{label}: nano={nano_score:.4f}"]
        for title, run_root in comparison_runs:
            score = _score_from_candidates(run_root, directories)
            short_title = title.split()[0].lower().replace("gpt-5.4", "gpt54")
            if score is None:
                parts.append(f"{short_title}=pending")
            else:
                parts.append(f"{short_title}={score:.4f}, delta={score - nano_score:+.4f}")
        lines.append("; ".join(parts))
    return lines


def _legacy_phase_line(label: str, result_path: Path) -> str | None:
    result = _read_json(result_path)
    if not result or result.get("kg_context_mode") == "graphrag":
        return None
    return _format_phase_result(label, result, "completed")


def _phase_line_from_candidates(label: str, run_root: Path, directories: list[str]) -> str:
    for directory in directories:
        result = _read_json(run_root / directory / "outputs" / "evaluation_results.json")
        if result and result.get("kg_context_mode") == "graphrag":
            return _format_phase_result(label, result, "completed")
    for directory in directories:
        dry = _read_json(run_root / directory / "outputs" / "dry_run_summary.json")
        if dry and dry.get("kg_context_mode") == "graphrag":
            return _format_phase_result(label, dry, "dry_run")
    return f"{label}: missing"


def _format_phase_result(label: str, data: dict[str, Any], status: str) -> str:
    score = f", score={_format_score(data.get('average_score'))}" if "average_score" in data else ""
    examples = data.get("evaluated_examples", data.get("examples"))
    return (
        f"{label}: {status}{score}, "
        f"examples={examples}, "
        f"kg_edges={data.get('kg_facts')}, "
        f"communities={data.get('kg_communities', 'n/a')}, "
        f"mode={data.get('kg_context_mode', 'legacy')}"
    )


def _score(path: Path) -> float | None:
    data = _read_json(path)
    value = data.get("average_score") if data else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_from_candidates(run_root: Path, directories: list[str]) -> float | None:
    for directory in directories:
        data = _read_json(run_root / directory / "outputs" / "evaluation_results.json")
        if data and data.get("kg_context_mode") == "graphrag":
            return _score(run_root / directory / "outputs" / "evaluation_results.json")
    return None


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "none"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


if __name__ == "__main__":
    main()
