"""Run the hysteresis-loop KG ablation matrix.

For each OpenAI model, this runner reuses the corresponding existing OntoGen
and base relation KG, generates extra KG from the separated hysteresis figure
captions, generates extra KG from caption+image evidence, and evaluates all
three KG variants on the 40 text + 40 image QA dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.run_context import initialize_manifest, record_pipeline_run
from common.usage_costs import estimate_usage_cost


DATA_DIR = PROJECT_ROOT / "data"
PYTHON = sys.executable

DEFAULT_SOURCE_RUNS = {
    "nano": DATA_DIR / "run_pre_refine_20260601_014021_current_nano",
    "mini": DATA_DIR / "run_pre_refine_20260601_014021_current_mini",
    "gpt54": DATA_DIR / "run_pre_refine_20260601_014021_current_gpt54",
}
DEFAULT_MODELS = [
    ("nano", "gpt-5.4-nano"),
    ("mini", "gpt-5.4-mini"),
    ("gpt54", "gpt-5.4"),
]
DEFAULT_ASSET_MANIFEST = (
    DATA_DIR
    / "hysteresis_ablation_manual_20260604"
    / "assets"
    / "hysteresis_assets.jsonl"
)
DEFAULT_DATASET = (
    DATA_DIR
    / "hysteresis_ablation_manual_20260604"
    / "qa"
    / "qa_dataset_hysteresis_augmented_80.jsonl"
)

RELATION_PHASES = {
    "caption_kg": {
        "phase": "hysteresis_caption",
        "include_captions": "1",
        "include_images": "0",
    },
    "caption_image_kg": {
        "phase": "hysteresis_caption_image",
        "include_captions": "1",
        "include_images": "1",
    },
}

EVAL_VARIANTS = {
    "existing_kg": "hysteresis_existing_kg",
    "caption_kg": "hysteresis_caption_kg",
    "caption_image_kg": "hysteresis_caption_image_kg",
}


@dataclass
class StepRecord:
    name: str
    status: str
    run_id: str
    output: str | None = None
    seconds: float = 0.0
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "run_id": self.run_id,
            "output": self.output,
            "seconds": round(self.seconds, 2),
            "detail": self.detail or {},
        }


def main() -> None:
    group_id = os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_GROUP_ID", "").strip()
    if not group_id:
        group_id = "hysteresis_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_path = Path(os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_DATASET", os.fspath(DEFAULT_DATASET)))
    asset_manifest = Path(os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_ASSETS", os.fspath(DEFAULT_ASSET_MANIFEST)))
    judge_model = os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_JUDGE_MODEL", "gpt-5.4").strip() or "gpt-5.4"
    models = _parse_models(os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_MODELS"))
    skip_completed = _truthy(os.getenv("ONTOLOGYM_HYSTERESIS_ABLATION_SKIP_COMPLETED", "1"))

    _assert_inputs(dataset_path, asset_manifest, models)

    summary: dict[str, Any] = {
        "group_id": group_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "asset_manifest": str(asset_manifest),
        "judge_model": judge_model,
        "models": [],
        "steps": [],
    }

    summary_json = DATA_DIR / f"{group_id}_summary.json"
    summary_txt = DATA_DIR / f"{group_id}_summary.txt"

    for label, model in models:
        source_run = DEFAULT_SOURCE_RUNS[label]
        run_id = f"run_{group_id}_{label}"
        run_root = DATA_DIR / run_id
        print(f"\n=== {label} / {model} ===", flush=True)
        initialize_manifest(
            run_root,
            project_root=PROJECT_ROOT,
            paper_dir=source_run / "reused_paper_inputs",
            pipeline_order=[
                "ontogen_baseline_reused",
                "relation_augmentation_baseline_reused",
                "relation_augmentation_hysteresis_caption",
                "relation_augmentation_hysteresis_caption_image",
                "qa_evaluation_hysteresis_existing_kg",
                "qa_evaluation_hysteresis_caption_kg",
                "qa_evaluation_hysteresis_caption_image_kg",
            ],
        )
        _copy_baseline_artifacts(source_run, run_root)
        _copy_dataset(dataset_path, run_root)

        model_record: dict[str, Any] = {
            "label": label,
            "model": model,
            "source_run": str(source_run),
            "run_id": run_id,
            "run_root": str(run_root),
            "relation": {},
            "eval": {},
        }

        for variant, spec in RELATION_PHASES.items():
            step = _run_relation_variant(
                run_root=run_root,
                label=label,
                model=model,
                asset_manifest=asset_manifest,
                variant=variant,
                phase=spec["phase"],
                include_captions=spec["include_captions"],
                include_images=spec["include_images"],
                skip_completed=skip_completed,
            )
            summary["steps"].append(step.to_dict())
            model_record["relation"][variant] = _relation_stats(_relation_output_dir(run_root, spec["phase"]))
            _write_summary(summary, summary_json, summary_txt)

        for variant, phase in EVAL_VARIANTS.items():
            step = _run_eval_variant(
                run_root=run_root,
                model=model,
                judge_model=judge_model,
                dataset_path=dataset_path,
                variant=variant,
                phase=phase,
                skip_completed=skip_completed,
            )
            summary["steps"].append(step.to_dict())
            model_record["eval"][variant] = _eval_stats(run_root, phase)
            _write_summary(summary, summary_json, summary_txt)

        usage = estimate_usage_cost(run_root / "logs" / "openai_usage.jsonl")
        model_record["usage"] = usage
        summary["models"].append(model_record)
        _write_summary(summary, summary_json, summary_txt)

    _write_summary(summary, summary_json, summary_txt)
    print(f"\nWrote {summary_json}", flush=True)
    print(f"Wrote {summary_txt}", flush=True)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_models(value: str | None) -> list[tuple[str, str]]:
    if not value:
        return DEFAULT_MODELS
    model_map = dict(DEFAULT_MODELS)
    parsed: list[tuple[str, str]] = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        if ":" in key:
            label, model = key.split(":", 1)
            parsed.append((label.strip(), model.strip()))
        elif key in model_map:
            parsed.append((key, model_map[key]))
        else:
            parsed.append((key.replace(".", "").replace("-", "_"), key))
    return parsed or DEFAULT_MODELS


def _assert_inputs(dataset_path: Path, asset_manifest: Path, models: list[tuple[str, str]]) -> None:
    missing = []
    if not dataset_path.exists():
        missing.append(dataset_path)
    if not asset_manifest.exists():
        missing.append(asset_manifest)
    for label, _ in models:
        source_run = DEFAULT_SOURCE_RUNS.get(label)
        if source_run is None:
            raise ValueError(f"No source run configured for model label: {label}")
        for path in [
            source_run / "ontogen" / "processed_data",
            source_run / "ontogen" / "taxonomy" / "tree_0.pkl",
            source_run / "ontogen" / "termo",
            source_run / "relation_augmentation" / "outputs" / "relation_graph.json",
        ]:
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("Missing hysteresis ablation inputs:\n" + "\n".join(str(path) for path in missing))

    asset_count = _count_jsonl(asset_manifest)
    dataset_count = _count_jsonl(dataset_path)
    if asset_count < 1:
        raise ValueError(f"Asset manifest is empty: {asset_manifest}")
    if dataset_count != 80:
        raise ValueError(f"Expected 80 QA examples, found {dataset_count}: {dataset_path}")


def _copy_baseline_artifacts(source_run: Path, run_root: Path) -> None:
    mappings = [
        (source_run / "ontogen" / "processed_data", run_root / "ontogen" / "processed_data"),
        (source_run / "ontogen" / "enriched", run_root / "ontogen" / "enriched"),
        (source_run / "ontogen" / "taxonomy", run_root / "ontogen" / "taxonomy"),
        (source_run / "ontogen" / "termo", run_root / "ontogen" / "termo"),
        (source_run / "relation_augmentation", run_root / "relation_augmentation"),
    ]
    for source, target in mappings:
        _copy_dir_if_missing(source, target)
    record_pipeline_run(
        run_root,
        "ontogen_baseline_reused",
        status="reused",
        inputs={"source_run": str(source_run)},
        outputs={"ontogen_dir": str(run_root / "ontogen")},
    )
    record_pipeline_run(
        run_root,
        "relation_augmentation_baseline_reused",
        status="reused",
        inputs={"source_relation_dir": str(source_run / "relation_augmentation")},
        outputs={"relation_dir": str(run_root / "relation_augmentation")},
    )


def _copy_dataset(dataset_path: Path, run_root: Path) -> None:
    target = run_root / "qa_extractor" / "qa_dataset.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(dataset_path, target)
    record_pipeline_run(
        run_root,
        "qa_dataset_reused",
        status="reused",
        inputs={"dataset_path": str(dataset_path)},
        outputs={"dataset_path": str(target)},
        extra={"examples": _count_jsonl(target)},
    )


def _copy_dir_if_missing(source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.exists():
        raise FileNotFoundError(f"Cannot copy missing source directory: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _run_relation_variant(
    *,
    run_root: Path,
    label: str,
    model: str,
    asset_manifest: Path,
    variant: str,
    phase: str,
    include_captions: str,
    include_images: str,
    skip_completed: bool,
) -> StepRecord:
    output_dir = _relation_output_dir(run_root, phase)
    expected = output_dir / "relation_graph.json"
    name = f"{label}:{variant}:relation"
    if skip_completed and expected.exists():
        return StepRecord(
            name=name,
            status="skipped",
            run_id=run_root.name,
            output=str(expected),
            detail=_relation_stats(output_dir),
        )

    env = _base_env(run_root, model)
    env.update(
        {
            "ONTOLOGYM_RELATION_PHASE": phase,
            "ONTOLOGYM_RELATION_INCLUDE_BASE_TEXT_CHUNKS": "0",
            "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTIONS": include_captions,
            "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_IMAGES": include_images,
            "ONTOLOGYM_RELATION_INCLUDE_HYSTERESIS_CAPTION_CHUNKS_WITH_IMAGES": "0",
            "ONTOLOGYM_RELATION_HYSTERESIS_ASSET_MANIFEST_JSONL": str(asset_manifest),
            "ONTOLOGYM_RELATION_ENABLE_HYSTERESIS_SCHEMA_EXTENSIONS": "1",
            "ONTOLOGYM_RELATION_REQUIRE_EVIDENCE_QUOTE_FOR_IMAGE_CHUNKS": "0",
            "ONTOLOGYM_RELATION_MAX_CANDIDATE_TERMS": "120",
            "ONTOLOGYM_RELATION_MAX_CLAIMS_PER_CHUNK": "12",
        }
    )
    return _run_command(name, run_root, "run_relation_augmentation.py", env, expected)


def _run_eval_variant(
    *,
    run_root: Path,
    model: str,
    judge_model: str,
    dataset_path: Path,
    variant: str,
    phase: str,
    skip_completed: bool,
) -> StepRecord:
    expected = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
    name = f"{run_root.name}:{variant}:eval"
    if skip_completed and expected.exists():
        return StepRecord(
            name=name,
            status="skipped",
            run_id=run_root.name,
            output=str(expected),
            detail=_eval_stats(run_root, phase),
        )

    kg_dirs = _kg_dirs_for_variant(run_root, variant)
    env = _base_env(run_root, model)
    env.update(
        {
            "ONTOLOGYM_QA_EVAL_PHASE": phase,
            "ONTOLOGYM_QA_EVAL_DATASET_PATH": str(dataset_path),
            "ONTOLOGYM_QA_EVAL_EXTRA_KG_DIRS": os.pathsep.join(str(path) for path in kg_dirs),
            "ONTOLOGYM_QA_EVAL_DRY_RUN": "0",
            "ONTOLOGYM_QA_EVAL_MODEL": model,
            "ONTOLOGYM_QA_EVAL_RUN_AIRQA_EVALUATOR": "1",
            "ONTOLOGYM_QA_EVAL_ALLOW_LLM_EVALUATORS": "1",
            "ONTOLOGYM_QA_EVAL_AIRQA_EVALUATOR_MODEL": judge_model,
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS": "0",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS": "1",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-small",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_BATCH_SIZE": "32",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_BM25_WEIGHT": "0.4",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT": "0.6",
            "ONTOLOGYM_QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY": "node_bundle_iterative",
            "ONTOLOGYM_QA_EVAL_REEVALUATE_EXISTING_SCORES": "0",
        }
    )
    return _run_command(name, run_root, "run_qa_evaluation.py", env, expected)


def _base_env(run_root: Path, model: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ONTOLOGYM_RUN_ID": run_root.name,
            "ONTOLOGYM_USAGE_LOG": str(run_root / "logs" / "openai_usage.jsonl"),
            "ONTOLOGYM_MODEL": model,
            "ONTOLOGYM_RELATION_MODEL": model,
            "ONTOLOGYM_OPENAI_TIMEOUT_SECONDS": os.getenv("ONTOLOGYM_OPENAI_TIMEOUT_SECONDS", "90"),
        }
    )
    return env


def _run_command(
    name: str,
    run_root: Path,
    script: str,
    env: dict[str, str],
    expected_output: Path,
) -> StepRecord:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    command = [PYTHON, script]
    print(f"  - {name}: {' '.join(command)}", flush=True)
    start = time.monotonic()
    proc = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
    seconds = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")
    if not expected_output.exists():
        raise FileNotFoundError(f"{name} finished but missing expected output: {expected_output}")
    detail = _relation_stats(expected_output.parent) if "relation" in name else _eval_stats(run_root, expected_output.parents[1].name.removeprefix("qa_evaluation_"))
    return StepRecord(
        name=name,
        status="completed",
        run_id=run_root.name,
        output=str(expected_output),
        seconds=seconds,
        detail=detail,
    )


def _relation_output_dir(run_root: Path, phase: str) -> Path:
    return run_root / f"relation_augmentation_{phase}" / "outputs"


def _kg_dirs_for_variant(run_root: Path, variant: str) -> list[Path]:
    dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
        run_root / "relation_augmentation" / "outputs",
    ]
    if variant == "caption_kg":
        dirs.append(_relation_output_dir(run_root, RELATION_PHASES["caption_kg"]["phase"]))
    elif variant == "caption_image_kg":
        dirs.append(_relation_output_dir(run_root, RELATION_PHASES["caption_image_kg"]["phase"]))
    return dirs


def _relation_stats(output_dir: Path) -> dict[str, Any]:
    summary = _read_json(output_dir / "run_summary.json")
    graph = _read_json(output_dir / "relation_graph.json")
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    source_types = Counter(
        edge.get("provenance", {}).get("source_type")
        for edge in edges
        if isinstance(edge, dict)
    )
    visual_edges = sum(
        1
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("provenance", {}).get("source_type") == "hysteresis_figure_image"
    )
    visible_quote_edges = sum(
        1
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("evidence_quote") == "VISIBLE_IN_FIGURE"
    )
    visual_qualifier_edges = sum(
        1
        for edge in edges
        if isinstance(edge, dict)
        and isinstance(edge.get("qualifiers"), dict)
        and bool(edge["qualifiers"].get("visual_evidence"))
    )
    return {
        "status": summary.get("status"),
        "evidence_chunks": summary.get("evidence_chunks"),
        "base_text_evidence_enabled": summary.get("base_text_evidence_enabled"),
        "base_text_evidence_chunks": summary.get("base_text_evidence_chunks"),
        "hysteresis_evidence_chunks": summary.get("hysteresis_evidence_chunks"),
        "accepted_claims": summary.get("accepted_claims"),
        "rejected_claims": summary.get("rejected_claims"),
        "nodes": len(nodes),
        "edges": len(edges),
        "source_types": dict(source_types),
        "visual_edges": visual_edges,
        "visible_in_figure_quote_edges": visible_quote_edges,
        "visual_evidence_qualifier_edges": visual_qualifier_edges,
    }


def _eval_stats(run_root: Path, phase: str) -> dict[str, Any]:
    path = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
    data = _read_json(path)
    records = data.get("records", []) if isinstance(data, dict) else []
    uuid_tags = _dataset_tags(Path(data.get("dataset_path", ""))) if data.get("dataset_path") else {}
    split = {
        "base40": _score_split(records, "hysteresis_ablation_base40", uuid_tags),
        "image40": _score_split(records, "hysteresis_ablation_image40", uuid_tags),
    }
    insufficient = sum(
        1
        for record in records
        if str(record.get("answer", "")).strip() == "[INSUFFICIENT_CONTEXT]"
    )
    failed_image = []
    for record in records:
        score = (record.get("airqa_score") or {}).get("score") if isinstance(record.get("airqa_score"), dict) else None
        if score is not None and float(score) < 0.5 and _has_tag(record, "hysteresis_ablation_image40", uuid_tags):
            failed_image.append(
                {
                    "uuid": record.get("uuid"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "score": score,
                    "retrieved_edges": len(record.get("retrieved_edges", [])),
                }
            )
    return {
        "results_json": str(path),
        "answers_jsonl": data.get("answers_jsonl"),
        "average_score": data.get("average_score"),
        "evaluated_examples": data.get("evaluated_examples"),
        "scored_examples": data.get("scored_examples"),
        "skipped_evaluators": data.get("skipped_evaluators"),
        "kg_facts": data.get("kg_facts"),
        "kg_communities": data.get("kg_communities"),
        "kg_retrieval_mode": data.get("kg_retrieval_mode"),
        "kg_retrieval_strategy": data.get("kg_retrieval_strategy"),
        "insufficient_context_answers": insufficient,
        "split_scores": split,
        "failed_image_examples": failed_image[:12],
    }


def _score_split(records: list[dict[str, Any]], tag: str, uuid_tags: dict[str, set[str]]) -> dict[str, Any]:
    scored = []
    total = 0
    insufficient = 0
    for record in records:
        if not _has_tag(record, tag, uuid_tags):
            continue
        total += 1
        if str(record.get("answer", "")).strip() == "[INSUFFICIENT_CONTEXT]":
            insufficient += 1
        score_info = record.get("airqa_score")
        if isinstance(score_info, dict) and score_info.get("score") is not None:
            scored.append(float(score_info["score"]))
    return {
        "examples": total,
        "scored_examples": len(scored),
        "average_score": sum(scored) / len(scored) if scored else None,
        "insufficient_context_answers": insufficient,
    }


def _has_tag(record: dict[str, Any], tag: str, uuid_tags: dict[str, set[str]]) -> bool:
    tags = record.get("tags")
    if not tags:
        tags = uuid_tags.get(str(record.get("uuid") or ""), set())
    return tag in set(str(item) for item in tags)


def _dataset_tags(dataset_path: Path) -> dict[str, set[str]]:
    tags_by_uuid: dict[str, set[str]] = {}
    if not dataset_path.exists():
        return tags_by_uuid
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError:
            continue
        uuid = str(example.get("uuid") or "")
        if uuid:
            tags_by_uuid[uuid] = {str(tag) for tag in example.get("tags", [])}
    return tags_by_uuid


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_summary(summary: dict[str, Any], summary_json: Path, summary_txt: Path) -> None:
    summary["total_estimated_cost_usd"] = sum(
        float(model.get("usage", {}).get("estimated_cost_usd", 0.0))
        for model in summary.get("models", [])
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_txt.write_text(_format_summary(summary), encoding="utf-8")


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Hysteresis KG Ablation Summary",
        f"group_id: {summary.get('group_id')}",
        f"dataset: {summary.get('dataset_path')}",
        f"asset_manifest: {summary.get('asset_manifest')}",
        f"judge_model: {summary.get('judge_model')}",
        f"estimated_cost_usd: ${float(summary.get('total_estimated_cost_usd') or 0.0):.4f}",
        "",
        "Results",
    ]
    for model in summary.get("models", []):
        lines.append(f"{model['label']} ({model['model']}):")
        for variant in EVAL_VARIANTS:
            stats = model.get("eval", {}).get(variant, {})
            split = stats.get("split_scores", {})
            base = split.get("base40", {})
            image = split.get("image40", {})
            lines.append(
                "  "
                f"{variant}: score={_score_text(stats.get('average_score'))}, "
                f"base40={_score_text(base.get('average_score'))}, "
                f"image40={_score_text(image.get('average_score'))}, "
                f"examples={stats.get('evaluated_examples')}, "
                f"kg_facts={stats.get('kg_facts')}, "
                f"insufficient={stats.get('insufficient_context_answers')}"
            )
        usage = model.get("usage", {})
        lines.append(f"  estimated_cost_usd=${float(usage.get('estimated_cost_usd') or 0.0):.4f}")
    lines.append("")
    lines.append("Step records and failure samples are in the JSON summary.")
    return "\n".join(lines) + "\n"


def _score_text(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
