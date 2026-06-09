"""Run hysteresis figure KG ablation from the taxonomy/TERMO baseline only.

This runner intentionally excludes the pre-existing relation augmentation KG.
Each eval variant starts from OntoGen taxonomy + TERMO outputs, then optionally
adds hysteresis figure caption, image-only, or caption+image relation KGs.
"""

from __future__ import annotations

import json
import os
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
from run_hysteresis_ablation_experiments import (
    DATA_DIR,
    DEFAULT_ASSET_MANIFEST,
    DEFAULT_MODELS,
    DEFAULT_SOURCE_RUNS,
    PROJECT_ROOT,
    PYTHON,
    StepRecord,
    _base_env,
    _copy_dataset,
    _copy_dir_if_missing,
    _count_jsonl,
    _dataset_tags,
    _eval_stats,
    _parse_models,
    _read_json,
    _run_command,
    _score_text,
    _truthy,
)


DEFAULT_DATASET = (
    DATA_DIR
    / "hysteresis_ablation_manual_20260604"
    / "qa"
    / "qa_dataset_hysteresis_academic_augmented_80.jsonl"
)

RELATION_PHASES = {
    "caption_kg": {
        "phase": "hysteresis_caption",
        "include_captions": "1",
        "include_images": "0",
        "image_include_caption_context": "1",
    },
    "image_kg": {
        "phase": "hysteresis_image",
        "include_captions": "0",
        "include_images": "1",
        "image_include_caption_context": "0",
    },
    "caption_image_kg": {
        "phase": "hysteresis_caption_image",
        "include_captions": "1",
        "include_images": "1",
        "image_include_caption_context": "1",
    },
}

EVAL_VARIANTS = {
    "taxonomy_only": "hysteresis_taxonomy_only_kg",
    "caption_kg": "hysteresis_taxonomy_caption_kg",
    "image_kg": "hysteresis_taxonomy_image_kg",
    "caption_image_kg": "hysteresis_taxonomy_caption_image_kg",
}


def main() -> None:
    group_id = os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_GROUP_ID", "").strip()
    if not group_id:
        group_id = "hysteresis_taxonomy_figure_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_path = Path(
        os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_DATASET", os.fspath(DEFAULT_DATASET))
    )
    asset_manifest = Path(
        os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_ASSETS", os.fspath(DEFAULT_ASSET_MANIFEST))
    )
    judge_model = (
        os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_JUDGE_MODEL", "gpt-5.4").strip()
        or "gpt-5.4"
    )
    models = _parse_models(os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_MODELS"))
    skip_completed = _truthy(os.getenv("ONTOLOGYM_HYSTERESIS_TAXONOMY_ABLATION_SKIP_COMPLETED", "1"))

    _assert_inputs(dataset_path, asset_manifest, models)

    summary: dict[str, Any] = {
        "group_id": group_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "ontogen_taxonomy_plus_termo_only",
        "excluded_from_eval_kg": ["relation_augmentation/outputs"],
        "dataset_path": str(dataset_path),
        "asset_manifest": str(asset_manifest),
        "judge_model": judge_model,
        "answer_prompt_mode": "strict_context_grounded",
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
                "ontogen_taxonomy_baseline_reused",
                "relation_augmentation_baseline_excluded",
                "relation_augmentation_hysteresis_caption",
                "relation_augmentation_hysteresis_image",
                "relation_augmentation_hysteresis_caption_image",
                "qa_evaluation_hysteresis_taxonomy_only_kg",
                "qa_evaluation_hysteresis_taxonomy_caption_kg",
                "qa_evaluation_hysteresis_taxonomy_image_kg",
                "qa_evaluation_hysteresis_taxonomy_caption_image_kg",
            ],
        )
        _copy_taxonomy_artifacts(source_run, run_root)
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
                image_include_caption_context=spec["image_include_caption_context"],
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
        ]:
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing taxonomy figure ablation inputs:\n" + "\n".join(str(path) for path in missing)
        )

    asset_count = _count_jsonl(asset_manifest)
    dataset_count = _count_jsonl(dataset_path)
    if asset_count < 1:
        raise ValueError(f"Asset manifest is empty: {asset_manifest}")
    if dataset_count != 80:
        raise ValueError(f"Expected 80 QA examples, found {dataset_count}: {dataset_path}")


def _copy_taxonomy_artifacts(source_run: Path, run_root: Path) -> None:
    mappings = [
        (source_run / "ontogen" / "processed_data", run_root / "ontogen" / "processed_data"),
        (source_run / "ontogen" / "enriched", run_root / "ontogen" / "enriched"),
        (source_run / "ontogen" / "taxonomy", run_root / "ontogen" / "taxonomy"),
        (source_run / "ontogen" / "termo", run_root / "ontogen" / "termo"),
    ]
    for source, target in mappings:
        _copy_dir_if_missing(source, target)
    record_pipeline_run(
        run_root,
        "ontogen_taxonomy_baseline_reused",
        status="reused",
        inputs={"source_run": str(source_run)},
        outputs={"ontogen_dir": str(run_root / "ontogen")},
        extra={"eval_kg_dirs": ["ontogen/taxonomy", "ontogen/termo"]},
    )
    record_pipeline_run(
        run_root,
        "relation_augmentation_baseline_excluded",
        status="excluded",
        inputs={"source_relation_dir": str(source_run / "relation_augmentation")},
        outputs={},
        extra={"reason": "taxonomy figure ablation starts from taxonomy/TERMO only"},
    )


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
    image_include_caption_context: str,
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
            "ONTOLOGYM_RELATION_HYSTERESIS_IMAGE_INCLUDE_CAPTION_CONTEXT": image_include_caption_context,
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
            "ONTOLOGYM_QA_EVAL_STRICT_CONTEXT_GROUNDING": "1",
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


def _relation_output_dir(run_root: Path, phase: str) -> Path:
    return run_root / f"relation_augmentation_{phase}" / "outputs"


def _kg_dirs_for_variant(run_root: Path, variant: str) -> list[Path]:
    dirs = [
        run_root / "ontogen" / "taxonomy",
        run_root / "ontogen" / "termo",
    ]
    if variant == "caption_kg":
        dirs.append(_relation_output_dir(run_root, RELATION_PHASES["caption_kg"]["phase"]))
    elif variant == "image_kg":
        dirs.append(_relation_output_dir(run_root, RELATION_PHASES["image_kg"]["phase"]))
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
    caption_edges = sum(
        1
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("provenance", {}).get("source_type") == "hysteresis_figure_caption"
    )
    visible_quote_edges = sum(
        1
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("evidence_quote") == "VISIBLE_IN_FIGURE"
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
        "caption_edges": caption_edges,
        "visual_edges": visual_edges,
        "visible_in_figure_quote_edges": visible_quote_edges,
    }


def _write_summary(summary: dict[str, Any], summary_json: Path, summary_txt: Path) -> None:
    summary["total_estimated_cost_usd"] = sum(
        float(model.get("usage", {}).get("estimated_cost_usd", 0.0))
        for model in summary.get("models", [])
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_txt.write_text(_format_summary(summary), encoding="utf-8")


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Hysteresis Taxonomy Figure KG Ablation Summary",
        f"group_id: {summary.get('group_id')}",
        f"baseline: {summary.get('baseline')}",
        f"excluded_from_eval_kg: {', '.join(summary.get('excluded_from_eval_kg') or [])}",
        f"dataset: {summary.get('dataset_path')}",
        f"asset_manifest: {summary.get('asset_manifest')}",
        f"judge_model: {summary.get('judge_model')}",
        f"answer_prompt_mode: {summary.get('answer_prompt_mode')}",
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
    lines.append("Relation KG stats, step records, and failure samples are in the JSON summary.")
    return "\n".join(lines) + "\n"


def write_markdown_report(summary_path: Path) -> Path:
    summary = _read_json(summary_path)
    report_path = summary_path.with_name(summary_path.stem.replace("_summary", "_report") + ".md")
    lines = [
        "# Hysteresis Taxonomy Figure KG Ablation",
        "",
        f"- group_id: `{summary.get('group_id')}`",
        f"- baseline: `{summary.get('baseline')}`",
        f"- excluded KG: `{', '.join(summary.get('excluded_from_eval_kg') or [])}`",
        f"- dataset: `{summary.get('dataset_path')}`",
        f"- asset_manifest: `{summary.get('asset_manifest')}`",
        f"- judge_model: `{summary.get('judge_model')}`",
        f"- answer_prompt_mode: `{summary.get('answer_prompt_mode')}`",
        "",
        "## Results",
        "",
        "| model | variant | overall | base40 | image40 | kg_facts | insufficient |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in summary.get("models", []):
        for variant in EVAL_VARIANTS:
            stats = model.get("eval", {}).get(variant, {})
            split = stats.get("split_scores", {})
            base = split.get("base40", {})
            image = split.get("image40", {})
            lines.append(
                "| "
                f"{model.get('label')} | {variant} | {_score_text(stats.get('average_score'))} | "
                f"{_score_text(base.get('average_score'))} | {_score_text(image.get('average_score'))} | "
                f"{stats.get('kg_facts')} | {stats.get('insufficient_context_answers')} |"
            )

    lines.extend(
        [
            "",
            "## Relation KG Stats",
            "",
            "| model | relation_variant | chunks | edges | caption_edges | visual_edges | visible_in_figure_edges |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in summary.get("models", []):
        for variant in RELATION_PHASES:
            stats = model.get("relation", {}).get(variant, {})
            lines.append(
                "| "
                f"{model.get('label')} | {variant} | {stats.get('hysteresis_evidence_chunks')} | "
                f"{stats.get('edges')} | {stats.get('caption_edges')} | {stats.get('visual_edges')} | "
                f"{stats.get('visible_in_figure_quote_edges')} |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def retrieval_source_audit(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    audit: dict[str, Any] = {}
    for model in summary.get("models", []):
        label = str(model.get("label"))
        run_root = Path(model.get("run_root", ""))
        audit[label] = {}
        for variant, phase in EVAL_VARIANTS.items():
            results_path = run_root / f"qa_evaluation_{phase}" / "outputs" / "evaluation_results.json"
            data = _read_json(results_path)
            records = data.get("records", []) if isinstance(data, dict) else []
            uuid_tags = _dataset_tags(Path(data.get("dataset_path", ""))) if data.get("dataset_path") else {}
            image_records = [
                record
                for record in records
                if _record_has_tag(record, "hysteresis_ablation_image40", uuid_tags)
            ]
            source_counter: Counter[str] = Counter()
            records_with_caption_edges = 0
            records_with_image_edges = 0
            for record in image_records:
                record_source_types = set()
                for edge in record.get("retrieved_edges", []):
                    if not isinstance(edge, dict):
                        continue
                    source_type = str(edge.get("metadata", {}).get("provenance", {}).get("source_type") or "")
                    if source_type:
                        source_counter[source_type] += 1
                        record_source_types.add(source_type)
                if "hysteresis_figure_caption" in record_source_types:
                    records_with_caption_edges += 1
                if "hysteresis_figure_image" in record_source_types:
                    records_with_image_edges += 1
            audit[label][variant] = {
                "image_records": len(image_records),
                "records_with_caption_edges": records_with_caption_edges,
                "records_with_image_edges": records_with_image_edges,
                "retrieved_source_types": dict(source_counter),
            }
    return audit


def _record_has_tag(record: dict[str, Any], tag: str, uuid_tags: dict[str, set[str]]) -> bool:
    tags = record.get("tags") or uuid_tags.get(str(record.get("uuid") or ""), set())
    return tag in {str(item) for item in tags}


if __name__ == "__main__":
    main()
