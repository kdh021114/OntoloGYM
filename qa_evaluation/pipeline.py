from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from common.project_config import load_env_file, load_project_config
from common.run_context import record_pipeline_run

from .dataset import load_airqa_dataset, load_baseline_answers
from .graphrag import (
    format_graphrag_context,
    load_graphrag_index,
    node_bundles_to_context,
    retrieve_graphrag_context,
    retrieve_graphrag_node_bundles,
)
from .llm import (
    OpenAIEmbeddingClient,
    OpenAITextClient,
    build_answer_prompt,
    build_pairwise_judge_prompt,
    json_dumps,
    parse_pairwise_winner,
)


logger = logging.getLogger(__name__)
_AIRQA_EVALUATE_FUNC = None


def _pipeline_name(config) -> str:
    phase = str(getattr(config, "QA_EVAL_PHASE", "") or "").strip()
    return "qa_evaluation" if not phase else f"qa_evaluation_{phase}"


def run_pipeline() -> dict[str, Any]:
    config = load_project_config()
    load_env_file(getattr(config, "ENV_FILE"))

    if not getattr(config, "QA_EVAL_RUN", False):
        logger.info("QA evaluation is disabled. Set QA_EVAL_RUN=True to run it.")
        result = {"status": "disabled"}
        record_pipeline_run(config.RUN_OUTPUT_DIR, _pipeline_name(config), status="disabled")
        return result

    output_dir = Path(config.QA_EVAL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.QA_EVAL_KG_INPUT_DIR).mkdir(parents=True, exist_ok=True)

    examples = load_airqa_dataset(Path(config.QA_EVAL_DATASET_PATH), config.QA_EVAL_MAX_EXAMPLES)
    kg_dirs = _resolve_kg_dirs(config)
    report_client = None
    if getattr(config, "QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS", True):
        report_client = OpenAITextClient(
            model=config.QA_EVAL_GRAPHRAG_REPORT_MODEL,
            backend=config.QA_EVAL_BACKEND,
            temperature=config.QA_EVAL_GRAPHRAG_REPORT_TEMPERATURE,
            max_completion_tokens=config.QA_EVAL_GRAPHRAG_REPORT_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.QA_EVAL_REASONING_EFFORT,
            usage_component="qa_evaluation_graphrag_reports",
        )
    embedding_client = None
    if getattr(config, "QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS", True):
        embedding_client = OpenAIEmbeddingClient(
            model=config.QA_EVAL_GRAPHRAG_EMBEDDING_MODEL,
            backend=config.QA_EVAL_BACKEND,
            dimensions=config.QA_EVAL_GRAPHRAG_EMBEDDING_DIMENSIONS,
            usage_component="qa_evaluation_graphrag_embeddings",
        )
    graphrag_index = load_graphrag_index(
        kg_dirs=kg_dirs,
        ontogen_root=Path(config.PROJECT_ROOT) / "ontogen",
        cache_dir=Path(config.QA_EVAL_GRAPHRAG_CACHE_DIR),
        report_client=report_client,
        max_communities=config.QA_EVAL_GRAPHRAG_MAX_COMMUNITIES,
        max_edges_per_report=config.QA_EVAL_GRAPHRAG_MAX_EDGES_PER_COMMUNITY_REPORT,
        max_nodes_per_community=config.QA_EVAL_GRAPHRAG_MAX_NODES_PER_COMMUNITY,
        enable_llm_reports=getattr(config, "QA_EVAL_GRAPHRAG_ENABLE_LLM_COMMUNITY_REPORTS", True),
        embedding_client=embedding_client,
        enable_embeddings=getattr(config, "QA_EVAL_GRAPHRAG_ENABLE_EMBEDDINGS", True),
        embedding_dimensions=config.QA_EVAL_GRAPHRAG_EMBEDDING_DIMENSIONS,
        embedding_batch_size=config.QA_EVAL_GRAPHRAG_EMBEDDING_BATCH_SIZE,
        bm25_weight=config.QA_EVAL_GRAPHRAG_BM25_WEIGHT,
        embedding_weight=config.QA_EVAL_GRAPHRAG_EMBEDDING_WEIGHT,
    )
    _write_json(
        Path(config.QA_EVAL_KG_CONTEXT_JSON),
        {
            "mode": "graphrag",
            "graph_hash": graphrag_index.graph_hash,
            "edges": len(graphrag_index.edges),
            "retrieval": {
                "mode": "hybrid" if graphrag_index.embedding_index else "bm25",
                "strategy": config.QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY,
                "bm25_weight": graphrag_index.bm25_weight,
                "embedding_weight": graphrag_index.embedding_weight,
                "embedding_model": getattr(graphrag_index.embedding_index, "model", None),
                "embedding_dimensions": getattr(graphrag_index.embedding_index, "dimensions", None),
                "taxonomy_edge_fraction": config.QA_EVAL_GRAPHRAG_TAXONOMY_EDGE_FRACTION,
                "taxonomy_community_limit": config.QA_EVAL_GRAPHRAG_TAXONOMY_COMMUNITY_LIMIT,
            },
            "communities": [
                community.to_dict(include_edges=False)
                for community in graphrag_index.communities
            ],
        },
    )

    summary = {
        "status": "dry_run" if getattr(config, "QA_EVAL_DRY_RUN", True) else "completed",
        "dataset_path": str(config.QA_EVAL_DATASET_PATH),
        "examples": len(examples),
        "kg_dirs": [str(path) for path in kg_dirs],
        "kg_facts": len(graphrag_index.edges),
        "kg_communities": len(graphrag_index.communities),
        "kg_context_mode": "graphrag",
        "kg_retrieval_mode": "hybrid" if graphrag_index.embedding_index else "bm25",
        "kg_context_json": str(config.QA_EVAL_KG_CONTEXT_JSON),
        "kg_retrieval_strategy": config.QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY,
    }

    if getattr(config, "QA_EVAL_DRY_RUN", True):
        summary_path = output_dir / "dry_run_summary.json"
        _write_json(summary_path, summary)
        record_pipeline_run(
            config.RUN_OUTPUT_DIR,
            _pipeline_name(config),
            status="dry_run",
            inputs={
                "dataset_path": str(config.QA_EVAL_DATASET_PATH),
                "kg_dirs": [str(path) for path in kg_dirs],
            },
            outputs={
                "kg_context_json": str(config.QA_EVAL_KG_CONTEXT_JSON),
                "summary": str(summary_path),
            },
            extra=summary,
        )
        logger.info("QA evaluation dry-run summary written to %s.", summary_path)
        return summary

    answer_client = OpenAITextClient(
        model=config.QA_EVAL_MODEL,
        backend=config.QA_EVAL_BACKEND,
        temperature=config.QA_EVAL_TEMPERATURE,
        max_completion_tokens=config.QA_EVAL_MAX_COMPLETION_TOKENS,
        reasoning_effort=config.QA_EVAL_REASONING_EFFORT,
    )
    judge_client = None
    if getattr(config, "QA_EVAL_RUN_PAIRWISE_JUDGE", False):
        judge_client = OpenAITextClient(
            model=config.QA_EVAL_JUDGE_MODEL,
            backend=config.QA_EVAL_BACKEND,
            temperature=0.0,
            max_completion_tokens=config.QA_EVAL_MAX_COMPLETION_TOKENS,
            reasoning_effort=config.QA_EVAL_REASONING_EFFORT,
        )

    baseline_answers = load_baseline_answers(config.QA_EVAL_BASELINE_ANSWERS_PATH)
    answers_path = Path(config.QA_EVAL_ANSWERS_JSONL)
    records_by_uuid = _load_existing_answer_records(answers_path)
    answer_cache_key = _answer_cache_key(config, graphrag_index)
    for example in examples:
        uuid = example.get("uuid")
        if uuid in records_by_uuid:
            if getattr(config, "QA_EVAL_REEVALUATE_EXISTING_SCORES", False):
                record = records_by_uuid[uuid]
                record["airqa_score"] = _evaluate_answer(record.get("answer", ""), example, config)
                records_by_uuid[uuid] = record
            elif records_by_uuid[uuid].get("answer_cache_key") == answer_cache_key:
                continue
            else:
                del records_by_uuid[uuid]
        if uuid in records_by_uuid:
            continue
        answer, retrieved, retrieval_metadata = _answer_with_retrieval(
            example=example,
            config=config,
            graphrag_index=graphrag_index,
            answer_client=answer_client,
        )
        record = {
            "uuid": example.get("uuid"),
            "question": example.get("question"),
            "answer": answer,
            "retrieved_communities": [
                community.to_dict(include_edges=False)
                for community in retrieved.communities
            ],
            "retrieved_edges": [edge.to_dict() for edge in retrieved.edges],
            "retrieved_facts": [edge.to_dict() for edge in retrieved.edges],
            "retrieved_node_bundles": [
                bundle.to_dict() for bundle in getattr(retrieved, "node_bundles", [])
            ],
            "retrieval_metadata": retrieval_metadata,
            "airqa_score": _evaluate_answer(answer, example, config),
            "answer_cache_key": answer_cache_key,
        }
        if judge_client is not None and example.get("uuid") in baseline_answers:
            judge_prompt = build_pairwise_judge_prompt(
                question=example.get("question", ""),
                answer1=baseline_answers[example["uuid"]],
                answer2=answer,
                criterion=config.QA_EVAL_JUDGE_CRITERION,
            )
            judge_answer = judge_client.complete(judge_prompt)
            record["pairwise_judge"] = {
                "answer1": "baseline",
                "answer2": "kg",
                "winner": parse_pairwise_winner(judge_answer),
                "raw": judge_answer,
            }
        records_by_uuid[uuid] = record
        _write_jsonl(answers_path, _ordered_records(examples, records_by_uuid))

    records = _ordered_records(examples, records_by_uuid)
    _write_jsonl(answers_path, records)
    results = _summarize_results(records, summary, answers_path)
    _write_json(Path(config.QA_EVAL_RESULTS_JSON), results)
    record_pipeline_run(
        config.RUN_OUTPUT_DIR,
        _pipeline_name(config),
        status="completed",
        inputs={
            "dataset_path": str(config.QA_EVAL_DATASET_PATH),
            "kg_dirs": [str(path) for path in kg_dirs],
        },
        outputs={
            "answers_jsonl": str(config.QA_EVAL_ANSWERS_JSONL),
            "results_json": str(config.QA_EVAL_RESULTS_JSON),
            "kg_context_json": str(config.QA_EVAL_KG_CONTEXT_JSON),
        },
        extra={
            "examples": len(examples),
            "kg_facts": len(graphrag_index.edges),
            "kg_communities": len(graphrag_index.communities),
            "kg_context_mode": "graphrag",
            "evaluated_examples": len(records),
            "average_score": results.get("average_score"),
        },
    )
    logger.info("QA evaluation finished for %s examples.", len(records))
    return results


def _resolve_kg_dirs(config) -> list[Path]:
    dirs = [Path(config.QA_EVAL_KG_INPUT_DIR)]
    if getattr(config, "QA_EVAL_AUTO_INCLUDE_ONTOGEN_OUTPUTS", True):
        dirs.extend(Path(path) for path in config.QA_EVAL_EXTRA_KG_DIRS)
    return dirs


def _answer_with_retrieval(
    *,
    example: dict[str, Any],
    config,
    graphrag_index,
    answer_client: OpenAITextClient,
):
    strategy = str(getattr(config, "QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY", "edge_topk") or "edge_topk")
    if strategy == "node_bundle_iterative":
        return _answer_with_node_bundle_retrieval(
            example=example,
            config=config,
            graphrag_index=graphrag_index,
            answer_client=answer_client,
        )

    retrieved = retrieve_graphrag_context(
        question=example.get("question", ""),
        index=graphrag_index,
        top_communities=config.QA_EVAL_GRAPHRAG_TOP_COMMUNITIES,
        top_edges=config.QA_EVAL_GRAPHRAG_TOP_EDGES,
        max_context_chars=config.QA_EVAL_MAX_CONTEXT_CHARS,
        taxonomy_edge_fraction=config.QA_EVAL_GRAPHRAG_TAXONOMY_EDGE_FRACTION,
        taxonomy_community_limit=config.QA_EVAL_GRAPHRAG_TAXONOMY_COMMUNITY_LIMIT,
    )
    answer = _answer_from_context(example, config, answer_client, retrieved)
    return answer, retrieved, {"retrieval_strategy": "edge_topk", "candidate_count": len(retrieved.edges)}


def _answer_with_node_bundle_retrieval(
    *,
    example: dict[str, Any],
    config,
    graphrag_index,
    answer_client: OpenAITextClient,
):
    max_candidates = max(1, int(getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_MAX_CANDIDATES", 6)))
    initial_candidates = max(1, int(getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_INITIAL_CANDIDATES", 2)))
    batch_size = max(1, int(getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_BATCH_SIZE", 2)))
    bundles = retrieve_graphrag_node_bundles(
        question=example.get("question", ""),
        index=graphrag_index,
        max_candidates=max_candidates,
        max_edges_per_bundle=getattr(config, "QA_EVAL_GRAPHRAG_MAX_EDGES_PER_NODE_BUNDLE", 32),
        taxonomy_ancestor_depth=int(getattr(config, "QA_EVAL_GRAPHRAG_TAXONOMY_ANCESTOR_DEPTH", 3)),
    )
    if not bundles:
        retrieved = node_bundles_to_context([], max_context_chars=config.QA_EVAL_MAX_CONTEXT_CHARS)
        answer = _answer_from_context(example, config, answer_client, retrieved)
        return answer, retrieved, {
            "retrieval_strategy": "node_bundle_iterative",
            "candidate_count": 0,
            "rounds": 1,
        }

    candidates_to_show = min(initial_candidates, len(bundles), max_candidates)
    rounds = 0
    last_answer = "[INSUFFICIENT_CONTEXT]"
    last_retrieved = node_bundles_to_context([], max_context_chars=config.QA_EVAL_MAX_CONTEXT_CHARS)
    while candidates_to_show <= min(len(bundles), max_candidates):
        rounds += 1
        last_retrieved = node_bundles_to_context(
            bundles[:candidates_to_show],
            max_context_chars=config.QA_EVAL_MAX_CONTEXT_CHARS,
        )
        last_answer = _answer_from_context(example, config, answer_client, last_retrieved)
        if not _is_insufficient_answer(last_answer):
            break
        if candidates_to_show >= min(len(bundles), max_candidates):
            break
        candidates_to_show = min(candidates_to_show + batch_size, len(bundles), max_candidates)

    return last_answer, last_retrieved, {
        "retrieval_strategy": "node_bundle_iterative",
        "candidate_count": len(bundles),
        "candidates_shown": len(getattr(last_retrieved, "node_bundles", [])),
        "rounds": rounds,
    }


def _answer_from_context(
    example: dict[str, Any],
    config,
    answer_client: OpenAITextClient,
    retrieved,
) -> str:
    context = format_graphrag_context(retrieved)
    prompt = build_answer_prompt(
        question=example.get("question", ""),
        answer_format=example.get("answer_format", ""),
        kg_context=context,
        allow_without_context=config.QA_EVAL_ALLOW_ANSWER_WITHOUT_CONTEXT,
        strict_context_grounding=getattr(config, "QA_EVAL_STRICT_CONTEXT_GROUNDING", False),
    )
    return answer_client.complete(prompt)


def _is_insufficient_answer(answer: str) -> bool:
    return answer.strip() == "[INSUFFICIENT_CONTEXT]"


def _answer_cache_key(config, graphrag_index) -> dict[str, Any]:
    return {
        "graph_hash": graphrag_index.graph_hash,
        "answer_model": str(config.QA_EVAL_MODEL),
        "retrieval_mode": "hybrid" if graphrag_index.embedding_index else "bm25",
        "top_communities": config.QA_EVAL_GRAPHRAG_TOP_COMMUNITIES,
        "top_edges": config.QA_EVAL_GRAPHRAG_TOP_EDGES,
        "max_context_chars": config.QA_EVAL_MAX_CONTEXT_CHARS,
        "taxonomy_edge_fraction": config.QA_EVAL_GRAPHRAG_TAXONOMY_EDGE_FRACTION,
        "taxonomy_community_limit": config.QA_EVAL_GRAPHRAG_TAXONOMY_COMMUNITY_LIMIT,
        "bm25_weight": graphrag_index.bm25_weight,
        "embedding_weight": graphrag_index.embedding_weight,
        "embedding_model": getattr(graphrag_index.embedding_index, "model", None),
        "embedding_dimensions": getattr(graphrag_index.embedding_index, "dimensions", None),
        "allow_without_context": config.QA_EVAL_ALLOW_ANSWER_WITHOUT_CONTEXT,
        "strict_context_grounding": getattr(config, "QA_EVAL_STRICT_CONTEXT_GROUNDING", False),
        "retrieval_strategy": str(getattr(config, "QA_EVAL_GRAPHRAG_RETRIEVAL_STRATEGY", "edge_topk")),
        "node_bundle_max_candidates": getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_MAX_CANDIDATES", None),
        "node_bundle_initial_candidates": getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_INITIAL_CANDIDATES", None),
        "node_bundle_batch_size": getattr(config, "QA_EVAL_GRAPHRAG_NODE_BUNDLE_BATCH_SIZE", None),
        "node_bundle_max_edges": getattr(config, "QA_EVAL_GRAPHRAG_MAX_EDGES_PER_NODE_BUNDLE", None),
        "taxonomy_ancestor_depth": getattr(config, "QA_EVAL_GRAPHRAG_TAXONOMY_ANCESTOR_DEPTH", None),
    }


def _evaluate_answer(answer: str, example: dict[str, Any], config) -> dict[str, Any] | None:
    if not getattr(config, "QA_EVAL_RUN_AIRQA_EVALUATOR", True):
        return None
    evaluator = example.get("evaluator", {})
    if _uses_llm_evaluator(evaluator) and not getattr(config, "QA_EVAL_ALLOW_LLM_EVALUATORS", False):
        return {
            "score": None,
            "skipped": True,
            "reason": "LLM evaluator is disabled by QA_EVAL_ALLOW_LLM_EVALUATORS=False.",
        }

    evaluate_airqa = _load_airqa_evaluator(config)
    try:
        score = evaluate_airqa(answer, _with_airqa_evaluator_model(example, config))
        return {"score": score, "skipped": False}
    except Exception as exc:
        return {"score": 0.0, "skipped": False, "error": str(exc)}


def _with_airqa_evaluator_model(example: dict[str, Any], config) -> dict[str, Any]:
    model = str(getattr(config, "QA_EVAL_AIRQA_EVALUATOR_MODEL", "") or "").strip()
    evaluator = example.get("evaluator", {})
    if not model or not _uses_llm_evaluator(evaluator):
        return example

    updated = dict(example)
    updated_evaluator = dict(evaluator)
    kwargs = dict(updated_evaluator.get("eval_kwargs", {}))
    func_name = str(updated_evaluator.get("eval_func", ""))
    if "llm" in func_name.lower():
        kwargs["llm_model"] = model

    func_list = kwargs.get("eval_func_list", [])
    kwargs_list = kwargs.get("eval_kwargs_list", [])
    if isinstance(func_list, list) and isinstance(kwargs_list, list):
        new_kwargs_list = []
        for nested_func, nested_kwargs in zip(func_list, kwargs_list):
            nested = dict(nested_kwargs or {})
            if "llm" in str(nested_func).lower():
                nested["llm_model"] = model
            new_kwargs_list.append(nested)
        kwargs["eval_kwargs_list"] = new_kwargs_list

    updated_evaluator["eval_kwargs"] = kwargs
    updated["evaluator"] = updated_evaluator
    return updated


def _load_airqa_evaluator(config):
    global _AIRQA_EVALUATE_FUNC
    if _AIRQA_EVALUATE_FUNC is not None:
        return _AIRQA_EVALUATE_FUNC

    qa_root = Path(config.PROJECT_ROOT) / "qa_extractor"
    qa_root_text = str(qa_root)
    if qa_root_text in sys.path:
        sys.path.remove(qa_root_text)
    sys.path.insert(0, qa_root_text)
    # OntoGen also has an ontogen/utils.py module. If it was imported as plain
    # "utils" first, AirQA's evaluator cannot import utils.llm_utils.
    for module_name in list(sys.modules):
        if (
            module_name == "utils"
            or module_name.startswith("utils.")
            or module_name == "evaluation"
            or module_name.startswith("evaluation.")
        ):
            del sys.modules[module_name]
    from evaluation.evaluator import evaluate_airqa
    _AIRQA_EVALUATE_FUNC = evaluate_airqa
    return evaluate_airqa


def _uses_llm_evaluator(evaluator: dict[str, Any]) -> bool:
    func = str(evaluator.get("eval_func", "")).lower()
    if "llm" in func:
        return True
    kwargs = evaluator.get("eval_kwargs", {})
    if isinstance(kwargs, dict):
        funcs = kwargs.get("eval_func_list", [])
        if isinstance(funcs, list) and any("llm" in str(item).lower() for item in funcs):
            return True
    return False


def _summarize_results(records: list[dict[str, Any]], base_summary: dict[str, Any], answers_path: Path) -> dict[str, Any]:
    scored = [
        record["airqa_score"]["score"]
        for record in records
        if isinstance(record.get("airqa_score"), dict) and record["airqa_score"].get("score") is not None
    ]
    skipped = [
        record
        for record in records
        if isinstance(record.get("airqa_score"), dict) and record["airqa_score"].get("skipped")
    ]
    summary = dict(base_summary)
    summary.update(
        {
            "answers_jsonl": str(answers_path),
            "evaluated_examples": len(records),
            "scored_examples": len(scored),
            "skipped_evaluators": len(skipped),
            "average_score": sum(scored) / len(scored) if scored else None,
            "records": records,
        }
    )
    return summary


def _load_existing_answer_records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        uuid = record.get("uuid")
        if uuid:
            records[uuid] = record
    return records


def _ordered_records(examples: list[dict[str, Any]], records_by_uuid: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        records_by_uuid[example.get("uuid")]
        for example in examples
        if example.get("uuid") in records_by_uuid
    ]


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
