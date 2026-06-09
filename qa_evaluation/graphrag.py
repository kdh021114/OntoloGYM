from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_.+%-]+")
GRAPHRAG_CACHE_VERSION = "v9"
logger = logging.getLogger(__name__)


GENERIC_ROOT_NODES = {"root", "thing", "entity"}
RELATION_SCORE_HINTS = {
    "REPORTS_RESULT": 2.8,
    "MEASURES_METRIC": 2.4,
    "HAS_CONDITION": 2.2,
    "USES_METHOD": 1.8,
    "USES_MATERIAL": 1.5,
    "DEFINED_AS": 1.0,
    "isA": 0.5,
}
TAXONOMY_SOURCE_TYPES = {"taxonomy"}
TAXONOMY_RELATIONS = {"isA"}
TAXONOMY_QUERY_HINTS = {
    "category",
    "categories",
    "class",
    "classification",
    "classify",
    "concept",
    "conceptual",
    "definition",
    "defined",
    "hierarchy",
    "isa",
    "ontology",
    "taxonomic",
    "taxonomy",
    "type",
    "types",
}


@dataclass(frozen=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    source_path: str
    source_type: str
    paper_id: str = ""
    evidence_quote: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.source} --{self.relation}--> {self.target}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "text": self.text,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "paper_id": self.paper_id,
            "evidence_quote": self.evidence_quote,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class CommunityReport:
    community_id: int
    title: str
    summary: str
    nodes: list[str]
    edges: list[GraphEdge]
    relation_counts: dict[str, int]
    paper_counts: dict[str, int]

    @property
    def search_text(self) -> str:
        pieces = [self.title, self.summary]
        pieces.extend(self.nodes[:80])
        pieces.extend(edge.text for edge in self.edges[:120])
        return " ".join(pieces)

    def to_dict(self, include_edges: bool = True) -> dict[str, Any]:
        data = {
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "nodes": self.nodes,
            "relation_counts": self.relation_counts,
            "paper_counts": self.paper_counts,
        }
        if include_edges:
            data["edges"] = [edge.to_dict() for edge in self.edges]
        return data


@dataclass
class GraphRAGEmbeddingIndex:
    model: str
    dimensions: int | None
    community_texts: list[str]
    edge_texts: list[str]
    node_texts: list[str]
    community_vectors: list[list[float]]
    edge_vectors: list[list[float]]
    node_names: list[str]
    node_vectors: list[list[float]]
    client: Any | None = field(default=None, repr=False)
    query_cache: dict[str, list[float]] = field(default_factory=dict, repr=False)


@dataclass
class GraphRAGIndex:
    edges: list[GraphEdge]
    communities: list[CommunityReport]
    graph_hash: str
    embedding_index: GraphRAGEmbeddingIndex | None = None
    bm25_weight: float = 1.0
    embedding_weight: float = 0.0


@dataclass
class GraphNodeBundle:
    node: str
    retrieval_method: str
    score: float
    edges: list[GraphEdge]
    taxonomy_edges: list[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "retrieval_method": self.retrieval_method,
            "score": self.score,
            "edges": [edge.to_dict() for edge in self.edges],
            "taxonomy_edges": [edge.to_dict() for edge in self.taxonomy_edges],
        }


@dataclass
class RetrievedGraphContext:
    communities: list[CommunityReport]
    edges: list[GraphEdge]
    node_bundles: list[GraphNodeBundle] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "communities": [community.to_dict(include_edges=False) for community in self.communities],
            "edges": [edge.to_dict() for edge in self.edges],
            "node_bundles": [bundle.to_dict() for bundle in self.node_bundles],
        }


def load_graphrag_index(
    kg_dirs: list[Path],
    ontogen_root: Path,
    cache_dir: Path,
    report_client: Any | None,
    *,
    max_communities: int,
    max_edges_per_report: int,
    max_nodes_per_community: int = 160,
    enable_llm_reports: bool,
    embedding_client: Any | None = None,
    enable_embeddings: bool = False,
    embedding_dimensions: int | None = None,
    embedding_batch_size: int = 64,
    bm25_weight: float = 0.55,
    embedding_weight: float = 0.45,
) -> GraphRAGIndex:
    edges = _dedupe_edges(_load_graph_edges(kg_dirs, ontogen_root))
    graph_hash = _graph_hash(edges)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_mode = "llm" if enable_llm_reports and report_client is not None else "deterministic"
    report_model = str(getattr(report_client, "model", "") or "none") if report_mode == "llm" else "none"
    report_model_hash = hashlib.sha1(report_model.encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / (
        f"community_reports_{GRAPHRAG_CACHE_VERSION}_{report_mode}_{report_model_hash}_{graph_hash}.json"
    )
    if cache_path.exists():
        index = _load_cached_index(cache_path, edges, graph_hash)
    else:
        communities = _detect_communities(
            edges,
            max_communities=max_communities,
            max_nodes_per_community=max_nodes_per_community,
        )
        reports = []
        for community_id, node_ids in enumerate(communities):
            community_edges = _community_edges(edges, node_ids)
            if not community_edges:
                continue
            report = _build_report(
                community_id=community_id,
                node_ids=node_ids,
                edges=community_edges,
                report_client=report_client if enable_llm_reports else None,
                max_edges_per_report=max_edges_per_report,
            )
            reports.append(report)

        payload = {
            "graph_hash": graph_hash,
            "cache_version": GRAPHRAG_CACHE_VERSION,
            "report_mode": report_mode,
            "report_model": report_model,
            "edges": [edge.to_dict() for edge in edges],
            "communities": [report.to_dict(include_edges=True) for report in reports],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        index = GraphRAGIndex(edges=edges, communities=reports, graph_hash=graph_hash)

    index.bm25_weight = bm25_weight
    index.embedding_weight = embedding_weight if enable_embeddings and embedding_client is not None else 0.0
    if enable_embeddings and embedding_client is not None:
        index.embedding_index = _load_or_build_embedding_index(
            index=index,
            cache_dir=cache_dir,
            embedding_client=embedding_client,
            embedding_dimensions=embedding_dimensions,
            embedding_batch_size=embedding_batch_size,
        )
    return index


def retrieve_graphrag_context(
    question: str,
    index: GraphRAGIndex,
    *,
    top_communities: int,
    top_edges: int,
    max_context_chars: int,
    preferred_paper_ids: set[str] | None = None,
    taxonomy_edge_fraction: float = 0.18,
    taxonomy_community_limit: int = 1,
) -> RetrievedGraphContext:
    del preferred_paper_ids  # Paper provenance is not used as an oracle retrieval signal.
    del top_communities, taxonomy_community_limit

    # Community-first retrieval was too noisy for the current scientific QA data:
    # a wrong community could push unrelated edges above directly relevant facts.
    # Retrieve edges directly, then pass only those evidence edges to the answerer.

    query_tokens = _token_list(question)
    taxonomy_query = _is_taxonomy_query(question, query_tokens)
    query_vector = _query_embedding(index, question)

    edge_scores = _hybrid_document_scores(
        query_tokens,
        [_edge_description(edge) for edge in index.edges],
        query_vector,
        index.embedding_index.edge_vectors if index.embedding_index else None,
        bm25_weight=index.bm25_weight,
        embedding_weight=index.embedding_weight,
    )
    candidate_edges = []
    for base_score, edge in zip(edge_scores, index.edges):
        if base_score > 0:
            candidate_edges.append((base_score, edge))

    candidate_edges.sort(key=lambda item: item[0], reverse=True)

    selected_edges = []
    selected_edge_keys = set()
    taxonomy_edge_limit = _taxonomy_edge_limit(
        top_edges=top_edges,
        taxonomy_edge_fraction=taxonomy_edge_fraction,
        taxonomy_query=taxonomy_query,
    )
    taxonomy_edge_count = 0
    edge_char_budget = max_context_chars
    used_edge_chars = 0
    for _, edge in candidate_edges:
        if len(selected_edges) >= top_edges:
            break
        edge_key = _edge_key(edge)
        if edge_key in selected_edge_keys:
            continue
        is_taxonomy_edge = _is_taxonomy_edge(edge)
        if is_taxonomy_edge and taxonomy_edge_count >= taxonomy_edge_limit:
            continue
        edge_chars = _estimated_edge_chars(edge)
        if used_edge_chars + edge_chars > edge_char_budget and selected_edges:
            continue
        selected_edges.append(edge)
        selected_edge_keys.add(edge_key)
        if is_taxonomy_edge:
            taxonomy_edge_count += 1
        used_edge_chars += edge_chars

    return RetrievedGraphContext(communities=[], edges=selected_edges)


def retrieve_graphrag_node_bundles(
    question: str,
    index: GraphRAGIndex,
    *,
    max_candidates: int,
    max_edges_per_bundle: int | None,
    taxonomy_ancestor_depth: int,
) -> list[GraphNodeBundle]:
    query_tokens = _token_list(question)
    query_vector = _query_embedding(index, question)
    candidates = _rank_node_candidates(
        index=index,
        query_tokens=query_tokens,
        query_vector=query_vector,
        max_candidates=max_candidates,
    )
    bundles = []
    for method, node, score in candidates:
        bundle = _build_node_bundle(
            node=node,
            retrieval_method=method,
            score=score,
            question_tokens=query_tokens,
            index=index,
            max_edges_per_bundle=max_edges_per_bundle,
            taxonomy_ancestor_depth=taxonomy_ancestor_depth,
        )
        if bundle.edges or bundle.taxonomy_edges:
            bundles.append(bundle)
        if len(bundles) >= max_candidates:
            break
    return bundles


def node_bundles_to_context(
    bundles: list[GraphNodeBundle],
    *,
    max_context_chars: int,
) -> RetrievedGraphContext:
    selected_bundles = []
    selected_edges = []
    selected_edge_keys = set()
    used_chars = 0
    for bundle in bundles:
        bundle_edges = []
        bundle_taxonomy_edges = []
        bundle_chars = 160 + len(bundle.node)
        for edge in bundle.edges + bundle.taxonomy_edges:
            edge_key = _edge_key(edge)
            if edge_key in selected_edge_keys:
                continue
            edge_chars = _estimated_edge_chars(edge)
            if used_chars + bundle_chars + edge_chars > max_context_chars and selected_edges:
                continue
            selected_edge_keys.add(edge_key)
            selected_edges.append(edge)
            if edge in bundle.taxonomy_edges:
                bundle_taxonomy_edges.append(edge)
            else:
                bundle_edges.append(edge)
            bundle_chars += edge_chars
        if bundle_edges or bundle_taxonomy_edges:
            selected_bundles.append(
                GraphNodeBundle(
                    node=bundle.node,
                    retrieval_method=bundle.retrieval_method,
                    score=bundle.score,
                    edges=bundle_edges,
                    taxonomy_edges=bundle_taxonomy_edges,
                )
            )
            used_chars += bundle_chars
    return RetrievedGraphContext(communities=[], edges=selected_edges, node_bundles=selected_bundles)


def _edge_retrieval_score(
    edge: GraphEdge,
    *,
    base_score: float,
    query: str,
    in_selected_community: bool,
    seed_nodes: set[str],
    taxonomy_query: bool,
) -> float:
    relation_bonus = _relation_score_hint(edge.relation)
    community_bonus = 2.0 if in_selected_community else 0.0
    node_bonus = 1.5 if edge.source in seed_nodes or edge.target in seed_nodes else 0.0
    query_lower = query.lower()
    phrase_bonus = 0.0
    for phrase in (edge.source, edge.target):
        phrase = phrase.lower().strip()
        if len(phrase) >= 4 and phrase in query_lower:
            phrase_bonus += 5.0
    if base_score <= 0 and phrase_bonus <= 0 and not in_selected_community and node_bonus <= 0:
        return 0.0
    confidence_bonus = min(1.0, edge.confidence or 0.0)
    if _is_taxonomy_edge(edge) and not taxonomy_query:
        base_score *= 0.55
        relation_bonus *= 0.15
        community_bonus *= 0.15
        node_bonus *= 0.25
        confidence_bonus *= 0.50
    return base_score + relation_bonus + community_bonus + node_bonus + phrase_bonus + confidence_bonus


def format_graphrag_context(context: RetrievedGraphContext) -> str:
    if not context.communities and not context.edges:
        return "(No GraphRAG context retrieved.)"
    lines = []
    if context.node_bundles:
        lines.append("GraphRAG retrieved node bundles:")
        for bundle_index, bundle in enumerate(context.node_bundles, start=1):
            lines.append(
                f"[Bundle {bundle_index}] node={bundle.node} "
                f"(retrieved_by={bundle.retrieval_method}; score={bundle.score:.4f})"
            )
            for edge_index, edge in enumerate(bundle.edges, start=1):
                lines.append(f"  - {edge_index}. {_format_edge_line(edge)}")
            if bundle.taxonomy_edges:
                lines.append("  Taxonomy context:")
                for edge in bundle.taxonomy_edges:
                    lines.append(f"  - {_format_edge_line(edge)}")
        return "\n".join(lines)
    if context.communities:
        lines.append("GraphRAG community reports:")
        for community in context.communities:
            relations = ", ".join(_top_counter_labels(community.relation_counts, 6)) or "unknown"
            lines.append(
                f"[Community {community.community_id}] {community.title}\n"
                f"Summary: {community.summary}\n"
                f"Relations: {relations}"
            )
    lines.append("Supporting graph edges:")
    for index, edge in enumerate(context.edges, start=1):
        lines.append(f"[Edge {index}] {_format_edge_line(edge)}")
    return "\n".join(lines)


def _format_edge_line(edge: GraphEdge) -> str:
    provenance = [f"source_type={edge.source_type}", f"source={Path(edge.source_path).name}"]
    if edge.paper_id:
        provenance.append(f"paper={edge.paper_id}")
    qualifier_text = ""
    qualifiers = {}
    if isinstance(edge.metadata, dict):
        raw_qualifiers = edge.metadata.get("qualifiers")
        if isinstance(raw_qualifiers, dict):
            qualifiers = {
                str(key): str(value)
                for key, value in raw_qualifiers.items()
                if value is not None and str(value).strip()
            }
    if qualifiers:
        short_qualifiers = dict(list(qualifiers.items())[:5])
        qualifier_text = f" qualifiers={json.dumps(short_qualifiers, ensure_ascii=False)}"
    evidence_text = f" evidence={json.dumps(edge.evidence_quote, ensure_ascii=False)}" if edge.evidence_quote else ""
    return f"{edge.text}{qualifier_text}{evidence_text} ({'; '.join(provenance)})"


def _load_graph_edges(kg_dirs: list[Path], ontogen_root: Path) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for kg_dir in kg_dirs:
        kg_dir = Path(kg_dir)
        if not kg_dir.exists():
            continue
        edges.extend(_load_relation_graphs(kg_dir))
        edges.extend(_load_relation_claims(kg_dir))
        edges.extend(_load_termo_csvs(kg_dir))
        edges.extend(_load_taxonomy_pickles(kg_dir, ontogen_root))
    return edges


def _load_relation_graphs(root: Path) -> list[GraphEdge]:
    edges = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {
            "kg_context.json",
            "evaluation_results.json",
            "dry_run_summary.json",
            "run_summary.json",
            "community_reports.json",
        }:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_edges = data.get("edges") if isinstance(data, dict) else None
        if not isinstance(raw_edges, list):
            continue
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            source = _clean(raw_edge.get("source"))
            relation = _clean(raw_edge.get("relation"))
            target = _clean(raw_edge.get("target"))
            if not source or not relation or not target:
                continue
            provenance = raw_edge.get("provenance") if isinstance(raw_edge.get("provenance"), dict) else {}
            edges.append(
                GraphEdge(
                    source=source,
                    relation=relation,
                    target=target,
                    source_path=str(path),
                    source_type="relation_graph",
                    paper_id=_clean(provenance.get("paper_id")),
                    evidence_quote=_clean(raw_edge.get("evidence_quote")),
                    confidence=_float_or_none(raw_edge.get("confidence")),
                    metadata=raw_edge,
                )
            )
    return edges


def _load_relation_claims(root: Path) -> list[GraphEdge]:
    edges = []
    for path in sorted(root.rglob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = _clean(data.get("subject"))
            relation = _clean(data.get("relation"))
            target = _clean(data.get("object"))
            if not source or not relation or not target:
                continue
            provenance = data.get("source") if isinstance(data.get("source"), dict) else {}
            edges.append(
                GraphEdge(
                    source=source,
                    relation=relation,
                    target=target,
                    source_path=str(path),
                    source_type="relation_claim",
                    paper_id=_clean(provenance.get("paper_id")),
                    evidence_quote=_clean(data.get("evidence_quote")),
                    confidence=_float_or_none(data.get("confidence")),
                    metadata=data,
                )
            )
    return edges


def _load_termo_csvs(root: Path) -> list[GraphEdge]:
    edges = []
    for path in sorted(root.rglob("*.relationships.csv")):
        for row in _read_csv_rows(path):
            if len(row) >= 3:
                source = _clean(row[0])
                relation = _clean(row[1])
                target = _clean(row[2])
                if source and relation and target:
                    edges.append(
                        GraphEdge(
                            source=source,
                            relation=relation,
                            target=target,
                            source_path=str(path),
                            source_type="termo_relationship",
                            paper_id=_paper_id_from_path(path),
                        )
                    )
    for path in sorted(root.rglob("*.definitions.csv")):
        for row in _read_csv_rows(path):
            if len(row) >= 2:
                source = _clean(row[0])
                target = _clean(row[1])
                if source and target:
                    edges.append(
                        GraphEdge(
                            source=source,
                            relation="DEFINED_AS",
                            target=target,
                            source_path=str(path),
                            source_type="termo_definition",
                            paper_id=_paper_id_from_path(path),
                        )
                    )
    return edges


def _load_taxonomy_pickles(root: Path, ontogen_root: Path) -> list[GraphEdge]:
    edges = []
    if str(ontogen_root) not in sys.path:
        sys.path.insert(0, str(ontogen_root))
    for path in sorted(root.rglob("tree_*.pkl")):
        try:
            with path.open("rb") as handle:
                tree = pickle.load(handle)
        except Exception:
            continue
        for line in str(tree).splitlines():
            line = _clean(line)
            if " isA " not in line:
                continue
            child, parent = line.split(" isA ", 1)
            child = _clean(child)
            parent = _clean(parent)
            if child and parent:
                edges.append(
                    GraphEdge(
                        source=child,
                        relation="isA",
                        target=parent,
                        source_path=str(path),
                        source_type="taxonomy",
                        paper_id=_paper_id_from_path(path),
                    )
                )
    return edges


def _detect_communities(
    edges: list[GraphEdge],
    max_communities: int,
    max_nodes_per_community: int,
) -> list[set[str]]:
    try:
        return _detect_communities_with_networkx(
            edges,
            max_communities=max_communities,
            max_nodes_per_community=max_nodes_per_community,
        )
    except ImportError:
        return _detect_communities_weighted(
            edges,
            max_communities=max_communities,
            max_nodes_per_community=max_nodes_per_community,
        )
    except Exception as exc:
        logger.warning("NetworkX community detection failed; using deterministic weighted fallback. Error: %s", exc)
        return _detect_communities_weighted(
            edges,
            max_communities=max_communities,
            max_nodes_per_community=max_nodes_per_community,
        )


def _detect_communities_with_networkx(
    edges: list[GraphEdge],
    max_communities: int,
    max_nodes_per_community: int,
) -> list[set[str]]:
    import networkx as nx

    graph = nx.Graph()
    for edge in edges:
        source = edge.source.strip()
        target = edge.target.strip()
        if not source or not target:
            continue
        if source.lower() in GENERIC_ROOT_NODES or target.lower() in GENERIC_ROOT_NODES:
            continue
        weight = _community_edge_weight(edge)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)

    if graph.number_of_nodes() == 0:
        return []

    communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=13)
    return _refine_communities(
        communities,
        edges=edges,
        max_communities=max_communities,
        max_nodes_per_community=max_nodes_per_community,
    )


def _detect_communities_weighted(
    edges: list[GraphEdge],
    max_communities: int,
    max_nodes_per_community: int,
) -> list[set[str]]:
    adjacency = _weighted_adjacency(edges)
    if not adjacency:
        return []

    labels = {node: node for node in adjacency}
    nodes = sorted(
        adjacency,
        key=lambda node: (-sum(adjacency[node].values()), node.lower()),
    )
    for _ in range(25):
        changed = False
        for node in nodes:
            scores: dict[str, float] = defaultdict(float)
            for neighbor, weight in adjacency[node].items():
                scores[labels[neighbor]] += weight
            if not scores:
                continue
            best_score = max(scores.values())
            best_label = sorted(label for label, score in scores.items() if score == best_score)[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    grouped: dict[str, set[str]] = defaultdict(set)
    for node, label in labels.items():
        grouped[label].add(node)
    communities = list(grouped.values())

    # Label propagation can occasionally collapse a dense component too far.
    # If that happens, connected components preserve a safer graph-based partition.
    components = _connected_components_from_adjacency(adjacency)
    if len(communities) < min(max_communities, len(components)) / 3:
        communities = components
    return _refine_communities(
        communities,
        edges=edges,
        max_communities=max_communities,
        max_nodes_per_community=max_nodes_per_community,
    )


def _weighted_adjacency(edges: list[GraphEdge]) -> dict[str, dict[str, float]]:
    weighted: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in edges:
        source = edge.source.strip()
        target = edge.target.strip()
        if not source or not target:
            continue
        if source.lower() in GENERIC_ROOT_NODES or target.lower() in GENERIC_ROOT_NODES:
            continue
        weight = _community_edge_weight(edge)
        weighted[source][target] = weighted[source].get(target, 0.0) + weight
        weighted[target][source] = weighted[target].get(source, 0.0) + weight
    return weighted


def _connected_components_from_adjacency(adjacency: dict[str, dict[str, float]]) -> list[set[str]]:
    if not adjacency:
        return []
    seen = set()
    communities = []
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(set(adjacency[current]) - seen)
        communities.append(component)
    return communities


def _refine_communities(
    raw_communities,
    *,
    edges: list[GraphEdge],
    max_communities: int,
    max_nodes_per_community: int,
) -> list[set[str]]:
    communities = [set(community) for community in raw_communities if community]
    if not communities:
        return []

    adjacency = _weighted_adjacency(edges)
    refined = []
    for community in communities:
        refined.extend(
            _split_oversized_community(
                community,
                adjacency=adjacency,
                max_nodes_per_community=max_nodes_per_community,
            )
        )

    refined.sort(
        key=lambda community: (
            -_community_importance(community, adjacency),
            -len(community),
            sorted(community)[0].lower(),
        )
    )
    if max_communities and len(refined) > max_communities:
        refined = refined[:max_communities]
    return refined


def _split_oversized_community(
    community: set[str],
    *,
    adjacency: dict[str, dict[str, float]],
    max_nodes_per_community: int,
) -> list[set[str]]:
    if max_nodes_per_community <= 0 or len(community) <= max_nodes_per_community:
        return [community]

    remaining = set(community)
    chunks = []
    while remaining:
        seed = max(
            remaining,
            key=lambda node: (
                sum(weight for neighbor, weight in adjacency.get(node, {}).items() if neighbor in remaining),
                len(adjacency.get(node, {})),
                node.lower(),
            ),
        )
        chunk = set()
        frontier = [seed]
        queued = {seed}
        while frontier and len(chunk) < max_nodes_per_community:
            node = frontier.pop(0)
            if node not in remaining:
                continue
            chunk.add(node)
            remaining.remove(node)
            neighbors = sorted(
                (neighbor for neighbor in adjacency.get(node, {}) if neighbor in remaining and neighbor not in queued),
                key=lambda neighbor: (-adjacency[node][neighbor], neighbor.lower()),
            )
            for neighbor in neighbors:
                frontier.append(neighbor)
                queued.add(neighbor)
                if len(frontier) + len(chunk) >= max_nodes_per_community * 2:
                    break

        if not chunk:
            chunk.add(remaining.pop())
        chunks.append(chunk)
    return chunks


def _community_importance(community: set[str], adjacency: dict[str, dict[str, float]]) -> float:
    importance = 0.0
    for node in community:
        importance += sum(weight for neighbor, weight in adjacency.get(node, {}).items() if neighbor in community)
    return importance / 2.0


def _community_edge_weight(edge: GraphEdge) -> float:
    if edge.relation == "isA":
        return 0.75
    if edge.relation == "DEFINED_AS":
        return 1.0
    return 2.0 + min(1.0, edge.confidence or 0.0)


def _community_edges(edges: list[GraphEdge], node_ids: set[str]) -> list[GraphEdge]:
    selected = [edge for edge in edges if edge.source in node_ids and edge.target in node_ids]
    selected.sort(
        key=lambda edge: (
            edge.relation == "isA",
            -_relation_score_hint(edge.relation),
            -(edge.confidence or 0),
            edge.relation,
            edge.source,
        )
    )
    return selected


def _build_report(
    community_id: int,
    node_ids: set[str],
    edges: list[GraphEdge],
    report_client: Any | None,
    max_edges_per_report: int,
) -> CommunityReport:
    relation_counts = Counter(edge.relation for edge in edges)
    paper_counts = Counter(edge.paper_id for edge in edges if edge.paper_id)
    title = _community_title(node_ids, edges, relation_counts)
    summary = _deterministic_summary(node_ids, edges, relation_counts, paper_counts)
    if report_client is not None:
        prompt = _community_report_prompt(title, node_ids, edges[:max_edges_per_report], relation_counts, paper_counts)
        try:
            llm_summary = report_client.complete(prompt).strip()
            if llm_summary:
                summary = llm_summary
        except Exception as exc:
            logger.warning(
                "GraphRAG community report LLM failed for community %s; using deterministic summary. Error: %s",
                community_id,
                exc,
            )
    return CommunityReport(
        community_id=community_id,
        title=title,
        summary=summary,
        nodes=sorted(node_ids),
        edges=edges,
        relation_counts=dict(relation_counts),
        paper_counts=dict(paper_counts),
    )


def _community_title(node_ids: set[str], edges: list[GraphEdge], relation_counts: Counter[str]) -> str:
    node_degree: Counter[str] = Counter()
    for edge in edges:
        node_degree.update([edge.source, edge.target])
    ignored = {"root", "thing", "entity"}
    node_candidates = [
        node
        for node, _ in node_degree.most_common(30)
        if node.lower() not in ignored and len(node) > 2
    ][:4]
    if not node_candidates:
        node_candidates = [
            node
            for node in sorted(node_ids, key=lambda value: (-len(value), value))
            if node.lower() not in ignored
        ][:4]
    relation_label = ", ".join(label for label, _ in relation_counts.most_common(3))
    if node_candidates:
        return f"{'; '.join(node_candidates)} ({relation_label})"
    return f"Graph community ({relation_label})"


def _deterministic_summary(
    node_ids: set[str],
    edges: list[GraphEdge],
    relation_counts: Counter[str],
    paper_counts: Counter[str],
) -> str:
    node_degree: Counter[str] = Counter()
    for edge in edges:
        node_degree.update([edge.source, edge.target])
    nodes = ", ".join(_truncate(node, 80) for node, _ in node_degree.most_common(8))
    relations = ", ".join(f"{name}={count}" for name, count in relation_counts.most_common(6))
    examples = " | ".join(_truncate(edge.text, 180) for edge in _representative_edges(edges, limit=4))
    return (
        f"This community connects {len(node_ids)} nodes through {len(edges)} graph edges. "
        f"Representative nodes: {nodes}. Relations: {relations}. "
        f"Representative edges: {examples}."
    )


def _community_report_prompt(
    title: str,
    node_ids: set[str],
    edges: list[GraphEdge],
    relation_counts: Counter[str],
    paper_counts: Counter[str],
) -> str:
    edge_lines = "\n".join(f"- {edge.text}" for edge in _representative_edges(edges, limit=len(edges)))
    node_lines = ", ".join(sorted(node_ids, key=lambda value: (len(value), value))[:60])
    relation_lines = ", ".join(f"{name}: {count}" for name, count in relation_counts.most_common())
    return f"""You are preparing a GraphRAG community report for scientific QA.

Summarize the graph community below. Use only the listed graph edges.
Return 3-5 concise bullet points. Emphasize methods, materials, conditions,
metrics, and reported results that would help answer paper-specific questions.
Keep the whole report compact, ideally under 180 words.

Community title: {title}
Representative nodes: {node_lines}
Relation counts: {relation_lines}

Graph edges:
{edge_lines}
"""


def _representative_edges(edges: list[GraphEdge], limit: int) -> list[GraphEdge]:
    ranked = sorted(
        edges,
        key=lambda edge: (
            -_relation_score_hint(edge.relation),
            -(edge.confidence or 0),
            len(edge.text),
            edge.source.lower(),
        ),
    )
    return ranked[:limit]


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _load_cached_index(cache_path: Path, current_edges: list[GraphEdge], graph_hash: str) -> GraphRAGIndex:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    edge_by_key = {_edge_key(edge): edge for edge in current_edges}
    reports = []
    for raw_report in data.get("communities", []):
        edges = []
        for raw_edge in raw_report.get("edges", []):
            key = _raw_edge_key(raw_edge)
            edge = edge_by_key.get(key)
            if edge is not None:
                edges.append(edge)
        reports.append(
            CommunityReport(
                community_id=int(raw_report.get("community_id") or 0),
                title=str(raw_report.get("title") or ""),
                summary=str(raw_report.get("summary") or ""),
                nodes=[str(node) for node in raw_report.get("nodes", [])],
                edges=edges,
                relation_counts={str(k): int(v) for k, v in raw_report.get("relation_counts", {}).items()},
                paper_counts={str(k): int(v) for k, v in raw_report.get("paper_counts", {}).items()},
            )
        )
    return GraphRAGIndex(edges=current_edges, communities=reports, graph_hash=graph_hash)


def _load_or_build_embedding_index(
    *,
    index: GraphRAGIndex,
    cache_dir: Path,
    embedding_client: Any,
    embedding_dimensions: int | None,
    embedding_batch_size: int,
) -> GraphRAGEmbeddingIndex:
    community_texts = [_community_description(community) for community in index.communities]
    edge_texts = [_edge_description(edge) for edge in index.edges]
    node_names, node_texts = _node_descriptions(index.edges)
    corpus_hash = _text_corpus_hash(community_texts + edge_texts + node_texts)
    model = str(getattr(embedding_client, "model", "") or "unknown")
    dimensions = embedding_dimensions or getattr(embedding_client, "dimensions", None)
    model_hash = hashlib.sha1(f"{model}:{dimensions}".encode("utf-8")).hexdigest()[:12]
    cache_path = cache_dir / (
        f"retrieval_embeddings_{GRAPHRAG_CACHE_VERSION}_{model_hash}_{index.graph_hash}_{corpus_hash}.json"
    )

    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            embedding_index = GraphRAGEmbeddingIndex(
                model=str(raw.get("model") or model),
                dimensions=raw.get("dimensions"),
                community_texts=community_texts,
                edge_texts=edge_texts,
                node_texts=node_texts,
                community_vectors=_coerce_vectors(raw.get("community_vectors", [])),
                edge_vectors=_coerce_vectors(raw.get("edge_vectors", [])),
                node_names=[str(name) for name in raw.get("node_names", [])],
                node_vectors=_coerce_vectors(raw.get("node_vectors", [])),
                client=embedding_client,
            )
            if _embedding_cache_matches(embedding_index, index, node_names):
                return embedding_index
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Ignoring invalid GraphRAG embedding cache %s: %s", cache_path, exc)

    texts = community_texts + edge_texts + node_texts
    vectors = _embed_texts(embedding_client, texts, batch_size=embedding_batch_size)
    community_end = len(community_texts)
    edge_end = community_end + len(edge_texts)
    embedding_index = GraphRAGEmbeddingIndex(
        model=model,
        dimensions=dimensions,
        community_texts=community_texts,
        edge_texts=edge_texts,
        node_texts=node_texts,
        community_vectors=vectors[:community_end],
        edge_vectors=vectors[community_end:edge_end],
        node_names=node_names,
        node_vectors=vectors[edge_end:],
        client=embedding_client,
    )
    payload = {
        "cache_version": GRAPHRAG_CACHE_VERSION,
        "graph_hash": index.graph_hash,
        "corpus_hash": corpus_hash,
        "model": model,
        "dimensions": dimensions,
        "community_vectors": embedding_index.community_vectors,
        "edge_vectors": embedding_index.edge_vectors,
        "node_names": embedding_index.node_names,
        "node_vectors": embedding_index.node_vectors,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return embedding_index


def _embedding_cache_matches(
    embedding_index: GraphRAGEmbeddingIndex,
    index: GraphRAGIndex,
    node_names: list[str],
) -> bool:
    return (
        len(embedding_index.community_vectors) == len(index.communities)
        and len(embedding_index.edge_vectors) == len(index.edges)
        and embedding_index.node_names == node_names
        and len(embedding_index.node_vectors) == len(node_names)
    )


def _coerce_vectors(raw_vectors: Any) -> list[list[float]]:
    vectors = []
    for raw_vector in raw_vectors or []:
        if not isinstance(raw_vector, list):
            continue
        vectors.append([float(value) for value in raw_vector])
    return vectors


def _embed_texts(embedding_client: Any, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    batch_size = max(1, batch_size)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = embedding_client.embed(batch)
        if len(batch_vectors) != len(batch):
            raise ValueError(
                f"Embedding client returned {len(batch_vectors)} vectors for {len(batch)} texts."
            )
        vectors.extend(batch_vectors)
    return vectors


def _text_corpus_hash(texts: list[str]) -> str:
    payload = "\n---\n".join(_clean(text) for text in texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    deduped = []
    seen = set()
    for edge in edges:
        key = _edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _edge_key(edge: GraphEdge) -> tuple[str, str, str, str, str]:
    return (
        edge.source.lower(),
        edge.relation.lower(),
        edge.target.lower(),
        edge.evidence_quote.lower(),
        edge.paper_id.lower(),
    )


def _raw_edge_key(raw_edge: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _clean(raw_edge.get("source")).lower(),
        _clean(raw_edge.get("relation")).lower(),
        _clean(raw_edge.get("target")).lower(),
        _clean(raw_edge.get("evidence_quote")).lower(),
        _clean(raw_edge.get("paper_id")).lower(),
    )


def _graph_hash(edges: list[GraphEdge]) -> str:
    payload = "\n".join("|".join(_edge_key(edge)) for edge in sorted(edges, key=_edge_key))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _query_embedding(index: GraphRAGIndex, question: str) -> list[float] | None:
    embedding_index = index.embedding_index
    if embedding_index is None or embedding_index.client is None:
        return None
    query_text = f"Scientific QA retrieval query: {_clean(question)}"
    cached = embedding_index.query_cache.get(query_text)
    if cached is not None:
        return cached
    vector = embedding_index.client.embed([query_text])[0]
    embedding_index.query_cache[query_text] = vector
    return vector


def _node_scores(
    question: str,
    query_tokens: list[str],
    query_vector: list[float] | None,
    index: GraphRAGIndex,
) -> list[tuple[float, str]]:
    del question
    embedding_index = index.embedding_index
    if embedding_index is None:
        return []
    scores = _hybrid_document_scores(
        query_tokens,
        embedding_index.node_texts,
        query_vector,
        embedding_index.node_vectors,
        bm25_weight=index.bm25_weight,
        embedding_weight=index.embedding_weight,
    )
    return list(zip(scores, embedding_index.node_names))


def _rank_node_candidates(
    *,
    index: GraphRAGIndex,
    query_tokens: list[str],
    query_vector: list[float] | None,
    max_candidates: int,
) -> list[tuple[str, str, float]]:
    node_names, node_texts = _node_descriptions(index.edges)
    if not node_names:
        return []

    bm25_scores = _bm25_scores(query_tokens, node_texts)
    bm25_ranked = [
        ("bm25", node, score)
        for node, score in sorted(
            zip(node_names, bm25_scores),
            key=lambda item: (item[1], len(item[0])),
            reverse=True,
        )
        if score > 0
    ]

    embedding_ranked: list[tuple[str, str, float]] = []
    embedding_index = index.embedding_index
    if embedding_index is not None and query_vector is not None:
        embedding_scores = [
            _cosine_similarity(query_vector, vector)
            for vector in embedding_index.node_vectors
        ]
        embedding_ranked = [
            ("embedding", node, score)
            for node, score in sorted(
                zip(embedding_index.node_names, embedding_scores),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ]

    candidates = []
    seen_nodes = set()
    max_rank = max(len(bm25_ranked), len(embedding_ranked))
    for rank in range(max_rank):
        for ranked in (bm25_ranked, embedding_ranked):
            if rank >= len(ranked):
                continue
            method, node, score = ranked[rank]
            node_key = node.lower()
            if node_key in seen_nodes:
                continue
            candidates.append((method, node, float(score)))
            seen_nodes.add(node_key)
            if len(candidates) >= max_candidates:
                return candidates

    if not candidates:
        incident_counts = Counter()
        for edge in index.edges:
            if edge.source.lower() not in GENERIC_ROOT_NODES:
                incident_counts[edge.source] += 1
            if edge.target.lower() not in GENERIC_ROOT_NODES:
                incident_counts[edge.target] += 1
        for node, count in incident_counts.most_common(max_candidates):
            candidates.append(("degree_fallback", node, float(count)))
    return candidates[:max_candidates]


def _build_node_bundle(
    *,
    node: str,
    retrieval_method: str,
    score: float,
    question_tokens: list[str],
    index: GraphRAGIndex,
    max_edges_per_bundle: int | None,
    taxonomy_ancestor_depth: int,
) -> GraphNodeBundle:
    incident_edges = [
        edge
        for edge in index.edges
        if edge.source == node or edge.target == node
    ]
    non_taxonomy_edges = _expand_incident_edges_to_event_bundle(index.edges, incident_edges)
    taxonomy_incident_edges = [edge for edge in incident_edges if _is_taxonomy_edge(edge)]
    ranked_edges = sorted(
        non_taxonomy_edges,
        key=lambda edge: _bundle_edge_score(edge, node=node, question_tokens=question_tokens),
        reverse=True,
    )
    if max_edges_per_bundle is not None and max_edges_per_bundle >= 0:
        ranked_edges = ranked_edges[:max_edges_per_bundle]

    taxonomy_nodes = {node}
    for edge in ranked_edges[:12]:
        taxonomy_nodes.add(edge.source)
        taxonomy_nodes.add(edge.target)
    taxonomy_edges = _taxonomy_ancestor_edges(
        index=index,
        nodes=taxonomy_nodes,
        depth=taxonomy_ancestor_depth,
        max_edges=18,
    )
    if not ranked_edges:
        taxonomy_edges = taxonomy_incident_edges + taxonomy_edges
    taxonomy_edges = _dedupe_edges(taxonomy_edges)
    return GraphNodeBundle(
        node=node,
        retrieval_method=retrieval_method,
        score=score,
        edges=ranked_edges,
        taxonomy_edges=taxonomy_edges,
    )


def _expand_incident_edges_to_event_bundle(
    all_edges: list[GraphEdge],
    incident_edges: list[GraphEdge],
) -> list[GraphEdge]:
    non_taxonomy_incident = [edge for edge in incident_edges if not _is_taxonomy_edge(edge)]
    event_sources = {
        edge.source
        for edge in non_taxonomy_incident
        if edge.source and edge.source.lower() not in GENERIC_ROOT_NODES
    }
    expanded = list(non_taxonomy_incident)
    seen = {_edge_key(edge) for edge in expanded}
    for edge in all_edges:
        if _is_taxonomy_edge(edge) or edge.source not in event_sources:
            continue
        key = _edge_key(edge)
        if key in seen:
            continue
        expanded.append(edge)
        seen.add(key)
    return expanded


def _bundle_edge_score(edge: GraphEdge, *, node: str, question_tokens: list[str]) -> float:
    query_token_set = set(question_tokens)
    edge_tokens = _tokens(_edge_description(edge))
    overlap = len(query_token_set.intersection(edge_tokens))
    central_bonus = 1.0 if edge.source == node else 0.8
    confidence_bonus = min(1.0, edge.confidence or 0.0)
    return (
        overlap * 3.0
        + _relation_score_hint(edge.relation)
        + confidence_bonus
        + central_bonus
    )


def _taxonomy_ancestor_edges(
    *,
    index: GraphRAGIndex,
    nodes: set[str],
    depth: int,
    max_edges: int,
) -> list[GraphEdge]:
    if depth <= 0 or max_edges <= 0:
        return []
    parents_by_child: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in index.edges:
        if _is_taxonomy_edge(edge):
            parents_by_child[edge.source].append(edge)

    selected = []
    seen = set()
    frontier = {node for node in nodes if node}
    for _ in range(depth):
        next_frontier = set()
        for node in sorted(frontier):
            for edge in parents_by_child.get(node, [])[:3]:
                key = _edge_key(edge)
                if key in seen:
                    continue
                selected.append(edge)
                seen.add(key)
                next_frontier.add(edge.target)
                if len(selected) >= max_edges:
                    return selected
        frontier = next_frontier
        if not frontier:
            break
    return selected


def _hybrid_document_scores(
    query_tokens: list[str],
    documents: list[str],
    query_vector: list[float] | None,
    document_vectors: list[list[float]] | None,
    *,
    bm25_weight: float,
    embedding_weight: float,
) -> list[float]:
    if not documents:
        return []
    bm25_scores = _bm25_scores(query_tokens, documents)
    if query_vector is None or not document_vectors or len(document_vectors) != len(documents):
        return bm25_scores

    normalized_bm25 = _normalize_scores(bm25_scores)
    embedding_scores = [_cosine_similarity(query_vector, vector) for vector in document_vectors]
    normalized_embeddings = _normalize_scores(embedding_scores)
    total_weight = max(1e-9, bm25_weight + embedding_weight)
    bm25_weight = bm25_weight / total_weight
    embedding_weight = embedding_weight / total_weight
    return [
        8.0 * (bm25_weight * bm25 + embedding_weight * embedding)
        for bm25, embedding in zip(normalized_bm25, normalized_embeddings)
    ]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum <= 0:
        return [0.0 for _ in scores]
    if maximum - minimum < 1e-9:
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(limit))
    left_norm = math.sqrt(sum(value * value for value in left[:limit]))
    right_norm = math.sqrt(sum(value * value for value in right[:limit]))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _community_seed_bonus(community: CommunityReport, seed_nodes: set[str]) -> float:
    if not seed_nodes:
        return 0.0
    overlap = len(seed_nodes.intersection(community.nodes))
    return min(3.0, overlap * 0.8)


def _select_retrieval_communities(
    scored_communities: list[tuple[float, CommunityReport]],
    *,
    top_communities: int,
    taxonomy_query: bool,
    taxonomy_community_limit: int,
) -> list[CommunityReport]:
    if top_communities <= 0:
        return []
    selected = []
    taxonomy_selected = 0
    for _, community in scored_communities:
        if len(selected) >= top_communities:
            break
        if _is_taxonomy_heavy_community(community) and not taxonomy_query:
            if taxonomy_selected >= max(0, taxonomy_community_limit):
                continue
            taxonomy_selected += 1
        selected.append(community)
    if not selected and scored_communities:
        selected.append(scored_communities[0][1])
    return selected


def _is_taxonomy_query(question: str, query_tokens: list[str]) -> bool:
    token_set = set(query_tokens)
    if token_set.intersection(TAXONOMY_QUERY_HINTS):
        return True
    lower = question.lower()
    return any(phrase in lower for phrase in ("what kind of", "what type of", "which type", "which category"))


def _taxonomy_edge_limit(*, top_edges: int, taxonomy_edge_fraction: float, taxonomy_query: bool) -> int:
    if taxonomy_query:
        return max(0, top_edges)
    fraction = min(1.0, max(0.0, taxonomy_edge_fraction))
    return max(1, int(round(top_edges * fraction)))


def _is_taxonomy_edge(edge: GraphEdge) -> bool:
    return edge.relation in TAXONOMY_RELATIONS or edge.source_type in TAXONOMY_SOURCE_TYPES


def _is_taxonomy_heavy_community(community: CommunityReport) -> bool:
    total = sum(community.relation_counts.values())
    if total <= 0:
        return False
    taxonomy_edges = sum(community.relation_counts.get(relation, 0) for relation in TAXONOMY_RELATIONS)
    return taxonomy_edges / total >= 0.60


def _community_description(community: CommunityReport) -> str:
    relations = ", ".join(_top_counter_labels(community.relation_counts, 8)) or "unknown"
    nodes = ", ".join(_truncate(node, 80) for node in community.nodes[:60])
    representative = " | ".join(
        _truncate(_edge_description(edge), 220)
        for edge in _representative_edges(community.edges, limit=8)
    )
    return (
        f"Community description. Title: {community.title}. "
        f"Summary: {community.summary}. Relation distribution: {relations}. "
        f"Important terms: {nodes}. Representative relationships: {representative}."
    )


def _edge_description(edge: GraphEdge) -> str:
    relation_words = edge.relation.replace("_", " ")
    confidence = f" Confidence: {edge.confidence:.2f}." if edge.confidence is not None else ""
    qualifier_text = ""
    if isinstance(edge.metadata, dict):
        qualifiers = edge.metadata.get("qualifiers")
        if isinstance(qualifiers, dict) and qualifiers:
            qualifier_text = f" Qualifiers: {json.dumps(qualifiers, ensure_ascii=False)}."
    evidence = f" Evidence: {edge.evidence_quote}." if edge.evidence_quote else ""
    return (
        f"Relationship description. Source term: {edge.source}. "
        f"Relation: {edge.relation} ({relation_words}). Target term: {edge.target}."
        f"{qualifier_text}{evidence}{confidence}"
    )


def _node_descriptions(edges: list[GraphEdge]) -> tuple[list[str], list[str]]:
    incident_edges: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        if edge.source.lower() not in GENERIC_ROOT_NODES:
            incident_edges[edge.source].append(edge)
        if edge.target.lower() not in GENERIC_ROOT_NODES:
            incident_edges[edge.target].append(edge)

    node_names = sorted(incident_edges, key=lambda node: (-len(incident_edges[node]), node.lower()))
    descriptions = [_node_description(node, incident_edges[node]) for node in node_names]
    return node_names, descriptions


def _node_description(node: str, edges: list[GraphEdge]) -> str:
    relation_counts = Counter(edge.relation for edge in edges)
    relations = ", ".join(_top_counter_labels(dict(relation_counts), 8)) or "unknown"
    examples = []
    for edge in _representative_edges(edges, limit=8):
        if edge.source == node:
            examples.append(f"{edge.source} --{edge.relation}--> {edge.target}")
        else:
            examples.append(f"{node} is target of {edge.source} --{edge.relation}--> {edge.target}")
    return (
        f"Term description. Term: {node}. Incident relation distribution: {relations}. "
        f"Representative incident relationships: {' | '.join(examples)}."
    )


def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


def _token_list(text: str) -> list[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "to",
        "for",
        "with",
        "what",
        "which",
        "how",
        "is",
        "are",
        "was",
        "were",
        "their",
        "paper",
        "according",
        "provided",
        "section",
    }
    return [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2 and token.lower() not in stopwords]


def _bm25_scores(query_tokens: list[str], documents: list[str]) -> list[float]:
    if not query_tokens or not documents:
        return [0.0 for _ in documents]
    doc_counts = [Counter(_token_list(document)) for document in documents]
    doc_lengths = [sum(counts.values()) for counts in doc_counts]
    avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
    doc_freq: Counter[str] = Counter()
    for counts in doc_counts:
        doc_freq.update(counts.keys())

    query_counts = Counter(query_tokens)
    total_docs = len(documents)
    k1 = 1.5
    b = 0.75
    scores = []
    for counts, doc_len in zip(doc_counts, doc_lengths):
        score = 0.0
        for token, query_tf in query_counts.items():
            tf = counts.get(token, 0)
            if tf <= 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += query_tf * idf * (tf * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _estimated_community_chars(community: CommunityReport) -> int:
    return (
        len(community.title)
        + len(community.summary)
        + len(json.dumps(community.relation_counts, ensure_ascii=False))
        + 260
    )


def _estimated_edge_chars(edge: GraphEdge) -> int:
    return len(_format_edge_line(edge)) + len(edge.paper_id) + len(Path(edge.source_path).name) + 40


def _relation_score_hint(relation: str) -> float:
    if relation in RELATION_SCORE_HINTS:
        return RELATION_SCORE_HINTS[relation]
    upper = relation.upper()
    if "RESULT" in upper or "FINDING" in upper:
        return 2.5
    if "METRIC" in upper or "MEASURE" in upper or "VALUE" in upper:
        return 2.2
    if "CONDITION" in upper or "SETTING" in upper:
        return 2.0
    if "METHOD" in upper or "TECHNIQUE" in upper:
        return 1.6
    if "MATERIAL" in upper or "SAMPLE" in upper:
        return 1.4
    return 1.0


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))
    except OSError:
        return []


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _paper_id_from_path(path: Path) -> str:
    stem = path.name
    for suffix in [
        ".relationships.csv",
        ".definitions.csv",
        ".terms.csv",
        ".acronyms.csv",
        ".processed_data.json",
        ".jsonl",
        ".json",
        ".pkl",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split(".processed.", 1)[0]


def _top_counter_labels(counts: dict[str, int], limit: int) -> list[str]:
    return [f"{label}={count}" for label, count in Counter(counts).most_common(limit) if label]
