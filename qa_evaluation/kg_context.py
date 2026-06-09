from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_.+%-]+")


@dataclass(frozen=True)
class KGFact:
    text: str
    source_path: str
    source_type: str
    paper_id: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "paper_id": self.paper_id,
            "metadata": self.metadata or {},
        }


def load_kg_facts(kg_dirs: list[Path], ontogen_root: Path) -> list[KGFact]:
    facts = []
    for kg_dir in kg_dirs:
        kg_dir = Path(kg_dir)
        if not kg_dir.exists():
            continue
        facts.extend(_load_relation_graphs(kg_dir))
        facts.extend(_load_relation_claims(kg_dir))
        facts.extend(_load_termo_csvs(kg_dir))
        facts.extend(_load_taxonomy_pickles(kg_dir, ontogen_root))
    return _dedupe_facts(facts)


def retrieve_facts(
    question: str,
    facts: list[KGFact],
    top_k: int,
    max_context_chars: int,
    preferred_paper_ids: set[str] | None = None,
) -> list[KGFact]:
    del preferred_paper_ids  # Provenance is kept for display only, not used as an oracle ranking signal.
    query_tokens = _tokens(question)
    scored = []
    for fact in facts:
        fact_tokens = _tokens(fact.text)
        overlap = len(query_tokens & fact_tokens)
        if overlap == 0:
            continue
        score = overlap
        scored.append((score, fact))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = []
    used_chars = 0
    for _, fact in scored:
        if len(selected) >= top_k:
            break
        if used_chars + len(fact.text) > max_context_chars and selected:
            continue
        selected.append(fact)
        used_chars += len(fact.text)
    return selected


def format_context(facts: list[KGFact]) -> str:
    if not facts:
        return "(No KG facts retrieved.)"
    lines = []
    for index, fact in enumerate(facts, start=1):
        provenance = []
        if fact.paper_id:
            provenance.append(f"paper_id={fact.paper_id}")
        provenance.append(f"source={Path(fact.source_path).name}")
        lines.append(f"[KG Fact {index}] {fact.text} ({'; '.join(provenance)})")
    return "\n".join(lines)


def _load_relation_graphs(root: Path) -> list[KGFact]:
    facts = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"kg_context.json", "evaluation_results.json", "dry_run_summary.json", "run_summary.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        edges = data.get("edges") if isinstance(data, dict) else None
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = _clean(edge.get("source"))
            relation = _clean(edge.get("relation"))
            target = _clean(edge.get("target"))
            if not source or not relation or not target:
                continue
            provenance = edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {}
            text = _relation_fact_text(source, relation, target, edge)
            facts.append(
                KGFact(
                    text=text,
                    source_path=str(path),
                    source_type="relation_graph",
                    paper_id=_clean(provenance.get("paper_id")),
                    metadata=edge,
                )
            )
    return facts


def _load_relation_claims(root: Path) -> list[KGFact]:
    facts = []
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
            subject = _clean(data.get("subject"))
            relation = _clean(data.get("relation"))
            obj = _clean(data.get("object"))
            if not subject or not relation or not obj:
                continue
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            text = _relation_fact_text(subject, relation, obj, data)
            facts.append(
                KGFact(
                    text=text,
                    source_path=str(path),
                    source_type="relation_claim",
                    paper_id=_clean(source.get("paper_id")),
                    metadata=data,
                )
            )
    return facts


def _load_termo_csvs(root: Path) -> list[KGFact]:
    facts = []
    for path in sorted(root.rglob("*.relationships.csv")):
        for row in _read_csv_rows(path):
            if len(row) >= 3:
                facts.append(
                    KGFact(
                        text=f"{_clean(row[0])} --{_clean(row[1])}--> {_clean(row[2])}",
                        source_path=str(path),
                        source_type="termo_relationship",
                        paper_id=_paper_id_from_path(path),
                    )
                )
    for path in sorted(root.rglob("*.definitions.csv")):
        for row in _read_csv_rows(path):
            if len(row) >= 2:
                facts.append(
                    KGFact(
                        text=f"{_clean(row[0])}: {_clean(row[1])}",
                        source_path=str(path),
                        source_type="termo_definition",
                        paper_id=_paper_id_from_path(path),
                    )
                )
    return facts


def _load_taxonomy_pickles(root: Path, ontogen_root: Path) -> list[KGFact]:
    facts = []
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
            if " isA " in line:
                facts.append(
                    KGFact(
                        text=line,
                        source_path=str(path),
                        source_type="taxonomy",
                        paper_id=_paper_id_from_path(path),
                    )
                )
    return facts


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))
    except OSError:
        return []


def _dedupe_facts(facts: list[KGFact]) -> list[KGFact]:
    seen = set()
    deduped = []
    for fact in facts:
        key = (fact.text.lower(), fact.source_type, fact.paper_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _tokens(text: str) -> set[str]:
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
    }
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2 and token.lower() not in stopwords}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _relation_fact_text(source: str, relation: str, target: str, payload: dict[str, Any]) -> str:
    text = f"{source} --{relation}--> {target}"
    qualifiers = payload.get("qualifiers") if isinstance(payload.get("qualifiers"), dict) else {}
    if qualifiers:
        text += f" qualifiers={json.dumps(qualifiers, ensure_ascii=False)}"
    evidence = _clean(payload.get("evidence_quote"))
    if evidence:
        text += f" evidence={evidence}"
    return text


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
