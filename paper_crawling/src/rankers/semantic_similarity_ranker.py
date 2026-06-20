from __future__ import annotations

from ..core.interfaces import TextEmbedder
from ..core.models import CandidatePaper
from .base import RankingStrategy
from .context import RankingContext


class SemanticSimilarityRanker(RankingStrategy):
    name = "semantic"

    def __init__(self, embedder: TextEmbedder) -> None:
        self._embedder = embedder

    def score(self, candidates: list[CandidatePaper], context: RankingContext) -> list[float]:
        query = context.combined_query.strip()
        if not candidates or not query:
            return [0.0 for _ in candidates]

        documents = [_build_document_text(candidate) for candidate in candidates]
        query_embeddings = self._embedder.encode([query])
        document_embeddings = self._embedder.encode(documents)

        if not query_embeddings or len(document_embeddings) != len(candidates):
            return [0.0 for _ in candidates]

        query_embedding = query_embeddings[0]
        return [_dot_product(query_embedding, document_embedding) for document_embedding in document_embeddings]


def _build_document_text(candidate: CandidatePaper) -> str:
    title = candidate.title.strip()
    abstract = candidate.abstract.strip()

    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    if title:
        return f"Title: {title}"
    if abstract:
        return f"Abstract: {abstract}"
    return ""


def _dot_product(left: list[float], right: list[float]) -> float:
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right)))
