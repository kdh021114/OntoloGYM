from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..common.utils import tokenize
from ..core.models import CandidatePaper
from .base import RankingStrategy
from .context import RankingContext


class BM25Ranker(RankingStrategy):
    name = "bm25"

    def score(self, candidates: list[CandidatePaper], context: RankingContext) -> list[float]:
        if not candidates or not context.query_tokens:
            return [0.0 for _ in candidates]

        corpus_tokens = [tokenize(candidate.combined_text()) or ["_empty_"] for candidate in candidates]
        bm25 = BM25Okapi(corpus_tokens)
        return [float(score) for score in bm25.get_scores(list(context.query_tokens))]
