from __future__ import annotations

from ..common.utils import tokenize
from ..core.models import CandidatePaper
from .base import RankingStrategy
from .context import RankingContext


class PhraseMatchRanker(RankingStrategy):
    name = "phrase"

    def score(self, candidates: list[CandidatePaper], context: RankingContext) -> list[float]:
        return [_phrase_score(candidate, context) for candidate in candidates]


def _phrase_score(candidate: CandidatePaper, context: RankingContext) -> float:
    title = candidate.title.casefold()
    abstract = candidate.abstract.casefold()
    document_tokens = set(tokenize(candidate.combined_text()))

    score = 0.0
    for phrase in context.phrases:
        lowered = phrase.casefold()
        if lowered in title:
            score += 2.0
        elif lowered in abstract:
            score += 1.0

    query_tokens = set(context.query_tokens)
    if query_tokens:
        score += len(query_tokens & document_tokens) / len(query_tokens)
    return score
