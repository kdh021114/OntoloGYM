from __future__ import annotations

import math

from ..core.models import CandidatePaper
from .base import RankingStrategy
from .context import RankingContext


class CitationRanker(RankingStrategy):
    name = "citation"

    def score(self, candidates: list[CandidatePaper], context: RankingContext) -> list[float]:
        del context
        return [math.log1p(candidate.citation_count) for candidate in candidates]
