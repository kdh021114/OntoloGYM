from __future__ import annotations

from ..common.utils import min_max_scale
from ..core.interfaces import CandidateRanker
from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig
from .base import WeightedStrategy
from .context import RankingContext


class HybridRanker(CandidateRanker):
    def __init__(self, weighted_strategies: list[WeightedStrategy]) -> None:
        self._weighted_strategies = [entry for entry in weighted_strategies if entry.weight > 0]

    def rank(self, candidates: list[CandidatePaper], config: PaperCrawlingConfig) -> list[CandidatePaper]:
        if not candidates:
            return []

        if not self._weighted_strategies:
            for candidate in candidates:
                candidate.ranking_score = 0.0
                candidate.ranking_breakdown = {"total": 0.0}
            return list(candidates)

        context = RankingContext.from_config(config)
        scaled_scores_by_name: dict[str, list[float]] = {}

        for weighted_strategy in self._weighted_strategies:
            raw_scores = weighted_strategy.strategy.score(candidates, context)
            if len(raw_scores) != len(candidates):
                raise ValueError(
                    f"Ranking strategy `{weighted_strategy.strategy.name}` returned "
                    f"{len(raw_scores)} scores for {len(candidates)} candidates."
                )
            scaled_scores_by_name[weighted_strategy.strategy.name] = min_max_scale([float(score) for score in raw_scores])

        for index, candidate in enumerate(candidates):
            total_score = 0.0
            breakdown: dict[str, float] = {}

            for weighted_strategy in self._weighted_strategies:
                strategy_name = weighted_strategy.strategy.name
                scaled_score = scaled_scores_by_name[strategy_name][index]
                total_score += weighted_strategy.weight * scaled_score
                breakdown[strategy_name] = round(scaled_score, 6)

            breakdown["total"] = round(total_score, 6)
            candidate.ranking_score = total_score
            candidate.ranking_breakdown = breakdown

        return sorted(candidates, key=lambda item: item.ranking_score, reverse=True)
