from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.models import CandidatePaper
from .context import RankingContext


class RankingStrategy(ABC):
    name: str

    @abstractmethod
    def score(self, candidates: list[CandidatePaper], context: RankingContext) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class WeightedStrategy:
    strategy: RankingStrategy
    weight: float
