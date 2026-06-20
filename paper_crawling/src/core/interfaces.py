from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import CandidatePaper
from .runtime_config import PaperCrawlingConfig
from .verification import CandidateVerificationStatus


class PaperSource(Protocol):
    key: str

    def collect(self, config: PaperCrawlingConfig) -> list[CandidatePaper]:
        ...


class CandidateEnricher(Protocol):
    def enrich_candidate(self, candidate: CandidatePaper) -> CandidatePaper:
        ...


class CandidateRanker(Protocol):
    def rank(self, candidates: list[CandidatePaper], config: PaperCrawlingConfig) -> list[CandidatePaper]:
        ...


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class ProgressReporter(Protocol):
    def info(self, message: str) -> None:
        ...


class CandidateVerifier(Protocol):
    def verify_candidate(
        self,
        candidate: CandidatePaper,
        *,
        allow_enrichment: bool = True,
        refresh: bool = False,
    ) -> CandidateVerificationStatus:
        ...
