from __future__ import annotations

from dataclasses import dataclass

from ..common.utils import tokenize
from ..core.runtime_config import PaperCrawlingConfig


@dataclass(frozen=True, slots=True)
class RankingContext:
    combined_query: str
    query_tokens: tuple[str, ...]
    phrases: tuple[str, ...]

    @classmethod
    def from_config(cls, config: PaperCrawlingConfig) -> "RankingContext":
        combined_query = config.combined_query()
        query_tokens = tuple(tokenize(combined_query))
        if not query_tokens:
            query_tokens = tuple(token for keyword in config.keywords for token in tokenize(keyword))
        return cls(
            combined_query=combined_query,
            query_tokens=query_tokens,
            phrases=tuple(config.phrases()),
        )
