from __future__ import annotations

from ..core.runtime_config import PaperCrawlingConfig
from .base import WeightedStrategy
from .bm25_ranker import BM25Ranker
from .citation_ranker import CitationRanker
from .embedders import SentenceTransformerEmbedder
from .hybrid_ranker import HybridRanker
from .phrase_match_ranker import PhraseMatchRanker
from .semantic_similarity_ranker import SemanticSimilarityRanker


def build_default_ranker(config: PaperCrawlingConfig) -> HybridRanker:
    weighted_strategies: list[WeightedStrategy] = []

    if config.ranking.bm25_weight > 0:
        weighted_strategies.append(WeightedStrategy(strategy=BM25Ranker(), weight=config.ranking.bm25_weight))
    if config.ranking.phrase_weight > 0:
        weighted_strategies.append(
            WeightedStrategy(strategy=PhraseMatchRanker(), weight=config.ranking.phrase_weight)
        )
    if config.ranking.citation_weight > 0:
        weighted_strategies.append(WeightedStrategy(strategy=CitationRanker(), weight=config.ranking.citation_weight))
    if config.ranking.semantic_weight > 0:
        weighted_strategies.append(
            WeightedStrategy(
                strategy=SemanticSimilarityRanker(
                    embedder=SentenceTransformerEmbedder(
                        model_name=config.ranking.semantic_model_name,
                        batch_size=config.ranking.semantic_batch_size,
                    )
                ),
                weight=config.ranking.semantic_weight,
            )
        )

    return HybridRanker(weighted_strategies=weighted_strategies)
