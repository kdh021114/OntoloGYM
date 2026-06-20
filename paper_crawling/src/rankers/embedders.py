from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ..core.interfaces import TextEmbedder


class SentenceTransformerEmbedder(TextEmbedder):
    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        normalized_texts = [text.strip() if text and text.strip() else "[empty]" for text in texts]

        model = self._load_model()
        embeddings = model.encode(
            normalized_texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.tolist() for embedding in embeddings]

    def _load_model(self) -> Any:
        cached_model = self._model_cache.get(self._model_name)
        if cached_model is not None:
            return cached_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic similarity requires `sentence-transformers`. "
                "Run `pip install -r paper_crawling/requirements.txt` first."
            ) from exc

        model = SentenceTransformer(self._model_name)
        self._model_cache[self._model_name] = model
        return model
