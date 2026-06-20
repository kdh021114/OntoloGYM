from __future__ import annotations

from .base import ConfiguredSource
from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig
from .catalog import SourceDefinition
from .providers.openalex import OpenAlexClient


class OpenAlexJournalSource(ConfiguredSource):
    def __init__(self, definition: SourceDefinition, openalex_client: OpenAlexClient) -> None:
        super().__init__(definition)
        self._openalex_client = openalex_client
        self._openalex_source_id = _require_str(definition.metadata, "openalex_source_id", definition.key)

    def collect(self, config: PaperCrawlingConfig) -> list[CandidatePaper]:
        return self._openalex_client.search_journal(
            source_key=self.definition.key,
            source_display_name=self.definition.display_name,
            openalex_source_id=self._openalex_source_id,
            config=config,
        )


def _require_str(metadata: dict[str, object] | object, key: str, source_key: str) -> str:
    value = getattr(metadata, "get", lambda _key, _default=None: None)(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing `{key}` metadata for source: {source_key}")
    return value.strip()
