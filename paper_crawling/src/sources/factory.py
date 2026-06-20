from __future__ import annotations

from collections.abc import Callable

from ..core.interfaces import PaperSource
from .catalog import SourceDefinition
from .providers.openalex import OpenAlexClient
from .providers.site_pdf_discovery import SitePdfDiscoveryClient
from .openalex_journal_source import OpenAlexJournalSource
from .site_pdf_source import SitePdfSource

SourceBuilder = Callable[[SourceDefinition], PaperSource]


class SourceFactory:
    def __init__(self) -> None:
        self._builders: dict[str, SourceBuilder] = {}

    def register(self, provider_key: str, builder: SourceBuilder) -> None:
        self._builders[provider_key] = builder

    def create(self, definition: SourceDefinition) -> PaperSource:
        builder = self._builders.get(definition.provider_key)
        if builder is None:
            raise KeyError(f"No source builder registered for provider: {definition.provider_key}")
        return builder(definition)

    def create_many(self, definitions: list[SourceDefinition]) -> list[PaperSource]:
        return [self.create(definition) for definition in definitions]


def build_default_source_factory(
    *,
    openalex_client: OpenAlexClient,
    site_pdf_discovery_client: SitePdfDiscoveryClient,
) -> SourceFactory:
    factory = SourceFactory()
    factory.register(
        "openalex_journal",
        lambda definition: OpenAlexJournalSource(definition=definition, openalex_client=openalex_client),
    )
    factory.register(
        "site_pdf",
        lambda definition: SitePdfSource(definition=definition, site_pdf_discovery_client=site_pdf_discovery_client),
    )
    return factory
