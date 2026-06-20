from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    key: str
    display_name: str
    provider_key: str
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


class SourceCatalog:
    def __init__(self, definitions: list[SourceDefinition]) -> None:
        self._definitions_by_key = {definition.key: definition for definition in definitions}
        self._lookup: dict[str, str] = {}

        for definition in definitions:
            self._register_alias(definition.key, definition.key)
            self._register_alias(definition.display_name, definition.key)
            for alias in definition.aliases:
                self._register_alias(alias, definition.key)

    def resolve(self, raw_source: str) -> SourceDefinition:
        normalized = _normalize_source_name(raw_source)
        key = self._lookup.get(normalized)
        if key is None:
            available = ", ".join(sorted(self._definitions_by_key))
            raise KeyError(f"Unknown source: {raw_source}. Available sources: {available}")
        return self._definitions_by_key[key]

    def resolve_many(self, raw_sources: list[str]) -> list[SourceDefinition]:
        resolved: list[SourceDefinition] = []
        seen: set[str] = set()

        for raw_source in raw_sources:
            definition = self.resolve(raw_source)
            if definition.key in seen:
                continue
            seen.add(definition.key)
            resolved.append(definition)
        return resolved

    def _register_alias(self, raw_name: str, key: str) -> None:
        self._lookup[_normalize_source_name(raw_name)] = key


def build_default_source_catalog() -> SourceCatalog:
    return SourceCatalog(
        definitions=[
            SourceDefinition(
                key="nature",
                display_name="Nature",
                provider_key="openalex_journal",
                aliases=("nature",),
                metadata={
                    "openalex_source_id": "https://openalex.org/S137773608",
                    "crossref_issn": "0028-0836",
                },
            ),
            SourceDefinition(
                key="science",
                display_name="Science",
                provider_key="openalex_journal",
                aliases=("science",),
                metadata={
                    "openalex_source_id": "https://openalex.org/S3880285",
                    "crossref_issn": "0036-8075",
                },
            ),
            SourceDefinition(
                key="nature_energy",
                display_name="Nature Energy",
                provider_key="openalex_journal",
                aliases=("nature energy",),
                metadata={
                    "openalex_source_id": "https://openalex.org/S2764528046",
                    "crossref_issn": "2058-7546",
                },
            ),
            SourceDefinition(
                key="ees",
                display_name="Energy & Environmental Science",
                provider_key="openalex_journal",
                aliases=(
                    "ees",
                    "energy & environmental science",
                    "energy and environmental science",
                ),
                metadata={
                    "openalex_source_id": "https://openalex.org/S117082959",
                    "crossref_issn": "1754-5692",
                },
            ),
            SourceDefinition(
                key="joule",
                display_name="Joule",
                provider_key="openalex_journal",
                aliases=("joule",),
                metadata={
                    "openalex_source_id": "https://openalex.org/S2898305631",
                    "crossref_issn": "2542-4351",
                },
            ),
            SourceDefinition(
                key="advanced_materials",
                display_name="Advanced Materials",
                provider_key="openalex_journal",
                aliases=("advanced materials", "adv materials", "adv mater"),
                metadata={
                    "openalex_source_id": "https://openalex.org/S99352657",
                    "crossref_issn": "0935-9648",
                },
            ),
            SourceDefinition(
                key="advanced_functional_materials",
                display_name="Advanced Functional Materials",
                provider_key="openalex_journal",
                aliases=(
                    "advanced functional materials",
                    "adv functional materials",
                    "adv funct mater",
                    "afm",
                ),
                metadata={
                    "openalex_source_id": "https://openalex.org/S135204980",
                    "crossref_issn": "1616-301X",
                },
            ),
            SourceDefinition(
                key="nature_materials",
                display_name="Nature Materials",
                provider_key="openalex_journal",
                aliases=("nature materials", "nat mater"),
                metadata={
                    "openalex_source_id": "https://openalex.org/S103895331",
                    "crossref_issn": "1476-1122",
                },
            ),
            SourceDefinition(
                key="acs_nano",
                display_name="ACS Nano",
                provider_key="openalex_journal",
                aliases=("acs nano",),
                metadata={
                    "openalex_source_id": "https://openalex.org/S145476921",
                    "crossref_issn": "1936-0851",
                },
            ),
            SourceDefinition(
                key="acs_applied_materials_interfaces",
                display_name="ACS Applied Materials & Interfaces",
                provider_key="openalex_journal",
                aliases=(
                    "acs applied materials & interfaces",
                    "acs applied materials & interface",
                    "acs applied materials and interfaces",
                    "acs applied materials and interface",
                    "acs applied materials interfaces",
                    "acs applied materials interface",
                    "acsami",
                ),
                metadata={
                    "openalex_source_id": "https://openalex.org/S164001016",
                    "crossref_issn": "1944-8244",
                },
            ),
        ]
    )


def _normalize_source_name(raw_name: str) -> str:
    return " ".join(raw_name.strip().casefold().split())
