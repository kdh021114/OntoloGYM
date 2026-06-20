from __future__ import annotations

from .base import ConfiguredSource
from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig
from .catalog import SourceDefinition
from .providers.site_pdf_discovery import SitePdfDiscoveryClient


class SitePdfSource(ConfiguredSource):
    def __init__(self, definition: SourceDefinition, site_pdf_discovery_client: SitePdfDiscoveryClient) -> None:
        super().__init__(definition)
        self._site_pdf_discovery_client = site_pdf_discovery_client
        self._start_urls = _require_str_list(definition.metadata, "start_urls", definition.key)
        self._allowed_domain = _require_optional_str(definition.metadata, "allowed_domain")
        self._max_depth = _require_int(definition.metadata, "max_depth", definition.key, default=2)
        self._max_pages = _require_int(definition.metadata, "max_pages", definition.key, default=25)

    def collect(self, config: PaperCrawlingConfig) -> list[CandidatePaper]:
        return self._site_pdf_discovery_client.discover(
            source_key=self.definition.key,
            source_display_name=self.definition.display_name,
            start_urls=self._start_urls,
            allowed_domain=self._allowed_domain,
            max_depth=self._max_depth,
            max_pages=self._max_pages,
            config=config,
        )


def _require_str_list(metadata: dict[str, object] | object, key: str, source_key: str) -> list[str]:
    value = getattr(metadata, "get", lambda _key, _default=None: None)(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing `{key}` metadata for source: {source_key}")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError(f"`{key}` must contain at least one value for source: {source_key}")
    return items


def _require_optional_str(metadata: dict[str, object] | object, key: str) -> str | None:
    value = getattr(metadata, "get", lambda _key, _default=None: None)(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_int(metadata: dict[str, object] | object, key: str, source_key: str, default: int) -> int:
    value = getattr(metadata, "get", lambda _key, _default=None: default)(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `{key}` metadata for source: {source_key}") from exc
