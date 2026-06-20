from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from ...common.utils import polite_sleep
from ...core.models import CandidatePaper


class CrossrefClient:
    def __init__(self, http_client: httpx.Client, delay_seconds: float = 1.0) -> None:
        self.http_client = http_client
        self.delay_seconds = delay_seconds

    def enrich_candidate(self, candidate: CandidatePaper) -> CandidatePaper:
        if not candidate.doi:
            return candidate

        encoded = quote(candidate.doi, safe="")
        response = self.http_client.get(f"https://api.crossref.org/works/{encoded}")
        response.raise_for_status()
        message = (response.json() or {}).get("message") or {}

        candidate.crossref_pdf_urls = _extract_crossref_pdf_urls(message)
        candidate.license_urls = _extract_license_urls(message)
        polite_sleep(self.delay_seconds)
        return candidate


def _extract_crossref_pdf_urls(message: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for link in message.get("link") or []:
        url = (link.get("URL") or "").strip()
        content_type = (link.get("content-type") or "").casefold()
        if not url:
            continue
        is_pdf = "pdf" in content_type or url.casefold().endswith(".pdf")
        if not is_pdf or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _extract_license_urls(message: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for license_info in message.get("license") or []:
        url = (license_info.get("URL") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls
