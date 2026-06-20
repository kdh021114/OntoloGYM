from __future__ import annotations

from collections import deque
from pathlib import PurePosixPath
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from ...common.utils import extract_year, polite_sleep
from ...core.models import CandidatePaper
from ...core.runtime_config import PaperCrawlingConfig


class SitePdfDiscoveryClient:
    def __init__(self, http_client: httpx.Client, delay_seconds: float = 1.0) -> None:
        self.http_client = http_client
        self.delay_seconds = delay_seconds

    def discover(
        self,
        *,
        source_key: str,
        source_display_name: str,
        start_urls: list[str],
        allowed_domain: str | None,
        max_depth: int,
        max_pages: int,
        config: PaperCrawlingConfig,
    ) -> list[CandidatePaper]:
        queue: deque[tuple[str, int]] = deque((url, 0) for url in start_urls)
        seen_pages: set[str] = set()
        seen_pdfs: set[str] = set()
        candidates: list[CandidatePaper] = []

        while queue and len(seen_pages) < max_pages and len(candidates) < config.max_candidates_per_source:
            url, depth = queue.popleft()
            normalized_url = url.strip()
            if not normalized_url or normalized_url in seen_pages:
                continue
            if not _allowed_url(normalized_url, allowed_domain):
                continue

            try:
                response = self.http_client.get(normalized_url, follow_redirects=True)
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            seen_pages.add(normalized_url)
            content_type = (response.headers.get("content-type") or "").casefold()

            if "xml" in content_type:
                for discovered_url in _extract_xml_urls(response.text):
                    if discovered_url.casefold().endswith(".pdf"):
                        candidate = _build_site_candidate(source_key, source_display_name, discovered_url, "")
                        if candidate and candidate.landing_url not in seen_pdfs:
                            seen_pdfs.add(candidate.landing_url or discovered_url)
                            if _within_year_range(candidate, config):
                                candidates.append(candidate)
                    elif depth < max_depth:
                        queue.append((discovered_url, depth + 1))
                polite_sleep(self.delay_seconds)
                continue

            if "html" not in content_type:
                polite_sleep(self.delay_seconds)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for href, anchor_text in _iter_links(soup):
                absolute = urljoin(str(response.url), href)
                if not _allowed_url(absolute, allowed_domain):
                    continue
                if absolute.casefold().endswith(".pdf"):
                    candidate = _build_site_candidate(source_key, source_display_name, absolute, anchor_text)
                    if candidate and candidate.landing_url not in seen_pdfs:
                        seen_pdfs.add(candidate.landing_url or absolute)
                        if _within_year_range(candidate, config):
                            candidates.append(candidate)
                        if len(candidates) >= config.max_candidates_per_source:
                            break
                elif depth < max_depth:
                    queue.append((absolute, depth + 1))

            polite_sleep(self.delay_seconds)

        return candidates


def _iter_links(soup: BeautifulSoup) -> Iterable[tuple[str, str]]:
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if href:
            yield href, text


def _extract_xml_urls(text: str) -> list[str]:
    soup = BeautifulSoup(text, "xml")
    urls: list[str] = []
    for loc in soup.find_all("loc"):
        value = (loc.get_text(strip=True) or "").strip()
        if value:
            urls.append(value)
    return urls


def _allowed_url(url: str, allowed_domain: str | None) -> bool:
    if not allowed_domain:
        return True
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").casefold()
    return hostname.endswith(allowed_domain.casefold())


def _build_site_candidate(
    source_key: str,
    source_display_name: str,
    url: str,
    anchor_text: str,
) -> CandidatePaper | None:
    parsed = urlparse(url)
    filename = PurePosixPath(parsed.path).name
    title = anchor_text.strip() or filename.replace("-", " ").replace("_", " ")
    if not title:
        return None
    return CandidatePaper(
        source_key=source_key,
        source_display_name=source_display_name,
        title=title,
        year=extract_year(f"{title} {url}"),
        landing_url=url,
        pdf_urls=[url],
        is_oa=True,
        extra={"site_filename": filename},
    )


def _within_year_range(candidate: CandidatePaper, config: PaperCrawlingConfig) -> bool:
    if candidate.year is None:
        return True
    return config.year_range.start <= candidate.year <= config.year_range.end
