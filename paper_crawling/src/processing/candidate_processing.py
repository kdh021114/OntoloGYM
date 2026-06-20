from __future__ import annotations

from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig


class CandidateFilter:
    def apply(self, candidates: list[CandidatePaper], config: PaperCrawlingConfig) -> list[CandidatePaper]:
        if not config.exclude_keywords:
            return list(candidates)

        exclude_terms = [term.casefold() for term in config.exclude_keywords if term.strip()]
        filtered: list[CandidatePaper] = []
        for candidate in candidates:
            haystack = candidate.combined_text().casefold()
            if any(term in haystack for term in exclude_terms):
                continue
            filtered.append(candidate)
        return filtered


class CandidateDeduplicator:
    def deduplicate(self, candidates: list[CandidatePaper]) -> list[CandidatePaper]:
        merged: dict[str, CandidatePaper] = {}

        for candidate in candidates:
            key = candidate.dedupe_key()
            current = merged.get(key)
            if current is None:
                merged[key] = candidate
                continue

            current.pdf_urls = _merge_strings(current.pdf_urls, candidate.pdf_urls)
            current.crossref_pdf_urls = _merge_strings(current.crossref_pdf_urls, candidate.crossref_pdf_urls)
            current.license_urls = _merge_strings(current.license_urls, candidate.license_urls)
            current.authors = _merge_strings(current.authors, candidate.authors)
            current.is_oa = current.is_oa or candidate.is_oa
            current.citation_count = max(current.citation_count, candidate.citation_count)
            if not current.abstract and candidate.abstract:
                current.abstract = candidate.abstract
            if not current.landing_url and candidate.landing_url:
                current.landing_url = candidate.landing_url
            current.extra = {**candidate.extra, **current.extra}

        return list(merged.values())


def _merge_strings(existing: list[str], incoming: list[str]) -> list[str]:
    seen = {value for value in existing if value}
    merged = list(existing)
    for value in incoming:
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged
