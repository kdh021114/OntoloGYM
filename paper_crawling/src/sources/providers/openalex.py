from __future__ import annotations

from typing import Any

import httpx

from ...common.utils import polite_sleep
from ...core.interfaces import CandidateVerifier
from ...core.interfaces import ProgressReporter
from ...core.models import CandidatePaper
from ...core.runtime_config import PaperCrawlingConfig


class OpenAlexClient:
    def __init__(
        self,
        http_client: httpx.Client,
        delay_seconds: float = 1.0,
        *,
        candidate_verifier: CandidateVerifier | None = None,
        progress_reporter: ProgressReporter | None = None,
        print_page_progress: bool = False,
    ) -> None:
        self.http_client = http_client
        self.delay_seconds = delay_seconds
        self.candidate_verifier = candidate_verifier
        self.progress_reporter = progress_reporter
        self.print_page_progress = print_page_progress

    def search_journal(
        self,
        *,
        source_key: str,
        source_display_name: str,
        openalex_source_id: str,
        config: PaperCrawlingConfig,
    ) -> list[CandidatePaper]:
        candidates: list[CandidatePaper] = []
        seen_keys: set[str] = set()
        verified_candidate_count = 0
        per_term_limit = min(
            config.max_results_per_retrieval_term,
            config.max_candidates_per_source,
        )
        per_page = min(100, per_term_limit)

        for query in config.retrieval_queries():
            if _should_stop_collection(
                candidate_count=len(candidates),
                verified_candidate_count=verified_candidate_count,
                min_pool_candidates_per_source=config.min_pool_candidates_per_source,
                max_candidates_per_source=config.max_candidates_per_source,
                target_verified_candidates_per_source=config.target_verified_candidates_per_source,
            ):
                break

            page = 1
            collected_for_query = 0
            while not _should_stop_collection(
                candidate_count=len(candidates),
                verified_candidate_count=verified_candidate_count,
                min_pool_candidates_per_source=config.min_pool_candidates_per_source,
                max_candidates_per_source=config.max_candidates_per_source,
                target_verified_candidates_per_source=config.target_verified_candidates_per_source,
            ) and collected_for_query < per_term_limit:
                params = {
                    "search": query,
                    "page": page,
                    "per-page": per_page,
                    "select": ",".join(
                        [
                            "id",
                            "display_name",
                            "doi",
                            "publication_year",
                            "cited_by_count",
                            "abstract_inverted_index",
                            "open_access",
                            "primary_location",
                            "best_oa_location",
                            "locations",
                            "authorships",
                        ]
                    ),
                    "filter": ",".join(
                        [
                            f"primary_location.source.id:{openalex_source_id}",
                            f"from_publication_date:{config.year_range.start}-01-01",
                            f"to_publication_date:{config.year_range.end}-12-31",
                            "is_paratext:false",
                        ]
                    ),
                }
                if config.email:
                    params["mailto"] = config.email

                response = self.http_client.get("https://api.openalex.org/works", params=params)
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("results") or []
                if not rows:
                    break

                total_before_page = len(candidates)
                verified_before_page = verified_candidate_count
                for row in rows:
                    candidate = self._build_candidate(
                        source_key=source_key,
                        source_display_name=source_display_name,
                        row=row,
                    )
                    if not candidate:
                        continue

                    dedupe_key = candidate.dedupe_key()
                    if dedupe_key in seen_keys:
                        continue
                    seen_keys.add(dedupe_key)
                    verification = (
                        self.candidate_verifier.verify_candidate(candidate, allow_enrichment=True, refresh=False)
                        if self.candidate_verifier is not None
                        else None
                    )
                    candidate.extra["retrieved_by"] = query
                    candidates.append(candidate)
                    collected_for_query += 1
                    if verification is not None and verification.is_verified:
                        verified_candidate_count += 1

                    if _should_stop_collection(
                        candidate_count=len(candidates),
                        verified_candidate_count=verified_candidate_count,
                        min_pool_candidates_per_source=config.min_pool_candidates_per_source,
                        max_candidates_per_source=config.max_candidates_per_source,
                        target_verified_candidates_per_source=config.target_verified_candidates_per_source,
                    ) or collected_for_query >= per_term_limit:
                        break

                if self.print_page_progress and self.progress_reporter is not None:
                    meta = payload.get("meta") or {}
                    self.progress_reporter.info(
                        "[CollectPage] "
                        f"source={source_key} "
                        f"term={query!r} "
                        f"page={page} "
                        f"api_rows={len(rows)} "
                        f"added={len(candidates) - total_before_page} "
                        f"term_total={collected_for_query}/{per_term_limit} "
                        f"source_total={len(candidates)}/{config.max_candidates_per_source} "
                        f"verified={verified_candidate_count}/{config.target_verified_candidates_per_source} "
                        f"min_pool={config.min_pool_candidates_per_source} "
                        f"added_verified={verified_candidate_count - verified_before_page} "
                        f"matched_total={meta.get('count', '?')}"
                    )

                page += 1
                polite_sleep(self.delay_seconds)

        return candidates

    def _build_candidate(
        self,
        *,
        source_key: str,
        source_display_name: str,
        row: dict[str, Any],
    ) -> CandidatePaper | None:
        title = (row.get("display_name") or "").strip()
        if not title:
            return None

        abstract = _reconstruct_abstract(row.get("abstract_inverted_index") or {})
        authors = []
        for authorship in row.get("authorships") or []:
            author = authorship.get("author") or {}
            name = (author.get("display_name") or "").strip()
            if name:
                authors.append(name)

        pdf_urls = _extract_pdf_urls(row)
        landing_url = ((row.get("primary_location") or {}).get("landing_page_url") or "").strip() or None
        open_access = row.get("open_access") or {}
        oa_url = (open_access.get("oa_url") or "").strip()
        if _looks_like_pdf_url(oa_url) and oa_url not in pdf_urls:
            pdf_urls.append(oa_url)

        return CandidatePaper(
            source_key=source_key,
            source_display_name=source_display_name,
            title=title,
            abstract=abstract,
            year=row.get("publication_year"),
            doi=_normalize_doi(row.get("doi")),
            landing_url=landing_url,
            pdf_urls=pdf_urls,
            authors=authors,
            is_oa=bool(open_access.get("is_oa")),
            citation_count=int(row.get("cited_by_count") or 0),
            extra={
                "openalex_id": row.get("id"),
                "oa_status": open_access.get("oa_status"),
                "oa_url": oa_url or None,
            },
        )


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    pairs: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            pairs.append((int(position), token))
    if not pairs:
        return ""
    pairs.sort(key=lambda item: item[0])
    return " ".join(token for _, token in pairs)


def _extract_pdf_urls(row: dict[str, Any]) -> list[str]:
    pdf_urls: list[str] = []
    seen: set[str] = set()

    def add_from_location(location: dict[str, Any]) -> None:
        url = (location.get("pdf_url") or "").strip()
        if not url or url in seen:
            return
        seen.add(url)
        pdf_urls.append(url)

    primary_location = row.get("primary_location") or {}
    best_oa_location = row.get("best_oa_location") or {}
    add_from_location(primary_location)
    add_from_location(best_oa_location)
    for location in row.get("locations") or []:
        add_from_location(location)
    return pdf_urls


def _normalize_doi(raw_doi: object) -> str | None:
    if not raw_doi:
        return None
    doi = str(raw_doi).strip()
    if doi.lower().startswith("https://doi.org/"):
        return doi[16:]
    if doi.lower().startswith("http://doi.org/"):
        return doi[15:]
    return doi or None


def _should_stop_collection(
    *,
    candidate_count: int,
    verified_candidate_count: int,
    min_pool_candidates_per_source: int,
    max_candidates_per_source: int,
    target_verified_candidates_per_source: int,
) -> bool:
    if candidate_count >= max_candidates_per_source:
        return True
    return (
        candidate_count >= min_pool_candidates_per_source
        and verified_candidate_count >= target_verified_candidates_per_source
    )


def _looks_like_pdf_url(url: str) -> bool:
    lowered = url.casefold()
    return lowered.endswith(".pdf") or "/pdf" in lowered or "download=true" in lowered
