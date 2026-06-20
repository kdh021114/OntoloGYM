from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx

from ..common.utils import append_jsonl, ensure_dir, write_json
from ..core.interfaces import CandidateVerifier
from ..core.interfaces import ProgressReporter
from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig
from ..processing.candidate_processing import CandidateDeduplicator, CandidateFilter
from ..processing.download_planner import DownloadCandidatePlanner
from ..processing.candidate_verifier import CandidateVerificationService
from ..processing.downloader import PdfDownloader
from ..processing.pdf_preflight_verifier import PdfPreflightVerifier
from ..rankers.factory import build_default_ranker
from ..sources.catalog import SourceCatalog, build_default_source_catalog
from ..sources.factory import build_default_source_factory
from ..sources.providers.crossref import CrossrefClient
from ..sources.providers.openalex import OpenAlexClient
from ..sources.providers.site_pdf_discovery import SitePdfDiscoveryClient


class PaperCrawlingPipeline:
    def __init__(
        self,
        output_root: Path,
        *,
        source_catalog: SourceCatalog | None = None,
        candidate_filter: CandidateFilter | None = None,
        candidate_deduplicator: CandidateDeduplicator | None = None,
        download_candidate_planner: DownloadCandidatePlanner | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.output_root = output_root
        self.source_catalog = source_catalog or build_default_source_catalog()
        self.candidate_filter = candidate_filter or CandidateFilter()
        self.candidate_deduplicator = candidate_deduplicator or CandidateDeduplicator()
        self.download_candidate_planner = download_candidate_planner or DownloadCandidatePlanner()
        self.progress_reporter = progress_reporter

    def run(self, config: PaperCrawlingConfig, dry_run: bool = False) -> dict[str, object]:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(self.output_root / run_id)
        total_download_target = config.top_k_per_source * len(config.sources)
        self._log(
            "[Run] "
            f"run_id={run_id} "
            f"sources={len(config.sources)} "
            f"retrieval_terms={len(config.retrieval_queries())} "
            f"top_k_per_source={config.top_k_per_source} "
            f"total_download_target={total_download_target} "
            f"download_mode={'dry_run' if dry_run or not config.download_pdfs else 'download'}"
        )

        headers = {
            "User-Agent": _build_user_agent(config.email),
            "Accept": "application/json, text/html, application/pdf;q=0.9, */*;q=0.8",
        }

        with httpx.Client(headers=headers, timeout=45.0) as http_client:
            candidate_enricher = CrossrefClient(
                http_client=http_client,
                delay_seconds=config.request_delay_seconds,
            )
            candidate_verifier: CandidateVerifier = CandidateVerificationService(
                PdfPreflightVerifier(
                    http_client=http_client,
                    delay_seconds=config.request_delay_seconds,
                ),
                candidate_enricher=candidate_enricher,
                open_access_only=config.download_open_access_only,
            )
            openalex_client = OpenAlexClient(
                http_client=http_client,
                delay_seconds=config.request_delay_seconds,
                candidate_verifier=candidate_verifier,
                progress_reporter=self.progress_reporter,
                print_page_progress=config.print_retrieval_page_progress,
            )
            site_pdf_discovery_client = SitePdfDiscoveryClient(
                http_client=http_client,
                delay_seconds=config.request_delay_seconds,
            )
            downloader = PdfDownloader(http_client=http_client, delay_seconds=config.request_delay_seconds)
            source_factory = build_default_source_factory(
                openalex_client=openalex_client,
                site_pdf_discovery_client=site_pdf_discovery_client,
            )
            ranker = build_default_ranker(config)

            raw_candidates: list[CandidatePaper] = []
            source_definitions = self.source_catalog.resolve_many(config.sources)
            sources = source_factory.create_many(source_definitions)
            source_order = [source.key for source in sources]
            source_counts: dict[str, int] = {}
            source_verified_counts: dict[str, int] = {}

            for source in sources:
                self._log(
                    "[CollectStart] "
                    f"source={source.key} "
                    f"display_name={getattr(source, 'display_name', source.key)!r} "
                    f"min_pool={config.min_pool_candidates_per_source} "
                    f"max_candidates={config.max_candidates_per_source} "
                    f"target_verified={config.target_verified_candidates_per_source}"
                )
                candidates = source.collect(config)
                source_counts[source.key] = len(candidates)
                source_verified_counts[source.key] = sum(
                    1 for candidate in candidates if candidate.extra.get("verified_downloadable") is True
                )
                raw_candidates.extend(candidates)
                self._log(
                    "[CollectDone] "
                    f"source={source.key} "
                    f"candidates={source_counts[source.key]} "
                    f"verified={source_verified_counts[source.key]}"
                )

            filtered_candidates = self.candidate_filter.apply(raw_candidates, config)
            merged_candidates = self.candidate_deduplicator.deduplicate(filtered_candidates)
            merged_candidates_by_source = _group_candidates_by_source(merged_candidates, source_order)
            ranked_candidates: list[CandidatePaper] = []
            verified_ranked_candidates_by_source: dict[str, list[CandidatePaper]] = {}
            source_verified_ranked_counts: dict[str, int] = {}
            source_selected_counts: dict[str, int] = {}
            selected_candidates: list[CandidatePaper] = []

            for source_key in source_order:
                source_merged_candidates = merged_candidates_by_source.get(source_key, [])
                source_ranked_candidates = ranker.rank(source_merged_candidates, config)
                for source_rank, candidate in enumerate(source_ranked_candidates, start=1):
                    candidate.extra["source_rank"] = source_rank
                ranked_candidates.extend(source_ranked_candidates)

                source_verified_queue = self.download_candidate_planner.build_queue(source_ranked_candidates, config)
                verified_ranked_candidates_by_source[source_key] = source_verified_queue
                source_verified_ranked_counts[source_key] = len(source_verified_queue)
                source_selected_counts[source_key] = min(len(source_verified_queue), config.top_k_per_source)
                selected_candidates.extend(source_verified_queue[: config.top_k_per_source])
                self._log(
                    "[RankSource] "
                    f"source={source_key} "
                    f"ranked={len(source_ranked_candidates)} "
                    f"verified_ranked={source_verified_ranked_counts[source_key]} "
                    f"selected_top_k={source_selected_counts[source_key]}"
                )

            verified_ranked_candidates = [
                candidate
                for source_key in source_order
                for candidate in verified_ranked_candidates_by_source.get(source_key, [])
            ]
            download_queue_count = (
                sum(source_selected_counts.values())
                if dry_run or not config.download_pdfs
                else len(verified_ranked_candidates)
            )
            self._log(
                "[Rank] "
                f"raw={len(raw_candidates)} "
                f"filtered={len(filtered_candidates)} "
                f"deduplicated={len(merged_candidates)} "
                f"ranked={len(ranked_candidates)} "
                f"verified_ranked={len(verified_ranked_candidates)} "
                f"selected_top_k={len(selected_candidates)} "
                f"download_queue={download_queue_count}"
            )

            download_records: list[dict[str, object]] = []
            downloaded_count = 0
            attempted_count = 0
            status_counts: dict[str, int] = {}
            source_downloaded_counts: dict[str, int] = {source_key: 0 for source_key in source_order}
            source_attempted_counts: dict[str, int] = {source_key: 0 for source_key in source_order}

            for source_key in source_order:
                source_queue = verified_ranked_candidates_by_source.get(source_key, [])
                if dry_run or not config.download_pdfs:
                    source_queue = source_queue[: config.top_k_per_source]

                for queue_index, candidate in enumerate(source_queue, start=1):
                    if dry_run or not config.download_pdfs:
                        if source_attempted_counts[source_key] >= config.top_k_per_source:
                            break
                    elif source_downloaded_counts[source_key] >= config.top_k_per_source:
                        break

                    verification = candidate_verifier.verify_candidate(candidate, allow_enrichment=True, refresh=True)
                    if not verification.is_verified:
                        record = downloader.write_metadata_only(candidate=candidate, run_dir=run_dir)
                        record.update(
                            {
                                "status": "skipped_preflight_failed",
                                "url": verification.url,
                                "final_url": verification.final_url,
                                "verification_reason": verification.reason,
                                "verification_status_code": verification.status_code,
                                "verification_content_type": verification.content_type,
                            }
                        )
                    elif dry_run or not config.download_pdfs:
                        record = downloader.write_metadata_only(candidate=candidate, run_dir=run_dir)
                    else:
                        record = downloader.download_candidate(candidate=candidate, run_dir=run_dir, config=config)
                    record["rank"] = candidate.extra.get("source_rank")
                    record["source_rank"] = candidate.extra.get("source_rank")
                    record["download_queue_index"] = queue_index
                    download_records.append(record)
                    attempted_count += 1
                    source_attempted_counts[source_key] += 1
                    status = str(record["status"])
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status == "downloaded":
                        downloaded_count += 1
                        source_downloaded_counts[source_key] += 1
                    self._log_download_progress(
                        record=record,
                        attempted_count=attempted_count,
                        downloaded_count=downloaded_count,
                        total_target_count=total_download_target,
                        source_attempted_count=source_attempted_counts[source_key],
                        source_downloaded_count=source_downloaded_counts[source_key],
                        source_target_count=config.top_k_per_source,
                    )

            append_jsonl(run_dir / "ranked_candidates.jsonl", [candidate.to_dict() for candidate in ranked_candidates])
            append_jsonl(run_dir / "download_manifest.jsonl", download_records)

            summary = {
                "run_id": run_id,
                "retrieval_terms": config.retrieval_queries(),
                "query": config.query,
                "keywords": config.keywords,
                "sources": config.sources,
                "candidate_count": len(ranked_candidates),
                "verified_candidate_count": len(verified_ranked_candidates),
                "selected_count": len(selected_candidates),
                "top_k_per_source": config.top_k_per_source,
                "download_target_count": total_download_target,
                "download_attempted_count": attempted_count,
                "downloaded_count": downloaded_count,
                "download_mode": "dry_run" if dry_run or not config.download_pdfs else "download",
                "output_dir": str(run_dir),
                "source_counts": source_counts,
                "source_verified_counts": source_verified_counts,
                "source_verified_ranked_counts": source_verified_ranked_counts,
                "source_selected_counts": source_selected_counts,
                "source_attempted_counts": source_attempted_counts,
                "source_downloaded_counts": source_downloaded_counts,
                "download_status_counts": status_counts,
            }
            write_json(run_dir / "run_summary.json", summary)
            self._log(
                "[RunDone] "
                f"run_id={run_id} "
                f"candidates={summary['candidate_count']} "
                f"verified={summary['verified_candidate_count']} "
                f"selected_top_k={summary['selected_count']} "
                f"attempted={summary['download_attempted_count']} "
                f"downloaded={summary['downloaded_count']} "
                f"output={summary['output_dir']}"
            )
            return summary

    def _log_download_progress(
        self,
        *,
        record: dict[str, object],
        attempted_count: int,
        downloaded_count: int,
        total_target_count: int,
        source_attempted_count: int,
        source_downloaded_count: int,
        source_target_count: int,
    ) -> None:
        title = _truncate_text(str(record.get("title") or ""), limit=80)
        self._log(
            "[Download] "
            f"queue_index={record.get('download_queue_index')} "
            f"source_rank={record.get('source_rank')} "
            f"status={record.get('status')} "
            f"source_downloaded={source_downloaded_count}/{source_target_count} "
            f"source_attempted={source_attempted_count} "
            f"downloaded={downloaded_count}/{total_target_count} "
            f"attempted={attempted_count} "
            f"source={record.get('source')} "
            f"title={title!r}"
        )

    def _log(self, message: str) -> None:
        if self.progress_reporter is None:
            return
        self.progress_reporter.info(message)


def _build_user_agent(email: str | None) -> str:
    if email:
        return f"senior-thesis-paper-crawler/0.1 ({email})"
    return "senior-thesis-paper-crawler/0.1"


def _group_candidates_by_source(
    candidates: list[CandidatePaper],
    source_order: list[str],
) -> dict[str, list[CandidatePaper]]:
    grouped: dict[str, list[CandidatePaper]] = {source_key: [] for source_key in source_order}
    for candidate in candidates:
        grouped.setdefault(candidate.source_key, []).append(candidate)
    return grouped


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
