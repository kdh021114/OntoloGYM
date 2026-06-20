from __future__ import annotations

from pathlib import Path

import httpx

from ..common.utils import ensure_dir, polite_sleep, write_json
from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig


class PdfDownloader:
    def __init__(self, http_client: httpx.Client, delay_seconds: float = 1.0) -> None:
        self.http_client = http_client
        self.delay_seconds = delay_seconds

    def write_metadata_only(self, candidate: CandidatePaper, run_dir: Path) -> dict[str, object]:
        source_root = ensure_dir(run_dir / candidate.source_key)
        metadata_path = candidate.metadata_path(source_root)
        write_json(metadata_path, candidate.to_dict())
        return {
            "source": candidate.source_key,
            "title": candidate.title,
            "doi": candidate.doi,
            "status": "metadata_only",
            "pdf_path": None,
            "metadata_path": str(metadata_path),
            "url": None,
        }

    def download_candidate(
        self,
        candidate: CandidatePaper,
        run_dir: Path,
        config: PaperCrawlingConfig,
    ) -> dict[str, object]:
        source_root = ensure_dir(run_dir / candidate.source_key)
        metadata_path = candidate.metadata_path(source_root)
        pdf_path = candidate.pdf_path(source_root)
        write_json(metadata_path, candidate.to_dict())
        attempts: list[dict[str, object]] = []

        if config.download_open_access_only and not candidate.is_oa:
            return {
                "source": candidate.source_key,
                "title": candidate.title,
                "doi": candidate.doi,
                "status": "skipped_non_oa",
                "pdf_path": None,
                "metadata_path": str(metadata_path),
                "url": None,
                "attempts": attempts,
            }

        for url in _iter_download_urls(candidate):
            try:
                response = self.http_client.get(url, follow_redirects=True)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                attempts.append(
                    {
                        "url": url,
                        "reason": "http_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                polite_sleep(self.delay_seconds)
                continue

            content_type = (response.headers.get("content-type") or "").casefold()
            content = response.content
            looks_like_pdf = "application/pdf" in content_type or content.startswith(b"%PDF")
            if not looks_like_pdf:
                attempts.append(
                    {
                        "url": url,
                        "final_url": str(response.url),
                        "status_code": response.status_code,
                        "content_type": content_type,
                        "reason": "non_pdf_response",
                    }
                )
                polite_sleep(self.delay_seconds)
                continue

            ensure_dir(pdf_path.parent)
            pdf_path.write_bytes(content)
            polite_sleep(self.delay_seconds)
            return {
                "source": candidate.source_key,
                "title": candidate.title,
                "doi": candidate.doi,
                "status": "downloaded",
                "pdf_path": str(pdf_path),
                "metadata_path": str(metadata_path),
                "url": url,
                "final_url": str(response.url),
                "attempts": attempts,
            }

        return {
            "source": candidate.source_key,
            "title": candidate.title,
            "doi": candidate.doi,
            "status": "download_failed",
            "pdf_path": None,
            "metadata_path": str(metadata_path),
            "url": None,
            "attempts": attempts,
        }


def _iter_download_urls(candidate: CandidatePaper) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for value in [
        candidate.extra.get("verified_pdf_url"),
        candidate.extra.get("verified_pdf_final_url"),
        *candidate.all_pdf_urls(),
    ]:
        if value is None:
            continue
        url = str(value).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return urls
