from __future__ import annotations

from collections.abc import Iterable

import httpx

from ..common.utils import polite_sleep
from ..core.verification import CandidateVerificationStatus


class PdfPreflightVerifier:
    def __init__(
        self,
        http_client: httpx.Client,
        *,
        delay_seconds: float = 1.0,
        byte_sample_size: int = 2048,
    ) -> None:
        self.http_client = http_client
        self.delay_seconds = delay_seconds
        self.byte_sample_size = byte_sample_size
        self._cache: dict[str, CandidateVerificationStatus] = {}

    def verify_urls(
        self,
        urls: Iterable[str],
        *,
        refresh: bool = False,
    ) -> CandidateVerificationStatus:
        last_result = CandidateVerificationStatus(is_verified=False, reason="no_pdf_url")
        for url in urls:
            normalized = url.strip()
            if not normalized:
                continue

            if not refresh and normalized in self._cache:
                result = self._cache[normalized]
            else:
                result = self._verify_single_url(normalized)
                self._cache[normalized] = result

            if result.is_verified:
                return result
            last_result = result
        return last_result

    def _verify_single_url(self, url: str) -> CandidateVerificationStatus:
        headers = {"Range": f"bytes=0-{self.byte_sample_size - 1}"}
        prefix = b""
        try:
            with self.http_client.stream("GET", url, follow_redirects=True, headers=headers) as response:
                status_code = response.status_code
                response.raise_for_status()
                content_type = (response.headers.get("content-type") or "").casefold() or None

                for chunk in response.iter_bytes():
                    prefix += chunk
                    if len(prefix) >= self.byte_sample_size:
                        break

                is_pdf = bool(content_type and "application/pdf" in content_type) or prefix.startswith(b"%PDF")
                reason = "verified_pdf" if is_pdf else "non_pdf_response"
                return CandidateVerificationStatus(
                    is_verified=is_pdf,
                    reason=reason,
                    url=url,
                    final_url=str(response.url),
                    status_code=status_code,
                    content_type=content_type,
                )
        except httpx.HTTPError as exc:
            status_code = None
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
            return CandidateVerificationStatus(
                is_verified=False,
                reason="http_error",
                url=url,
                status_code=status_code,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        finally:
            polite_sleep(self.delay_seconds)
