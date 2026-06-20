from __future__ import annotations

import httpx

from ..core.interfaces import CandidateEnricher
from ..core.models import CandidatePaper
from ..core.verification import CandidateVerificationStatus
from .pdf_preflight_verifier import PdfPreflightVerifier


class CandidateVerificationService:
    def __init__(
        self,
        pdf_preflight_verifier: PdfPreflightVerifier,
        *,
        candidate_enricher: CandidateEnricher | None = None,
        open_access_only: bool = True,
    ) -> None:
        self._pdf_preflight_verifier = pdf_preflight_verifier
        self._candidate_enricher = candidate_enricher
        self._open_access_only = open_access_only

    def verify_candidate(
        self,
        candidate: CandidatePaper,
        *,
        allow_enrichment: bool = True,
        refresh: bool = False,
    ) -> CandidateVerificationStatus:
        if not refresh and candidate.extra.get("verification_checked") is True:
            return _status_from_candidate(candidate)

        if self._open_access_only and not candidate.is_oa:
            status = CandidateVerificationStatus(is_verified=False, reason="closed_access")
            self._apply_status(candidate, status)
            return status

        status = self._pdf_preflight_verifier.verify_urls(candidate.all_pdf_urls(), refresh=refresh)
        if status.is_verified:
            self._apply_status(candidate, status)
            return status

        if allow_enrichment and self._should_enrich(candidate):
            self._enrich_candidate(candidate)
            status = self._pdf_preflight_verifier.verify_urls(candidate.all_pdf_urls(), refresh=refresh)

        self._apply_status(candidate, status)
        return status

    def _should_enrich(self, candidate: CandidatePaper) -> bool:
        return self._candidate_enricher is not None and bool(candidate.doi) and candidate.extra.get("crossref_enriched") is not True

    def _enrich_candidate(self, candidate: CandidatePaper) -> None:
        if self._candidate_enricher is None:
            return
        try:
            self._candidate_enricher.enrich_candidate(candidate)
            candidate.extra["crossref_enriched"] = True
        except httpx.HTTPError as exc:
            candidate.extra["crossref_enriched"] = False
            candidate.extra["crossref_error"] = str(exc)

    def _apply_status(self, candidate: CandidatePaper, status: CandidateVerificationStatus) -> None:
        candidate.extra["verification_checked"] = True
        candidate.extra["verified_downloadable"] = status.is_verified
        candidate.extra["verification_reason"] = status.reason
        candidate.extra["verified_pdf_url"] = status.url
        candidate.extra["verified_pdf_final_url"] = status.final_url
        candidate.extra["verification_status_code"] = status.status_code
        candidate.extra["verification_content_type"] = status.content_type
        candidate.extra["verification_error_type"] = status.error_type
        candidate.extra["verification_message"] = status.message


def _status_from_candidate(candidate: CandidatePaper) -> CandidateVerificationStatus:
    return CandidateVerificationStatus(
        is_verified=bool(candidate.extra.get("verified_downloadable")),
        reason=str(candidate.extra.get("verification_reason") or "not_checked"),
        url=_optional_str(candidate.extra.get("verified_pdf_url")),
        final_url=_optional_str(candidate.extra.get("verified_pdf_final_url")),
        status_code=_optional_int(candidate.extra.get("verification_status_code")),
        content_type=_optional_str(candidate.extra.get("verification_content_type")),
        error_type=_optional_str(candidate.extra.get("verification_error_type")),
        message=_optional_str(candidate.extra.get("verification_message")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
