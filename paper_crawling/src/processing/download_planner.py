from __future__ import annotations

from ..core.models import CandidatePaper
from ..core.runtime_config import PaperCrawlingConfig


class DownloadCandidatePlanner:
    def build_queue(
        self,
        ranked_candidates: list[CandidatePaper],
        config: PaperCrawlingConfig,
    ) -> list[CandidatePaper]:
        candidates = list(ranked_candidates)
        if config.download_open_access_only:
            candidates = [candidate for candidate in candidates if candidate.is_oa]
        return [candidate for candidate in candidates if candidate.extra.get("verified_downloadable") is True]
