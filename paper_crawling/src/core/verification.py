from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CandidateVerificationStatus:
    is_verified: bool
    reason: str
    url: str | None = None
    final_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    error_type: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
