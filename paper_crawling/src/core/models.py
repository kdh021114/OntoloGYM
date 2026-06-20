from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

@dataclass(slots=True)
class CandidatePaper:
    source_key: str
    source_display_name: str
    title: str
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    landing_url: str | None = None
    pdf_urls: list[str] = field(default_factory=list)
    crossref_pdf_urls: list[str] = field(default_factory=list)
    license_urls: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    is_oa: bool = False
    citation_count: int = 0
    ranking_score: float = 0.0
    ranking_breakdown: dict[str, float] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    def combined_text(self) -> str:
        return " ".join(part for part in [self.title, self.abstract] if part).strip()

    def slug(self) -> str:
        year_part = str(self.year) if self.year else "unknown"
        base = _slugify(self.title) or _slugify(self.doi or "") or "paper"
        return f"{year_part}_{base}"[:180]

    def dedupe_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.casefold()}"
        if self.landing_url:
            return f"url:{self.landing_url.casefold()}"
        return f"title:{self.title.casefold()}|year:{self.year or 'unknown'}|source:{self.source_key}"

    def all_pdf_urls(self) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for url in [*self.pdf_urls, *self.crossref_pdf_urls]:
            if not url:
                continue
            key = url.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
        return merged

    def metadata_path(self, source_root: Path) -> Path:
        return source_root / "_metadata" / f"{self.slug()}.json"

    def pdf_path(self, source_root: Path) -> Path:
        return source_root / f"{self.slug()}.pdf"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _slugify(text: str) -> str:
    safe_chars: list[str] = []
    previous_dash = False
    for char in text.lower():
        if char.isalnum():
            safe_chars.append(char)
            previous_dash = False
            continue
        if previous_dash:
            continue
        safe_chars.append("-")
        previous_dash = True
    return "".join(safe_chars).strip("-")
