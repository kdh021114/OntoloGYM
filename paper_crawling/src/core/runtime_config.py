from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    output_root: Path = Path("paper_crawling/output")
    dry_run: bool = False
    print_progress: bool = True
    print_retrieval_page_progress: bool = False

    @classmethod
    def from_module(cls, config_module: object) -> "ExecutionConfig":
        return cls(
            output_root=_parse_path(_read_attr(config_module, "OUTPUT_ROOT", Path("paper_crawling/output"))),
            dry_run=_parse_bool(_read_attr(config_module, "DRY_RUN", False), field_name="DRY_RUN"),
            print_progress=_parse_bool(
                _read_attr(config_module, "PRINT_PROGRESS", True),
                field_name="PRINT_PROGRESS",
            ),
            print_retrieval_page_progress=_parse_bool(
                _read_attr(config_module, "PRINT_RETRIEVAL_PAGE_PROGRESS", False),
                field_name="PRINT_RETRIEVAL_PAGE_PROGRESS",
            ),
        )


@dataclass(frozen=True, slots=True)
class YearRange:
    start: int
    end: int

    @classmethod
    def from_module(cls, config_module: object) -> "YearRange":
        return cls(
            start=_parse_int(_read_attr(config_module, "START_YEAR", 2020), field_name="START_YEAR"),
            end=_parse_int(_read_attr(config_module, "END_YEAR", 2026), field_name="END_YEAR"),
        )

    def validate(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Invalid year range: start={self.start} end={self.end}")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    retrieval_terms: tuple[str, ...] = ()
    query: str = ""
    keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()
    year_range: YearRange = field(default_factory=lambda: YearRange(start=2020, end=2026))

    @classmethod
    def from_module(cls, config_module: object) -> "SearchConfig":
        return cls(
            retrieval_terms=_parse_string_tuple(_read_attr(config_module, "RETRIEVAL_TERMS", ())),
            query=_parse_string(_read_attr(config_module, "QUERY", "")),
            keywords=_parse_string_tuple(_read_attr(config_module, "KEYWORDS", ())),
            exclude_keywords=_parse_string_tuple(_read_attr(config_module, "EXCLUDE_KEYWORDS", ())),
            year_range=YearRange.from_module(config_module),
        )

    def validate(self) -> None:
        if not self.query and not self.keywords:
            raise ValueError("At least one of `QUERY` or `KEYWORDS` must be provided.")
        self.year_range.validate()

    def combined_query(self) -> str:
        if self.query and self.keywords:
            return f"{self.query} {' '.join(self.keywords)}".strip()
        if self.query:
            return self.query.strip()
        return " ".join(self.keywords).strip()

    def retrieval_queries(self) -> tuple[str, ...]:
        raw_queries = self.retrieval_terms or self.keywords or ((self.query.strip(),) if self.query.strip() else ())

        deduped: list[str] = []
        seen: set[str] = set()
        for query in raw_queries:
            lowered = query.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(query)
        return tuple(deduped)

    def phrases(self) -> tuple[str, ...]:
        phrases: list[str] = []
        if self.query.strip():
            phrases.append(self.query.strip())
        phrases.extend(keyword.strip() for keyword in self.keywords if keyword.strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            lowered = phrase.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(phrase)
        return tuple(deduped)


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    sources: tuple[str, ...] = ()
    max_candidates_per_source: int = 50
    target_verified_candidates_per_source: int = 20
    max_results_per_retrieval_term: int = 10
    top_k_per_source: int = 20

    @classmethod
    def from_module(cls, config_module: object) -> "SelectionConfig":
        return cls(
            sources=_parse_string_tuple(_read_attr(config_module, "SOURCES", ())),
            max_candidates_per_source=_parse_int(
                _read_attr(config_module, "MAX_CANDIDATES_PER_SOURCE", 50),
                field_name="MAX_CANDIDATES_PER_SOURCE",
            ),
            target_verified_candidates_per_source=_parse_int(
                _read_attr(
                    config_module,
                    "TARGET_VERIFIED_CANDIDATES_PER_SOURCE",
                    _read_attr(config_module, "TARGET_DOWNLOADABLE_CANDIDATES_PER_SOURCE", 20),
                ),
                field_name="TARGET_VERIFIED_CANDIDATES_PER_SOURCE",
            ),
            max_results_per_retrieval_term=_parse_int(
                _read_attr(config_module, "MAX_RESULTS_PER_RETRIEVAL_TERM", 10),
                field_name="MAX_RESULTS_PER_RETRIEVAL_TERM",
            ),
            top_k_per_source=_parse_int(
                _read_attr(config_module, "TOP_K_PER_SOURCE", _read_attr(config_module, "TOP_K", 20)),
                field_name="TOP_K_PER_SOURCE",
            ),
        )

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("`SOURCES` must contain at least one source key.")
        if self.max_candidates_per_source <= 0:
            raise ValueError("`MAX_CANDIDATES_PER_SOURCE` must be positive.")
        if self.target_verified_candidates_per_source <= 0:
            raise ValueError("`TARGET_VERIFIED_CANDIDATES_PER_SOURCE` must be positive.")
        if self.target_verified_candidates_per_source > self.max_candidates_per_source:
            raise ValueError(
                "`TARGET_VERIFIED_CANDIDATES_PER_SOURCE` must be less than or equal to "
                "`MAX_CANDIDATES_PER_SOURCE`."
            )
        derived_min_pool = max(self.target_verified_candidates_per_source * 2, 80)
        if derived_min_pool > self.max_candidates_per_source:
            raise ValueError(
                "Derived minimum pool size "
                f"({derived_min_pool}) exceeds `MAX_CANDIDATES_PER_SOURCE` "
                f"({self.max_candidates_per_source}). Increase `MAX_CANDIDATES_PER_SOURCE`."
            )
        if self.max_results_per_retrieval_term <= 0:
            raise ValueError("`MAX_RESULTS_PER_RETRIEVAL_TERM` must be positive.")
        if self.top_k_per_source <= 0:
            raise ValueError("`TOP_K_PER_SOURCE` must be positive.")


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    enabled: bool = True
    open_access_only: bool = True

    @classmethod
    def from_module(cls, config_module: object) -> "DownloadConfig":
        return cls(
            enabled=_parse_bool(_read_attr(config_module, "DOWNLOAD_PDFS", True), field_name="DOWNLOAD_PDFS"),
            open_access_only=_parse_bool(
                _read_attr(config_module, "DOWNLOAD_OPEN_ACCESS_ONLY", True),
                field_name="DOWNLOAD_OPEN_ACCESS_ONLY",
            ),
        )


@dataclass(frozen=True, slots=True)
class RequestConfig:
    delay_seconds: float = 1.0
    email: str | None = None

    @classmethod
    def from_module(cls, config_module: object) -> "RequestConfig":
        return cls(
            delay_seconds=_parse_float(
                _read_attr(config_module, "REQUEST_DELAY_SECONDS", 1.0),
                field_name="REQUEST_DELAY_SECONDS",
            ),
            email=_parse_optional_string(_read_attr(config_module, "EMAIL", None)),
        )

    def validate(self) -> None:
        if self.delay_seconds < 0:
            raise ValueError("`REQUEST_DELAY_SECONDS` must be non-negative.")


@dataclass(frozen=True, slots=True)
class RankingConfig:
    bm25_weight: float = 0.45
    phrase_weight: float = 0.15
    citation_weight: float = 0.05
    semantic_weight: float = 0.35
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_batch_size: int = 32

    @classmethod
    def from_module(cls, config_module: object) -> "RankingConfig":
        semantic_weight_value = _read_attr(
            config_module,
            "SEMANTIC_WEIGHT",
            _read_attr(config_module, "COSINE_WEIGHT", 0.0),
        )
        return cls(
            bm25_weight=_parse_float(_read_attr(config_module, "BM25_WEIGHT", 0.45), field_name="BM25_WEIGHT"),
            phrase_weight=_parse_float(
                _read_attr(config_module, "PHRASE_WEIGHT", 0.15),
                field_name="PHRASE_WEIGHT",
            ),
            citation_weight=_parse_float(
                _read_attr(config_module, "CITATION_WEIGHT", 0.05),
                field_name="CITATION_WEIGHT",
            ),
            semantic_weight=_parse_float(
                semantic_weight_value,
                field_name="SEMANTIC_WEIGHT",
            ),
            semantic_model_name=_parse_string(
                _read_attr(config_module, "SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            semantic_batch_size=_parse_int(
                _read_attr(config_module, "SEMANTIC_BATCH_SIZE", 32),
                field_name="SEMANTIC_BATCH_SIZE",
            ),
        )

    def validate(self) -> None:
        weights = [self.bm25_weight, self.phrase_weight, self.citation_weight, self.semantic_weight]
        if any(weight < 0 for weight in weights):
            raise ValueError("Ranking weights must be non-negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one ranking weight must be positive.")
        if self.semantic_weight > 0 and not self.semantic_model_name.strip():
            raise ValueError("`SEMANTIC_MODEL_NAME` must not be blank when semantic ranking is enabled.")
        if self.semantic_weight > 0 and self.semantic_batch_size <= 0:
            raise ValueError("`SEMANTIC_BATCH_SIZE` must be positive when semantic ranking is enabled.")

    @property
    def cosine_weight(self) -> float:
        return self.semantic_weight


@dataclass(frozen=True, slots=True)
class PaperCrawlingConfig:
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    @classmethod
    def from_module(cls, config_module: object) -> "PaperCrawlingConfig":
        return cls(
            execution=ExecutionConfig.from_module(config_module),
            search=SearchConfig.from_module(config_module),
            selection=SelectionConfig.from_module(config_module),
            download=DownloadConfig.from_module(config_module),
            request=RequestConfig.from_module(config_module),
            ranking=RankingConfig.from_module(config_module),
        )

    def validate(self) -> None:
        self.search.validate()
        self.selection.validate()
        self.request.validate()
        self.ranking.validate()

    @property
    def output_root(self) -> Path:
        return self.execution.output_root

    @property
    def dry_run(self) -> bool:
        return self.execution.dry_run

    @property
    def print_progress(self) -> bool:
        return self.execution.print_progress

    @property
    def print_retrieval_page_progress(self) -> bool:
        return self.execution.print_retrieval_page_progress

    @property
    def query(self) -> str:
        return self.search.query

    @property
    def retrieval_terms(self) -> tuple[str, ...]:
        return self.search.retrieval_terms

    @property
    def keywords(self) -> tuple[str, ...]:
        return self.search.keywords

    @property
    def exclude_keywords(self) -> tuple[str, ...]:
        return self.search.exclude_keywords

    @property
    def year_range(self) -> YearRange:
        return self.search.year_range

    @property
    def sources(self) -> tuple[str, ...]:
        return self.selection.sources

    @property
    def max_candidates_per_source(self) -> int:
        return self.selection.max_candidates_per_source

    @property
    def min_pool_candidates_per_source(self) -> int:
        return max(self.target_verified_candidates_per_source * 2, 80)

    @property
    def target_verified_candidates_per_source(self) -> int:
        return self.selection.target_verified_candidates_per_source

    @property
    def max_results_per_retrieval_term(self) -> int:
        return self.selection.max_results_per_retrieval_term

    @property
    def top_k_per_source(self) -> int:
        return self.selection.top_k_per_source

    @property
    def top_k(self) -> int:
        return self.selection.top_k_per_source

    @property
    def download_pdfs(self) -> bool:
        return self.download.enabled

    @property
    def download_open_access_only(self) -> bool:
        return self.download.open_access_only

    @property
    def request_delay_seconds(self) -> float:
        return self.request.delay_seconds

    @property
    def email(self) -> str | None:
        return self.request.email

    def combined_query(self) -> str:
        return self.search.combined_query()

    def retrieval_queries(self) -> tuple[str, ...]:
        return self.search.retrieval_queries()

    def phrases(self) -> tuple[str, ...]:
        return self.search.phrases()


def load_runtime_config(config_module: object) -> PaperCrawlingConfig:
    config = PaperCrawlingConfig.from_module(config_module)
    config.validate()
    return config


def _read_attr(config_module: object, name: str, default: object) -> object:
    return getattr(config_module, name, default)


def _parse_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _parse_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_optional_string(value: object) -> str | None:
    text = _parse_string(value)
    return text or None


def _parse_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a string or list of strings.")

    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid boolean value for `{field_name}`: {value!r}")


def _parse_int(value: object, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for `{field_name}`: {value!r}") from exc


def _parse_float(value: object, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value for `{field_name}`: {value!r}") from exc
