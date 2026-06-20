from __future__ import annotations

from ..core.interfaces import ProgressReporter
from ..core.runtime_config import PaperCrawlingConfig


class ConsoleProgressReporter:
    def info(self, message: str) -> None:
        print(message, flush=True)


class NullProgressReporter:
    def info(self, message: str) -> None:
        return None


def build_progress_reporter(config: PaperCrawlingConfig) -> ProgressReporter:
    if config.print_progress:
        return ConsoleProgressReporter()
    return NullProgressReporter()
