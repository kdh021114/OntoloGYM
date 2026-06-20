from __future__ import annotations

from ... import config as user_config

from ..core.runtime_config import load_runtime_config
from .pipeline import PaperCrawlingPipeline
from .progress import build_progress_reporter


def main() -> None:
    config = load_runtime_config(user_config)
    pipeline = PaperCrawlingPipeline(
        output_root=config.output_root,
        progress_reporter=build_progress_reporter(config),
    )
    result = pipeline.run(config=config, dry_run=config.dry_run)

    print(
        "Finished. "
        f"run_id={result['run_id']} "
        f"top_k_per_source={result['top_k_per_source']} "
        f"candidates={result['candidate_count']} "
        f"verified={result['verified_candidate_count']} "
        f"selected_total={result['selected_count']} "
        f"target_total={result['download_target_count']} "
        f"attempted={result['download_attempted_count']} "
        f"downloaded={result['downloaded_count']} "
        f"output={result['output_dir']}"
    )
