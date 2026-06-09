"""Run-folder helpers for grouping connected OntoloGYM pipeline outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PIPELINE_ORDER = [
    "qa_extractor",
    "ontogen",
    "relation_augmentation",
    "qa_evaluation",
    "kg_refinement",
    "kg_visualization",
]

ACTIVE_RUN_FILE = "_active_run.txt"
MANIFEST_FILE = "run_manifest.json"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def _active_path(runs_dir: Path) -> Path:
    return runs_dir / ACTIVE_RUN_FILE


def _manifest_path(run_root: Path) -> Path:
    return run_root / MANIFEST_FILE


def _read_active_run_id(runs_dir: Path) -> str | None:
    path = _active_path(runs_dir)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _write_active_run_id(runs_dir: Path, run_id: str) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    _active_path(runs_dir).write_text(run_id + "\n", encoding="utf-8")


def initialize_manifest(
    run_root: Path,
    *,
    project_root: Path,
    paper_dir: Path,
    pipeline_order: list[str] | tuple[str, ...],
) -> None:
    manifest_path = _manifest_path(run_root)
    if manifest_path.exists():
        return

    manifest = {
        "run_id": run_root.name,
        "created_at": _utc_now(),
        "project_root": str(project_root),
        "input": {
            "paper_dir": str(paper_dir),
        },
        "pipeline_order": list(pipeline_order),
        "pipelines": {},
    }
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_run_root(
    *,
    project_root: Path,
    runs_dir: Path,
    paper_dir: Path,
    output_mode: str = "run",
    explicit_run_id: str | None = None,
    create_new: bool = False,
    run_id_prefix: str = "run",
    pipeline_order: list[str] | tuple[str, ...] = DEFAULT_PIPELINE_ORDER,
) -> Path:
    """Return the output root for this execution and keep an active run pointer."""
    if output_mode != "run":
        return project_root / "data"

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    env_run_id = os.getenv("ONTOLOGYM_RUN_ID", "").strip()
    env_create_new = _truthy(os.getenv("ONTOLOGYM_CREATE_RUN"))

    if explicit_run_id:
        run_id = explicit_run_id
    elif env_run_id:
        run_id = env_run_id
    elif create_new or env_create_new:
        run_id = make_run_id(run_id_prefix)
    else:
        run_id = _read_active_run_id(runs_dir) or make_run_id(run_id_prefix)

    _write_active_run_id(runs_dir, run_id)
    run_root = runs_dir / run_id
    initialize_manifest(
        run_root,
        project_root=project_root,
        paper_dir=paper_dir,
        pipeline_order=pipeline_order,
    )
    return run_root


def create_new_run(
    *,
    project_root: Path,
    runs_dir: Path,
    paper_dir: Path,
    run_id_prefix: str = "run",
    pipeline_order: list[str] | tuple[str, ...] = DEFAULT_PIPELINE_ORDER,
) -> Path:
    run_id = make_run_id(run_id_prefix)
    _write_active_run_id(Path(runs_dir), run_id)
    run_root = Path(runs_dir) / run_id
    initialize_manifest(
        run_root,
        project_root=project_root,
        paper_dir=paper_dir,
        pipeline_order=pipeline_order,
    )
    return run_root


def record_pipeline_run(
    run_root: str | Path,
    pipeline_name: str,
    *,
    status: str,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Update run_manifest.json with the latest state of one pipeline."""
    run_root = Path(run_root)
    manifest_path = _manifest_path(run_root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "run_id": run_root.name,
            "created_at": _utc_now(),
            "pipeline_order": DEFAULT_PIPELINE_ORDER,
            "pipelines": {},
        }

    record = {
        "status": status,
        "updated_at": _utc_now(),
    }
    if inputs is not None:
        record["inputs"] = inputs
    if outputs is not None:
        record["outputs"] = outputs
    if extra:
        record.update(extra)

    manifest.setdefault("pipelines", {})[pipeline_name] = record
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
