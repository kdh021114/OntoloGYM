from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    os.chdir(project_root)


def main() -> None:
    _prepare_environment()

    from paper_crawling.src.app.cli import main as package_main

    package_main()


if __name__ == "__main__":
    main()
