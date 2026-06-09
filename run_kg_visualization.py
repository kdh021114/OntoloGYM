"""Render local KG outputs to an inspectable HTML file."""

from __future__ import annotations

import json

from kg_visualization.visualize import run_visualization


if __name__ == "__main__":
    print(json.dumps(run_visualization(), ensure_ascii=False, indent=2))
