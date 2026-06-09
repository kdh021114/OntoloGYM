# Experiments

This directory contains thesis-specific reproduction scripts, model-comparison
runners, and ablation workflows. They are kept out of the repository root so
the main OntoloGYM entrypoints stay focused on the reusable pipeline.

Most scripts here assume local `data/<run_id>/` artifacts that are intentionally
not committed to git. Run them from the repository root, for example:

```bash
python experiments/run_fair_model_comparison.py
```

The default public pipeline remains in the root-level `run_*.py` files.
