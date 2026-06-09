# Data Directory

This directory is intentionally excluded from git because it contains local
paper inputs and generated experiment outputs that can become very large.

Expected local layout:

```text
data/
  papers/
    Journal_or_Conference_0/
      <paper json>
      figures/
  <run_id>/
    ...
```

Runners create and reuse `data/<run_id>/` folders based on the settings in
`configs/`. Keep API keys in the root `.env` file and commit only
`.env.example`.
