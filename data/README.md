# data/

Datasets are **not** committed.

- `raw/` — put real Kaggle images here (`raw/pcb/`, `raw/concrete/`), e.g. via
  `python scripts/download_data.py`. Gitignored except `.gitkeep`.
- When `raw/` is empty the pipeline uses deterministic synthetic stand-ins
  (`cvlab.datasets.synthetic_pcb` / `synthetic_concrete`) that carry an exact
  ground-truth edge mask, so every experiment and metric is still reproducible.
- `samples/` — reserved for a handful of tiny committed example images (none by
  default; the synthetic generators cover the demos).

Sources:
[PCB defects](https://www.kaggle.com/datasets/akhatova/pcb-defects) ·
[Surface crack](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection).

See `docs/ASSUMPTIONS.md` (A1–A3) for the full rationale.
