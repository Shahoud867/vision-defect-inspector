# notebooks/

- **`walkthrough.ipynb`** — a ~5-minute interactive tour of the `cvlab` public
  API (convolution engines, Gaussian/unsharp, pooling + gradient, the Canny
  stages, one ablation). Outputs are committed so it renders on GitHub without
  running.

Run it yourself:

```bash
pip install -e ".[dev]"
jupyter lab notebooks/walkthrough.ipynb      # or: jupyter notebook
```

For the full reproducible figure set used in the report, run `cvlab all` from
the repo root instead — the notebook is only a guided tour.
