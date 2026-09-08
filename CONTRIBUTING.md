# Contributing

This is a university assignment repository, but it is engineered like a real
project. If you are extending it:

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev,io,data]"
pre-commit install
```

## Before you push

```bash
ruff check . && ruff format --check .
mypy src/cvlab
pytest -q                 # 120 tests; add one for every behaviour change
cvlab all                 # figures + metrics must still regenerate
```

`pre-commit run --all-files` runs the first three automatically.

## Ground rules

- **The core library imports only `numpy`** (plus `matplotlib` inside `viz.py`).
  `cv2` is I/O-only and lazily imported; `scipy` is dev/test-only. Do not add a
  filtering primitive from any library — the point is first principles.
- Every public function has a docstring with the governing equation and shape
  contract, and at least one test with a hand-computed or analytic oracle.
- Experiments are deterministic: take an explicit `numpy.random.Generator`,
  never touch global `np.random`.
- Keep `main.tex` at 4 pages. Longer material goes in `supplementary.tex`.

## Academic integrity

If you are enrolled in CS4059 or an equivalent course, this repo is for
reference only. Do not submit it, in whole or in part, as your own work.
