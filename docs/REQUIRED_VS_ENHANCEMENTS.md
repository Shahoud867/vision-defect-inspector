# Required work vs. production improvements vs. optional enhancements

Explicit three-way split, as requested. **Grading depends only on column 1.**

## 1. Required by the brief (assessed)

| Task | Deliverable | Location |
|---|---|---|
| 1 | `conv2d(image, kernel, stride, padding)` — `same`/`valid`, discrete size formula | `src/cvlab/convolution.py::conv2d` |
| 1 | `conv3d(image, kernel_stack, ...)` — K filters → `(H_out, W_out, K)` | `src/cvlab/convolution.py::conv3d` |
| 2 | `generate_gaussian_kernel_2d(ksize, sigma)` — meshgrid, Σ = 1 | `src/cvlab/filters.py` |
| 2 | `unsharp_mask(image, sigma, alpha)` — `F + α(F − F∗G)`, clip `[0,255]` | `src/cvlab/filters.py` |
| 2 | `pool2d(feature_map, pool_size, stride, mode)` — max & average | `src/cvlab/pooling.py` |
| 2 | Max-pool gradient for `[[8,7],[12,9]]` + why gradients route to the max | `report/sections/02_methods.tex` §II-C, `report/sections/A_derivations.tex` |
| 3 | Stage 1 — normalised Sobel (1/8), `M = √(Sx²+Sy²)`, `θ = atan2(Sy,Sx)` | `src/cvlab/canny.py::sobel_gradients` |
| 3 | Stage 2 — 4-sector NMS, strict `>`, borders 0 | `src/cvlab/canny.py::non_max_suppression` |
| 3 | Stage 3 — double threshold + 8-connected hysteresis | `src/cvlab/canny.py::double_threshold`, `hysteresis` |
| 4 | Ablation 1 — noise `σ_n∈{10,25,40}`, gradient with/without pre-smoothing | `src/experiments/run_task4_ablations.py` |
| 4 | Ablation 2 — edge map before vs. after NMS | ″ |
| 4 | Ablation 3 — hysteresis vs. single global threshold | ″ |
| 5 | Scale-space causality `σ∈{1.0,2.5,5.0}` | `src/experiments/run_task5_analysis.py` |
| 5 | 3×3 `(T_high, T_low)` sensitivity grid on noisy concrete & PCB | ″ |
| 5 | ≥ 2 documented Canny failure cases | ″ + `report/sections/05_failure.tex` |
| — | Submission `src/` + `output_images/` + `report.pdf` (≤ 4 pages) | `scripts/build_submission.py` |

## 2. Production-quality improvements (same behaviour, better engineering)

- **Vectorised cores** — `einsum` + `sliding_window_view` convolution; `np.select`
  NMS; shifted-OR dilation. No Python pixel loops. ~16 s for the whole suite.
- **Second hysteresis implementation** (fixed-point dilation) cross-checked against
  the BFS one in tests.
- **Robust thresholds** — quantile-based defaults with an absolute override and a
  degenerate-image guard.
- **`reflect` border padding** for blur/Sobel to kill the dark-rim / false-edge
  artefact (border still zeroed post-NMS per the brief).
- **Exact-unity Gaussian** — closed form then renormalise.
- **Packaging** — `pyproject.toml`, `src/` layout, `cvlab` console script,
  `py.typed`, pinned extras (`dev`, `io`, `data`).
- **Test suite** — 124 `pytest` tests, ~85 % coverage, analytic + property +
  cross-impl + smoke; `ruff` + `mypy` + `pre-commit`.
- **Reproducibility** — every RNG seeded (`2515`); `results/figure_manifest.json`;
  `scripts/report_numbers.py` re-derives every cited number.
- **CI** — GitHub Actions matrix (3.10–3.12) for tests + a TeX Live job that
  builds both PDFs.
- **Docs** — this file, `DESIGN.md`, `ASSUMPTIONS.md`, a full `README.md`.

## 3. Optional enhancements (added because they materially help)

- **Synthetic datasets with exact ground truth** — makes the pipeline runnable
  and *quantitatively* evaluable with no Kaggle account; `--prefer real` uses the
  actual images unchanged.
- **Tolerance-aware P/R/F₁ metrics** (`cvlab.metrics`) — turns the ablations from
  "looks better" into measured deltas.
- **Separable Gaussian** — two 1-D passes, `O(k)` instead of `O(k²)`.
- **Orientation colour-wheel visualisation** in the Task 3 stage gallery.
- **Scale-space fingerprint** (2nd-derivative zero-crossing tracking) alongside
  the edge-count curve for the causality argument.
- **Supplementary report** — title page, ToC, LoF/LoT, full derivations,
  assumptions appendix. Not part of the 4-page limit.

## Deliberately *not* added (out of scope / low value)

- A learned segmenter (defeats the "from first principles" purpose).
- GPU / Numba backends (CPU einsum is already fast enough; would add a heavy dep).
- A web UI beyond the roadmap note (no assessment value).
