# Changelog

All notable changes to this project. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-14

Initial release — full Assignment 1 submission.

### Added

- **Task 1** — `conv2d` / `conv3d` engines (vectorised `einsum` over
  `sliding_window_view`); `same` / `valid` / int / tuple padding; strides;
  discrete size-formula guarantee; `flip_kernel` for true convolution.
- **Task 2** — `generate_gaussian_kernel_2d` (closed form + exact
  renormalisation), separable 1-D variant, `unsharp_mask`, `pool2d`
  (max / average, 2-D & 3-D), `max_pool_gradient` / `average_pool_gradient`.
- **Task 3** — `sobel_gradients`, vectorised 4-sector `non_max_suppression`,
  `double_threshold`, `hysteresis` (BFS + fixed-point dilation), `canny`
  orchestrator with `return_stages` and quantile auto-thresholds.
- **Task 4** — `add_gaussian_noise` / `add_salt_pepper_noise`; three ablation
  drivers with side-by-side figures and measured deltas.
- **Task 5** — scale-space causality (edge-count + zero-crossing fingerprint),
  3×3 `(T_high, T_low)` sensitivity grid, two failure-case studies.
- `cvlab.metrics` — tolerance-aware precision / recall / F1, edge density,
  8-connected component stats.
- `cvlab.datasets` — real-image loader + deterministic procedural PCB / concrete
  generators with ground-truth edge masks.
- `cvlab` CLI, 124-test `pytest` suite (~85 % coverage), `ruff` + `mypy` +
  `pre-commit`, GitHub Actions for tests and the LaTeX report.
- Modular LaTeX report (`report.pdf`, 4-page IEEE) + extended `supplementary.pdf`.
- `scripts/`: `download_data.py`, `report_numbers.py`, `build_submission.py`.
