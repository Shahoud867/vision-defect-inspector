# Submission, verification & portfolio checklist

## How to reproduce everything from a clean checkout

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cvlab all            # regenerate all 22 figures + 6 metrics JSON (~16 s)
pytest -q            # 124 tests, 1 skipped (SciPy oracle; dev-only)
ruff check . && ruff format --check . && mypy src/cvlab

cd report && make    # report.pdf (4 pp IEEE) + supplementary.pdf (14 pp)
cd .. && python scripts/build_submission.py   # -> 23i-2515_Assignment01.zip
```

Real Kaggle data (optional, exact-benchmark reproduction):

```bash
pip install -e ".[data]"          # kagglehub;  token at ~/.kaggle/kaggle.json
python scripts/download_data.py
cvlab all --data-dir data/raw --prefer real
```

## Assignment-compliance checklist (grading rubric)

| # | Rubric item | Where | Verified by |
|---|---|---|---|
| T1 | `conv2d` — `same`/`valid`, stride, discrete size formula | `src/cvlab/convolution.py` | `tests/test_convolution.py` (7-geometry sweep), `results/task1_shape_verification.json` (7/7) |
| T1 | `conv3d` — K filters → `(H_out,W_out,K)` | `src/cvlab/convolution.py` | `tests/test_convolution.py::test_conv3d_*` |
| T1 | zero built-in filtering calls | whole `cvlab/` | `cvlab` imports only numpy (+ matplotlib in `viz`) — see `docs/ASSUMPTIONS.md` A20 |
| T2 | `generate_gaussian_kernel_2d` — meshgrid, Σ=1 | `src/cvlab/filters.py` | `tests/test_filters.py` (unit sum, symmetry, closed form) |
| T2 | `unsharp_mask` — `F+α(F−F∗G)`, clip `[0,255]` | `src/cvlab/filters.py` | `tests/test_filters.py::test_unsharp_*` |
| T2 | `pool2d` — max & average | `src/cvlab/pooling.py` | `tests/test_pooling.py` (hand-computed 4×4) |
| T2 | max-pool gradient for `[[8,7],[12,9]]` = `[[0,0],[1,0]]` + explanation | `report` §II-C + Appendix A; `src/cvlab/pooling.py::max_pool_gradient` | `tests/test_pooling.py` (analytic == finite-difference), `results/task2_metrics.json` |
| T3 | Sobel `1/8`, `M=√(Sx²+Sy²)`, `θ=atan2(Sy,Sx)` | `src/cvlab/canny.py::sobel_gradients` | `tests/test_filters.py`, `tests/test_canny.py` |
| T3 | 4-sector NMS, strict `>`, borders 0 | `src/cvlab/canny.py::non_max_suppression` | `tests/test_canny.py` (thinning, border zero, suppression) |
| T3 | double threshold + 8-connected hysteresis, no broken edges | `src/cvlab/canny.py` | `tests/test_canny.py` (BFS==iterative, diagonal link, weak bridge) |
| T4 | Ablation 1 — `σ_n∈{10,25,40}`, grad w/ vs w/o pre-smoothing | `task4_ablation1_noise_smoothing*.png` | `results/task4_metrics.json` (suppression 2.4×→4.7×) |
| T4 | Ablation 2 — edge map pre/post NMS | `task4_ablation2_nms.png` | thinning ratio 8.1× |
| T4 | Ablation 3 — hysteresis vs single global threshold | `task4_ablation3_hysteresis.png` | F1 0.943→0.988, 40→12 components |
| T5 | Scale-space causality `σ∈{1.0,2.5,5.0}` | `task5_scale_space*.png` | edge counts 2055→1418→881, monotone |
| T5 | 3×3 `(T_high,T_low)` grid on noisy concrete & PCB | `task5_sensitivity_grid_*.png` | `results/task5_metrics.json` |
| T5 | ≥2 documented failure cases + theory | `task5_failure_*.png`, `report` §V | shadow P=0.41, texture P=0.35 |
| — | `<Roll>_Assignment01.zip` = `src/` + `output_images/` + `report.pdf` (≤4 pp) | `scripts/build_submission.py` | zip = 4-page `report.pdf` + `src/` + 22 figures (+ extras) |

## Testing / verification checklist

- [x] `pytest` — 124 passed, 1 skipped (SciPy oracle unavailable in this env; runs in CI)
- [x] `ruff check .` — clean
- [x] `ruff format --check .` — clean
- [x] `mypy src/cvlab` — clean
- [x] `cvlab all` regenerates every figure deterministically (seed 2515)
- [x] `scripts/report_numbers.py` — every number in `report.pdf` matches `results/*.json`
- [x] clean-venv run (numpy + matplotlib only, no cv2/scipy) — tests + `cvlab all` pass
- [x] `report.pdf` compiles (tectonic/latexmk) — 4 pages, no undefined refs / overfull boxes (1 benign underfull line in the abstract)
- [x] `supplementary.pdf` compiles — 14 pages, ToC / LoF / LoT / bookmarks
- [x] submission zip builds, contains the 3 required trees, SHA256SUMS included

## Portfolio-readiness checklist

- [x] `README.md` with badges, hero image, quickstart, results gallery, API sample, compliance table
- [x] `pyproject.toml` (src layout, console script, typed), pinned extras `dev`/`io`/`data`
- [x] `LICENSE` (MIT + academic-integrity note), `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`
- [x] `.github/workflows/ci.yml` — 3.10/3.11/3.12 matrix: ruff + mypy + pytest + `cvlab all`
- [x] `.github/workflows/report.yml` — TeX Live build of both PDFs
- [x] `.pre-commit-config.yaml`, `.editorconfig`, `.gitattributes`
- [x] `docs/` — DESIGN, ASSUMPTIONS, REQUIRED_VS_ENHANCEMENTS, this file
- [x] `notebooks/walkthrough.ipynb` — executed, outputs committed
- [x] deterministic synthetic datasets so the repo runs with zero external assets
- [x] git initialised, clean history, no build artefacts tracked

## Known limitations / future work

- **SciPy oracle test** skips on the dev machine (broken user-site SciPy); it is a
  dev-only cross-check and runs in CI. The library itself never imports SciPy.
- **Real datasets not bundled** (Kaggle credentials). Synthetic stand-ins match
  the statistics and carry ground truth; `--data-dir` switches to the real data.
- **Failure modes are inherent to Canny** (shadows, dense texture) — the report
  explains why and points at retinex pre-processing, oriented crack-matched
  filters and multi-scale ridge fusion as the fixes.
- **One benign underfull line** in the IEEE abstract (fixed-width block, many
  numeric tokens) — cosmetic, no visible defect.
- Roadmap in `README.md`: illumination-invariant pre-processing, Frangi ridge
  measure, separable scale-space pyramid, optional Numba fast path, Streamlit demo.
