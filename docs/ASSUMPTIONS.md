# Assumptions & Design Decisions

Every place where the brief left a choice, plus the reasoning. Nothing here
changes what is graded — it documents *how* the requirements were met.

## Datasets

| # | Assumption | Rationale |
|---|---|---|
| A1 | The Kaggle **PCB-Defect** and **Surface-Crack** datasets are **not committed**. When `data/raw/<category>/` is empty the loader generates deterministic procedural stand-ins with matched statistics **and an exact ground-truth edge mask**. | Both datasets require Kaggle credentials; a marker (`gitkeep`) keeps the folder. The synthetic path makes the whole pipeline runnable — and quantitatively evaluable — from a clean checkout with zero external assets. |
| A2 | `scripts/download_data.py` + `--data-dir data/raw --prefer real` switches every experiment to the real images with **no code change**. | Keeps the assignment's benchmark intent intact; the abstraction boundary is `cvlab.datasets.get_samples`. |
| A3 | Real images are centre-cropped to a square and nearest-neighbour resampled to `--image-size` (default 256). | Uniform figure geometry; NN resample needs no `cv2`. |

## Filtering conventions

| # | Assumption | Rationale |
|---|---|---|
| A4 | All spatial filtering is **cross-correlation** (no kernel flip). A `flip_kernel=True` flag gives true convolution. | Brief §1.2 explicitly permits it; matches every DL framework. |
| A5 | Kernels are **odd-sized**; `'same'` ⇒ `P = (F−1)//2`, which preserves size **only at stride 1** (consistent with the discrete formula). | Brief §1.2. |
| A6 | Gaussian blur and Sobel use **`reflect`** border padding, *not* zero padding. NMS then explicitly zeroes the 1-px border. | Zero padding darkens the frame and injects a false step gradient at the edge; reflect removes both. The brief only mandates zero border **after NMS**, which is still done. Overridable via `pad_mode`. |
| A7 | The Gaussian kernel is evaluated from the closed form **then** renormalised so `Σ = 1` exactly. | Removes the finite-window truncation error; guarantees unit DC gain. |
| A8 | `unsharp_mask` returns a `float64` array already clipped to `[0,255]`; the caller casts to `uint8` at save time. | Avoids double rounding; keeps the value range explicit. |

## Canny

| # | Assumption | Rationale |
|---|---|---|
| A9 | Orientation `θ = atan2(Gy, Gx)` with the brief's `Sy` (positive row on top) is used **verbatim**. The 4 NMS sectors and their neighbour pairs are exactly as listed in §2.3. | The brief's `Sy`/sector table is internally consistent (its `Sy` sign makes `θ` a y-axis-up angle, which matches the sector→neighbour mapping). |
| A10 | NMS keeps a pixel iff `M` is **strictly greater** than *both* directional neighbours (`>`), not `>=`. | Brief wording: "not *strictly* greater than both". Ridge plateaus lose at most one duplicated pixel. |
| A11 | Default `(T_high, T_low)` are **quantiles** of the non-zero NMS response (`high_q=0.90`, `low_q=0.70`); absolute thresholds are accepted via `t_low`/`t_high`. A degenerate all-zero gradient short-circuits to an empty edge map. | Quantiles transfer across illumination and dataset; the assignment's sensitivity study varies them explicitly. |
| A12 | Two hysteresis implementations — deque BFS (the brief skeleton) and fixed-point 8-connected dilation — are both provided and asserted equal in tests. `method='bfs'` is the default. | BFS mirrors the skeleton for grading; the vectorised one is a cross-check and a fast path. |
| A13 | Grayscale conversion uses **ITU-R BT.601 luma** (`0.299, 0.587, 0.114`), implemented in NumPy (not `cv2.cvtColor`). | Keeps the core pipeline OpenCV-free; `cv2.cvtColor` would also be allowed. |

## Experiments & evaluation

| # | Assumption | Rationale |
|---|---|---|
| A14 | Master seed = **2515** (last four digits of the roll number). Every stochastic step takes an explicit `numpy.random.Generator`. | Bit-reproducible figures and metrics. |
| A15 | Where ground truth exists (synthetic data), quality is **tolerance-aware precision/recall/F₁** (a hit if within Chebyshev distance 2). Elsewhere: edge density, 8-connected component count, mean component length. | Gives the report real numbers, not just visual claims; tolerance matches standard boundary-detection evaluation. |
| A16 | Ablation 3 uses a deliberately **faint, depth-modulated** synthetic crack. | So the single-high-threshold *break* and the hysteresis *link* are both visible; a high-contrast crack would make all three methods look identical. |
| A17 | The Task 5 monotonicity check allows a **1.15× slack**. | Strict extremum monotonicity under Gaussian smoothing is a 1-D result; in 2-D the edge-*count* trend is decreasing but not provably strict every step. |
| A18 | Report figures are 150 DPI (`--dpi` overrides); the LaTeX build pulls them from `../output_images/`. | Balance of file size and print quality. |

## Environment

| # | Assumption | Rationale |
|---|---|---|
| A19 | Use a **virtual environment**. On the development machine a broken `numpy` in the user site-packages (`%APPDATA%\Python`) can shadow the install; `python -m venv` isolates it. CI uses a clean matrix (3.10–3.12). | Reproducibility. |
| A20 | The core library imports **only** `numpy` (+ `matplotlib` in `viz`). `cv2` is imported lazily and only for I/O, with a `matplotlib.image` fallback. `scipy` appears **only** in dev/oracle tests. | Honours the "allowed libraries" constraint with room to spare. |
