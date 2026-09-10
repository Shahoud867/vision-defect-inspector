# Design Notes

Architecture, algorithm choices and complexity for `cvlab`.

## Module map

```
cvlab/
  utils.py        pure helpers: size arithmetic, padding, normalisation, I/O
  convolution.py  conv2d, conv3d           (depends on utils)
  filters.py      Gaussian, Sobel, unsharp (depends on convolution, utils)
  pooling.py      pool2d + gradients       (depends on utils)
  noise.py        additive noise models    (depends on utils)
  canny.py        4-stage detector         (depends on convolution, filters, utils)
  metrics.py      P/R/F1, density, CCs     (depends on nothing but numpy)
  datasets.py     real loader + synthetic  (depends on utils)
  viz.py          matplotlib helpers       (depends on matplotlib only)
  cli.py          argparse dispatch        (depends on experiments/)
```

Dependency graph is a DAG rooted at `utils`; `metrics` and `viz` are leaves.
The library imports **only** `numpy` (and `matplotlib` inside `viz`). `cv2` is
imported lazily in `utils` for file I/O and degrades to `matplotlib.image`.

## Convolution — one contraction, no pixel loop

The naïve skeleton is a double `for i, j` loop over output pixels. Instead:

1. `np.pad` the image per the resolved `(pad_h, pad_w)`.
2. `sliding_window_view(padded, (k_h, k_w), axis=(0,1))` → a **read-only view**
   of shape `(Oh_full, Ow_full, [C,] k_h, k_w)` (zero copy).
3. `moveaxis` so window axes are adjacent, slice `[::stride, ::stride]`.
4. Contract against the kernel in one call:
   - 2-D: `einsum("ijhw,hw->ij", W, K)`
   - 3-D: `einsum("ijhwc,khwc->ijk", W, stack)` — the `k` index is the filter
     bank, so all `K` filters are computed together.

**Complexity** `O(Oh·Ow·k_h·k_w·C·K)` FLOPs, but vectorised into BLAS-backed
`einsum`. The strided view costs `O(1)` memory; `einsum` allocates only the
output. Measured: `512² * 5×5` in ~3 ms, `256²×3` with `K=4` in ~6 ms.

**Border handling** is a parameter (`pad_mode`), defaulting to `constant` in
`conv2d` (skeleton-compatible) but `reflect` in `gaussian_blur`/`sobel_gradients`
(see ASSUMPTIONS A6).

## NMS — vectorised 4-sector suppression

Rather than looping pixels and branching on angle:

1. `angle = rad2deg(θ) % 180` → 4 boolean sector masks partitioning `[0,180)`.
2. Pad `M` by 1 and materialise the 8 neighbour planes by slicing
   (`north`, `south`, `east`, `west`, and the 4 diagonals).
3. `neigh_a = np.select([s0,s45,s90,s135], [west, ne, north, nw])` and
   likewise `neigh_b` — each pixel now has its two along-gradient neighbours.
4. `keep = (M > neigh_a) & (M > neigh_b)`; `out = where(keep, M, 0)`; zero the
   border.

All array ops, `O(HW)`, no Python-level branching.

## Hysteresis — two implementations, asserted equal

- **`_hysteresis_bfs`** — seed a `deque` with all strong pixels, pop and mark
  any 8-neighbour that is weak and unmarked. `O(N)` with `O(1)` pops; this is
  the brief's skeleton.
- **`_hysteresis_iterative`** — `edges = strong`; repeat
  `edges = dilate8(edges) & (strong | weak)` until a fixed point. `dilate8` is
  9 shifted OR-assignments. Fewer iterations than BFS has pops when edges are
  long and thin; fully vectorised.

`tests/test_canny.py::test_hysteresis_bfs_and_iterative_agree` fuzzes random
strong/weak masks and asserts the outputs are identical.

## Thresholds

`quantile_thresholds(nms, low_q, high_q)` takes quantiles of the **non-zero**
NMS response — robust to illumination and image content. `canny()` accepts
absolute `t_low`/`t_high` too. A guard: if `t_high <= 0` (flat image) the result
is an all-zero edge map, and `double_threshold` requires `M > 0` so a degenerate
`t_high = 0` never promotes a constant image to "all strong".

## Synthetic datasets

`synthetic_pcb` rasterises orthogonal bright traces + circular pads on a dark
substrate (pure NumPy drawing) and injects an open circuit and a spur.
`synthetic_concrete` builds a fractal value-noise surface, multiplies a radial
illumination gradient, and subtracts thin dark cracks drawn as **distance-field
polylines** (`exp(-d²/2w²)`), with a low-frequency depth modulation so a single
crack has strong and faint stretches. Both return a `Sample` with `gt_edges`
(bool), enabling P/R/F₁.

## Testing strategy

- **Analytic oracles first** — closed-form Gaussian on a 3×3, hand-computed box
  filter, the discrete size formula, the max-pool routing matrix.
- **Property tests** — linearity of convolution, `conv2d ≡ conv3d` on one
  channel, kernel symmetry, noise-σ recovery, rotation-invariance of Canny edge
  counts.
- **Cross-implementation** — BFS vs iterative hysteresis.
- **Optional SciPy oracle** — `@pytest.mark.oracle`, `importorskip`, never a
  hard dependency (and the core library never imports scipy).
- **End-to-end smoke** — `@pytest.mark.slow`, runs every task driver at tiny
  resolution and checks artefacts land on disk.
