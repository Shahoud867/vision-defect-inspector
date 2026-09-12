"""Task 3 -- first-principles four-stage Canny edge detector.

Stages
------
1. Pre-smoothing (optional Gaussian) + normalised Sobel gradients -> magnitude, orientation.
2. Non-maximum suppression with 4-sector orientation quantisation.
3. Double thresholding into strong / weak / non-edge.
4. 8-connected hysteresis edge tracking (BFS and vectorised implementations agree).

Only :mod:`numpy` is used.  ``cv2.Canny`` / ``cv2.Sobel`` are never called.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .convolution import conv2d
from .filters import SOBEL_X, SOBEL_Y, gaussian_blur
from .utils import to_grayscale

__all__ = [
    "CannyStages",
    "sobel_gradients",
    "non_max_suppression",
    "double_threshold",
    "hysteresis",
    "quantile_thresholds",
    "canny",
]

HysteresisMethod = Literal["bfs", "iterative"]


@dataclass(frozen=True)
class CannyStages:
    """Container for every intermediate map produced by :func:`canny`."""

    gray: np.ndarray
    smoothed: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    magnitude: np.ndarray
    orientation: np.ndarray  # radians, atan2(Gy, Gx)
    nms: np.ndarray
    strong: np.ndarray  # bool
    weak: np.ndarray  # bool
    edges: np.ndarray  # uint8 {0, 255}
    t_low: float
    t_high: float


# --------------------------------------------------------------------------- #
# Stage 1 -- gradients
# --------------------------------------------------------------------------- #
def sobel_gradients(
    image: np.ndarray,
    smooth_sigma: float | None = None,
    ksize: int | None = None,
    pad_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(Gx, Gy, magnitude, orientation)`` for a grayscale image.

    ``magnitude`` :math:`= \sqrt{G_x^2 + G_y^2}` and ``orientation``
    :math:`= \operatorname{atan2}(G_y, G_x)` in radians.  When ``smooth_sigma``
    is given the image is Gaussian pre-smoothed first (Stage 1 of Canny).
    """
    gray = to_grayscale(image)
    if smooth_sigma is not None and smooth_sigma > 0:
        gray = gaussian_blur(gray, sigma=smooth_sigma, ksize=ksize, pad_mode=pad_mode)
    gx = conv2d(gray, SOBEL_X, padding="same", pad_mode=pad_mode)
    gy = conv2d(gray, SOBEL_Y, padding="same", pad_mode=pad_mode)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    return gx, gy, magnitude, orientation


# --------------------------------------------------------------------------- #
# Stage 2 -- non-maximum suppression
# --------------------------------------------------------------------------- #
def non_max_suppression(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Thin ridges to 1-px width by suppressing non-maximal gradient pixels.

    Orientation (radians) is mapped to ``[0, 180)`` degrees and quantised into
    four sectors; a pixel survives only if its magnitude is **strictly greater**
    than both neighbours along the gradient direction.  Borders are set to 0.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.shape != orientation.shape:
        raise ValueError("magnitude and orientation must have identical shape")
    H, W = mag.shape

    angle = np.rad2deg(orientation) % 180.0
    s0 = (angle < 22.5) | (angle >= 157.5)  # horizontal gradient
    s45 = (angle >= 22.5) & (angle < 67.5)  # diagonal  (/)
    s90 = (angle >= 67.5) & (angle < 112.5)  # vertical gradient
    s135 = (angle >= 112.5) & (angle < 157.5)  # anti-diagonal (\)

    p = np.pad(mag, 1, mode="constant", constant_values=0.0)
    north = p[0:H, 1 : W + 1]
    south = p[2 : H + 2, 1 : W + 1]
    west = p[1 : H + 1, 0:W]
    east = p[1 : H + 1, 2 : W + 2]
    n_e = p[0:H, 2 : W + 2]
    n_w = p[0:H, 0:W]
    s_e = p[2 : H + 2, 2 : W + 2]
    s_w = p[2 : H + 2, 0:W]

    neigh_a = np.select([s0, s45, s90, s135], [west, n_e, north, n_w], default=0.0)
    neigh_b = np.select([s0, s45, s90, s135], [east, s_w, south, s_e], default=0.0)

    keep = (mag > neigh_a) & (mag > neigh_b)
    out = np.where(keep, mag, 0.0)
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Stage 3 -- double threshold + hysteresis
# --------------------------------------------------------------------------- #
def quantile_thresholds(
    nms: np.ndarray, low_q: float = 0.70, high_q: float = 0.90
) -> tuple[float, float]:
    """Data-adaptive ``(t_low, t_high)`` from quantiles of the non-zero NMS response."""
    vals = nms[nms > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(np.quantile(vals, low_q)), float(np.quantile(vals, high_q))


def double_threshold(nms: np.ndarray, t_low: float, t_high: float) -> tuple[np.ndarray, np.ndarray]:
    """Split the NMS map into ``(strong, weak)`` boolean masks.

    ``strong`` = ``M >= t_high``; ``weak`` = ``t_low <= M < t_high``.
    """
    if t_low > t_high:
        raise ValueError(f"t_low ({t_low}) must not exceed t_high ({t_high})")
    mag = np.asarray(nms, dtype=np.float64)
    # A zero-magnitude pixel is a non-edge by definition; the ``> 0`` guard keeps
    # a degenerate ``t_high == 0`` from promoting a flat image to "all strong".
    strong = (mag >= t_high) & (mag > 0.0)
    weak = (mag >= t_low) & (mag > 0.0) & ~strong
    return strong, weak


def _hysteresis_bfs(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Deque-based 8-connectivity flood fill from strong seeds into weak pixels."""
    H, W = strong.shape
    out = np.zeros((H, W), dtype=np.uint8)
    si, sj = np.where(strong)
    out[si, sj] = 255
    q: deque[tuple[int, int]] = deque(zip(si.tolist(), sj.tolist(), strict=False))
    while q:
        ci, cj = q.popleft()
        for ni in range(max(0, ci - 1), min(H, ci + 2)):
            for nj in range(max(0, cj - 1), min(W, cj + 2)):
                if weak[ni, nj] and out[ni, nj] == 0:
                    out[ni, nj] = 255
                    q.append((ni, nj))
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """8-connected binary dilation via nine shifted ORs (no SciPy)."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _hysteresis_iterative(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Fixed-point 8-connected dilation of strong seeds constrained to weak|strong."""
    allowed = strong | weak
    edges = strong.copy()
    while True:
        grown = _dilate8(edges) & allowed
        if np.array_equal(grown, edges):
            break
        edges = grown
    return (edges * 255).astype(np.uint8)


def hysteresis(
    strong: np.ndarray, weak: np.ndarray, method: HysteresisMethod = "bfs"
) -> np.ndarray:
    """Retain weak pixels iff 8-connected (transitively) to a strong pixel.

    ``method='bfs'`` mirrors the assignment skeleton; ``'iterative'`` is a fully
    vectorised fixed-point dilation.  Both yield identical edge maps.
    """
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    if strong.shape != weak.shape:
        raise ValueError("strong and weak masks must have identical shape")
    if method == "bfs":
        return _hysteresis_bfs(strong, weak)
    if method == "iterative":
        return _hysteresis_iterative(strong, weak)
    raise ValueError(f"method must be 'bfs' or 'iterative'; got {method!r}")


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def canny(
    image: np.ndarray,
    t_low: float | None = None,
    t_high: float | None = None,
    sigma: float = 1.0,
    ksize: int | None = None,
    smooth: bool = True,
    low_q: float = 0.70,
    high_q: float = 0.90,
    pad_mode: str = "reflect",
    hysteresis_method: HysteresisMethod = "bfs",
    return_stages: bool = False,
) -> np.ndarray | CannyStages:
    """Run the complete four-stage detector.

    ``t_low`` / ``t_high`` are absolute magnitude thresholds.  If either is
    ``None`` both are derived from ``low_q`` / ``high_q`` quantiles of the
    post-NMS response (robust across illumination and dataset).

    Returns a ``uint8`` ``{0, 255}`` edge map, or a :class:`CannyStages` record
    of every intermediate array when ``return_stages`` is set.
    """
    gray = to_grayscale(image)
    smoothed = gaussian_blur(gray, sigma=sigma, ksize=ksize, pad_mode=pad_mode) if smooth else gray
    gx = conv2d(smoothed, SOBEL_X, padding="same", pad_mode=pad_mode)
    gy = conv2d(smoothed, SOBEL_Y, padding="same", pad_mode=pad_mode)
    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)

    nms = non_max_suppression(magnitude, orientation)

    if t_low is None or t_high is None:
        t_low, t_high = quantile_thresholds(nms, low_q, high_q)

    if t_high <= 0.0:  # no gradient structure at all -> no edges
        strong = np.zeros(nms.shape, dtype=bool)
        weak = np.zeros(nms.shape, dtype=bool)
        edges = np.zeros(nms.shape, dtype=np.uint8)
    else:
        strong, weak = double_threshold(nms, t_low, t_high)
        edges = hysteresis(strong, weak, method=hysteresis_method)

    if not return_stages:
        return edges
    return CannyStages(
        gray=gray,
        smoothed=smoothed,
        grad_x=gx,
        grad_y=gy,
        magnitude=magnitude,
        orientation=orientation,
        nms=nms,
        strong=strong,
        weak=weak,
        edges=edges,
        t_low=float(t_low),
        t_high=float(t_high),
    )
