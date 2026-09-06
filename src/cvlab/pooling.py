"""Task 2 (part 3) -- 2-D spatial pooling and its analytical gradient.

``pool2d`` supports ``'max'`` and ``'average'`` reduction over non-overlapping or
strided windows.  ``max_pool_gradient`` returns the routing matrix
:math:`\\partial\\,\\mathrm{Output}/\\partial\\,\\mathrm{Input}` used for the
theoretical reflection in the report.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .utils import compute_output_size

__all__ = ["pool2d", "max_pool_gradient", "average_pool_gradient"]

PoolMode = Literal["max", "average"]


def _window_view(fmap: np.ndarray, ph: int, pw: int, stride: int) -> tuple[np.ndarray, int, int]:
    """Strided, non-overlapping-by-default window view of a 2-D feature map."""
    H, W = fmap.shape
    out_h = compute_output_size(H, ph, 0, stride)
    out_w = compute_output_size(W, pw, 0, stride)
    win = sliding_window_view(fmap, (ph, pw))[::stride, ::stride]
    return win[:out_h, :out_w], out_h, out_w


def pool2d(
    feature_map: np.ndarray,
    pool_size: tuple[int, int] = (2, 2),
    stride: int = 2,
    mode: PoolMode = "max",
) -> np.ndarray:
    r"""Down-sample ``feature_map`` by pooling over ``pool_size`` windows.

    Parameters
    ----------
    feature_map
        2-D array ``(H, W)`` or 3-D array ``(H, W, C)`` (pooled channel-wise).
    pool_size
        ``(pool_h, pool_w)`` window extent.
    stride
        Window step (``stride == pool_h == pool_w`` gives classic tiling).
    mode
        ``'max'`` or ``'average'``.

    Returns
    -------
    numpy.ndarray
        Pooled map of shape
        :math:`(\lfloor (H-p_h)/S \rfloor + 1,\ \lfloor (W-p_w)/S \rfloor + 1)`
        (with a trailing channel axis preserved for 3-D input).

    Notes
    -----
    Incomplete border windows are dropped (``'valid'`` semantics), matching the
    discrete spatial formula used throughout Task 1.
    """
    arr = np.asarray(feature_map, dtype=np.float64)
    ph, pw = int(pool_size[0]), int(pool_size[1])
    if ph < 1 or pw < 1:
        raise ValueError(f"pool_size must be >= 1 on both axes; got {pool_size!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if mode not in ("max", "average"):
        raise ValueError(f"mode must be 'max' or 'average'; got {mode!r}")

    if arr.ndim == 3:
        return np.stack(
            [pool2d(arr[..., c], pool_size, stride, mode) for c in range(arr.shape[2])],
            axis=-1,
        )
    if arr.ndim != 2:
        raise ValueError(f"feature_map must be 2-D or 3-D; got shape {arr.shape!r}")

    win, out_h, out_w = _window_view(arr, ph, pw, stride)
    if mode == "max":
        pooled = np.max(win, axis=(2, 3))
    else:
        pooled = np.mean(win, axis=(2, 3))
    return np.ascontiguousarray(pooled)


def max_pool_gradient(
    feature_map: np.ndarray,
    pool_size: tuple[int, int] = (2, 2),
    stride: int = 2,
    upstream: np.ndarray | None = None,
) -> np.ndarray:
    r"""Routing matrix :math:`\partial\,\mathrm{Output}/\partial\,\mathrm{Input}` for max pooling.

    Each pooling window contributes a sub-gradient that is ``1`` at the position
    of the (first) maximum and ``0`` elsewhere -- the max operator is locally the
    identity on its argmax and constant (zero derivative) in every other input.
    With ``upstream`` supplied, the upstream gradient of each output cell is
    scattered back to its argmax (full back-propagation).

    For non-overlapping windows the returned array has the same shape as
    ``feature_map``.  Overlapping windows accumulate contributions.
    """
    arr = np.asarray(feature_map, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"feature_map must be 2-D; got shape {arr.shape!r}")
    ph, pw = int(pool_size[0]), int(pool_size[1])
    win, out_h, out_w = _window_view(arr, ph, pw, stride)

    grad = np.zeros_like(arr)
    up = np.ones((out_h, out_w)) if upstream is None else np.asarray(upstream, dtype=np.float64)
    if up.shape != (out_h, out_w):
        raise ValueError(f"upstream must have shape {(out_h, out_w)}; got {up.shape!r}")

    for i in range(out_h):
        for j in range(out_w):
            block = win[i, j]
            r, c = np.unravel_index(int(np.argmax(block)), block.shape)
            grad[i * stride + r, j * stride + c] += up[i, j]
    return grad


def average_pool_gradient(
    feature_map_shape: tuple[int, int],
    pool_size: tuple[int, int] = (2, 2),
    stride: int = 2,
    upstream: np.ndarray | None = None,
) -> np.ndarray:
    r"""Routing matrix for average pooling: every window input gets weight
    :math:`1 / (p_h p_w)` (times the upstream gradient of its output cell)."""
    H, W = feature_map_shape
    ph, pw = int(pool_size[0]), int(pool_size[1])
    out_h = compute_output_size(H, ph, 0, stride)
    out_w = compute_output_size(W, pw, 0, stride)
    grad = np.zeros((H, W), dtype=np.float64)
    up = np.ones((out_h, out_w)) if upstream is None else np.asarray(upstream, dtype=np.float64)
    scale = 1.0 / (ph * pw)
    for i in range(out_h):
        for j in range(out_w):
            grad[i * stride : i * stride + ph, j * stride : j * stride + pw] += up[i, j] * scale
    return grad
