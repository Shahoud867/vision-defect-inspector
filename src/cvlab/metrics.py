"""Quantitative edge-map metrics for the ablation and sensitivity studies.

When a ground-truth edge mask is available (the synthetic generators expose one)
we report tolerance-aware precision / recall / F1.  Otherwise we fall back to
reference-free structural descriptors -- edge density, connected-component count
and mean component length -- which still expose over- and under-detection.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "EdgeMetrics",
    "binarize",
    "edge_density",
    "count_components",
    "prf1",
    "evaluate_edges",
]


@dataclass(frozen=True)
class EdgeMetrics:
    density: float
    n_components: int
    mean_component_len: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def binarize(edge_map: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Coerce a grayscale / uint8 / bool edge map to a boolean mask."""
    arr = np.asarray(edge_map)
    if arr.dtype == bool:
        return arr
    return arr > threshold


def edge_density(edge_map: np.ndarray) -> float:
    """Fraction of pixels flagged as an edge."""
    mask = binarize(edge_map)
    return float(mask.mean())


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element dilation by ``radius`` (Chebyshev ball)."""
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        out = grown
    return out


def count_components(edge_map: np.ndarray) -> tuple[int, float]:
    """8-connected component count and mean component size (pixels)."""
    mask = binarize(edge_map)
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    sizes: list[int] = []
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or seen[i, j]:
                continue
            size = 0
            q = deque([(i, j)])
            seen[i, j] = True
            while q:
                ci, cj = q.popleft()
                size += 1
                for ni in range(max(0, ci - 1), min(H, ci + 2)):
                    for nj in range(max(0, cj - 1), min(W, cj + 2)):
                        if mask[ni, nj] and not seen[ni, nj]:
                            seen[ni, nj] = True
                            q.append((ni, nj))
            sizes.append(size)
    if not sizes:
        return 0, 0.0
    return len(sizes), float(np.mean(sizes))


def prf1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> tuple[float, float, float]:
    """Tolerance-aware precision / recall / F1 (Canny-style boundary matching).

    A predicted edge pixel counts as a true positive if a ground-truth edge lies
    within a Chebyshev distance of ``tolerance`` (and vice-versa for recall).
    """
    p = binarize(pred)
    g = binarize(gt)
    if p.shape != g.shape:
        raise ValueError("pred and gt must have identical shape")
    g_dil = _dilate(g, tolerance)
    p_dil = _dilate(p, tolerance)
    tp_p = np.count_nonzero(p & g_dil)
    tp_r = np.count_nonzero(g & p_dil)
    n_pred = int(p.sum())
    n_gt = int(g.sum())
    precision = tp_p / n_pred if n_pred else 0.0
    recall = tp_r / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def evaluate_edges(
    edge_map: np.ndarray, gt: np.ndarray | None = None, tolerance: int = 2
) -> EdgeMetrics:
    """Bundle all applicable metrics for a single edge map."""
    n_comp, mean_len = count_components(edge_map)
    p = r = f = None
    if gt is not None:
        p, r, f = prf1(edge_map, gt, tolerance=tolerance)
    return EdgeMetrics(
        density=edge_density(edge_map),
        n_components=n_comp,
        mean_component_len=mean_len,
        precision=p,
        recall=r,
        f1=f,
    )
