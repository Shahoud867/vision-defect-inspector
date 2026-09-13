"""Matplotlib helpers for consistent, publication-quality figures.

Uses the non-interactive ``Agg`` backend so figures render identically in CI and
on head-less machines.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["FIGURE_DPI", "apply_style", "show_grid", "save_figure"]

FIGURE_DPI = 150

_STYLE = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "font.size": 9,
    "image.cmap": "gray",
    "image.interpolation": "nearest",
}


def apply_style() -> None:
    plt.rcParams.update(_STYLE)


def show_grid(
    images: Sequence,
    titles: Sequence[str] | None = None,
    ncols: int = 3,
    cmaps: Sequence[str] | str | None = None,
    suptitle: str | None = None,
    figsize_scale: float = 3.2,
    vranges: Sequence[tuple[float, float] | None] | None = None,
) -> plt.Figure:
    """Lay out ``images`` on a tidy grid of imshow panels."""
    apply_style()
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_scale * ncols, figsize_scale * nrows + (0.6 if suptitle else 0.0)),
        squeeze=False,
        constrained_layout=True,
    )
    flat = axes.ravel()
    for idx, ax in enumerate(flat):
        if idx >= n:
            ax.axis("off")
            continue
        cmap = cmaps[idx] if isinstance(cmaps, (list, tuple)) else cmaps
        vr = vranges[idx] if vranges is not None else None
        kw = {} if vr is None else {"vmin": vr[0], "vmax": vr[1]}
        ax.imshow(images[idx], cmap=cmap or "gray", **kw)
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])
        ax.set_xticks([])
        ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    return fig


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = FIGURE_DPI) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def close_all(_: Iterable | None = None) -> None:
    plt.close("all")
