"""Task 3 -- full four-stage Canny pipeline visualised stage by stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvlab import canny, evaluate_edges
from cvlab.datasets import get_samples
from cvlab.utils import min_max_scale
from cvlab.viz import apply_style, save_figure

from ._common import ExperimentConfig, banner, dump_json


def _orientation_rgb(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Colour-wheel visualisation: hue = gradient angle, value = magnitude."""
    import matplotlib.colors as mcolors

    hue = (np.rad2deg(orientation) % 180.0) / 180.0
    val = min_max_scale(magnitude, 0, 1)
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def _stage_figure(cfg: ExperimentConfig, cat: str) -> tuple[Path, dict]:
    sample = get_samples(
        cat, n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
    )[0]
    st = canny(sample.image, sigma=1.4, low_q=0.70, high_q=0.92, return_stages=True)

    strong_weak = np.zeros((*st.nms.shape, 3))
    strong_weak[st.weak] = (0.30, 0.55, 1.0)
    strong_weak[st.strong] = (1.0, 0.85, 0.0)

    panels = [
        (st.gray, "1. grayscale input", "gray"),
        (st.smoothed, "1. Gaussian pre-smoothing", "gray"),
        (min_max_scale(st.magnitude, 0, 255), "1. Sobel gradient |M|", "gray"),
        (_orientation_rgb(st.magnitude, st.orientation), "1. orientation (hue) x |M| (val)", None),
        (min_max_scale(st.nms, 0, 255), "2. non-maximum suppression", "gray"),
        (
            strong_weak,
            f"3. double threshold\nweak=blue strong=yellow\n(t_low={st.t_low:.1f}, t_high={st.t_high:.1f})",
            None,
        ),
        (st.edges, "3. hysteresis edge linking", "gray"),
    ]
    if sample.gt_edges is not None:
        panels.append((sample.gt_edges.astype(float), "ground-truth edges (synthetic)", "gray"))

    apply_style()
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (img, title, cmap) in zip(axes.ravel(), panels, strict=False):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        f"Task 3 -- Canny pipeline on '{sample.name}' ({sample.source})",
        fontsize=13,
        fontweight="bold",
    )
    path = save_figure(fig, cfg.out_dir / f"task3_canny_{cat}.png", dpi=cfg.dpi)

    m = evaluate_edges(st.edges, sample.gt_edges)
    return path, {
        "sample": sample.name,
        "source": sample.source,
        "t_low": st.t_low,
        "t_high": st.t_high,
        **m.as_dict(),
    }


def run(cfg: ExperimentConfig) -> list[Path]:
    banner("Task 3 -- first-principles Canny edge detection")
    outputs: list[Path] = []
    summary = {}
    for cat in ("pcb", "concrete"):
        path, metrics = _stage_figure(cfg, cat)
        outputs.append(path)
        summary[cat] = metrics
        p, r, f1 = metrics.get("precision"), metrics.get("recall"), metrics.get("f1")
        extra = f"  P={p:.3f} R={r:.3f} F1={f1:.3f}" if p is not None else ""
        print(
            f"  {cat:9s}: density={metrics['density']:.4f} "
            f"components={metrics['n_components']}{extra}"
        )

    # side-by-side final edge maps for the report hero figure
    apply_style()
    finals, titles = [], []
    for cat in ("pcb", "concrete"):
        s = get_samples(
            cat, n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
        )[0]
        finals += [s.image, canny(s.image, sigma=1.4, low_q=0.70, high_q=0.92)]
        titles += [f"{cat} input", f"{cat} Canny edges"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.5), constrained_layout=True)
    for ax, img, t in zip(axes.ravel(), finals, titles, strict=False):
        ax.imshow(img, cmap="gray")
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Task 3 -- final edge maps", fontsize=13, fontweight="bold")
    outputs.append(save_figure(fig, cfg.out_dir / "task3_canny_summary.png", dpi=cfg.dpi))

    outputs.append(dump_json(cfg, "task3_metrics", summary))
    print(f"  wrote {len(outputs)} artefacts to {cfg.out_dir}")
    return outputs


if __name__ == "__main__":  # pragma: no cover
    run(ExperimentConfig())
