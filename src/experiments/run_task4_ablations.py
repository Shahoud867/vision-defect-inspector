"""Task 4 -- step-by-step visual ablation study.

Ablation 1: noise level vs. Gaussian pre-smoothing (gradient noise maps).
Ablation 2: gradient magnitude thresholding *before* vs. *after* NMS.
Ablation 3: single global threshold vs. double-threshold hysteresis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvlab import add_gaussian_noise, evaluate_edges
from cvlab.canny import (
    double_threshold,
    hysteresis,
    non_max_suppression,
    quantile_thresholds,
    sobel_gradients,
)
from cvlab.datasets import get_samples
from cvlab.utils import min_max_scale
from cvlab.viz import apply_style, save_figure

from ._common import ExperimentConfig, banner, dump_json

NOISE_SIGMAS = (10.0, 25.0, 40.0)


def _ablation1(cfg: ExperimentConfig) -> tuple[list[Path], dict]:
    sample = get_samples(
        "concrete",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer=cfg.prefer,
    )[0]
    clean = sample.image
    apply_style()

    # metrics over ALL three noise levels required by the brief
    stats, cache = {}, {}
    for row, sn in enumerate(NOISE_SIGMAS):
        noisy = add_gaussian_noise(clean, sigma=sn, seed=cfg.seed + row)
        _, _, mag_raw, _ = sobel_gradients(noisy, smooth_sigma=None)
        _, _, mag_smooth, _ = sobel_gradients(noisy, smooth_sigma=1.4)
        cache[sn] = (noisy, mag_raw, mag_smooth)
        stats[f"sigma_n={sn:g}"] = {
            "grad_std_raw": float(mag_raw.std()),
            "grad_std_smoothed": float(mag_smooth.std()),
            "noise_suppression_ratio": float(mag_raw.std() / (mag_smooth.std() + 1e-9)),
        }

    def _grid(sigmas: tuple[float, ...], path: str, dpi: int) -> Path:
        fig, axes = plt.subplots(
            len(sigmas),
            3,
            figsize=(9.6, 3.35 * len(sigmas)),
            constrained_layout=True,
            squeeze=False,
        )
        for row, sn in enumerate(sigmas):
            noisy, mag_raw, mag_smooth = cache[sn]
            panels = [
                (noisy, f"noisy input  sigma_n={sn:g}"),
                (
                    min_max_scale(mag_raw, 0, 255),
                    f"|grad| NO pre-smoothing\nstd={mag_raw.std():.2f}",
                ),
                (
                    min_max_scale(mag_smooth, 0, 255),
                    f"|grad| WITH Gaussian sigma=1.4\nstd={mag_smooth.std():.2f}",
                ),
            ]
            for col, (img, title) in enumerate(panels):
                axes[row, col].imshow(img, cmap="gray")
                axes[row, col].set_title(title)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        fig.suptitle(
            "Task 4 -- Ablation 1: noise and Gaussian pre-smoothing on gradients",
            fontsize=12,
            fontweight="bold",
        )
        return save_figure(fig, cfg.out_dir / path, dpi=dpi)

    compact = _grid(
        (NOISE_SIGMAS[0], NOISE_SIGMAS[-1]), "task4_ablation1_noise_smoothing.png", cfg.dpi
    )
    full = _grid(NOISE_SIGMAS, "task4_ablation1_noise_smoothing_full.png", cfg.dpi)
    return [compact, full], stats


def _ablation2(cfg: ExperimentConfig) -> tuple[Path, dict]:
    sample = get_samples(
        "pcb", n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
    )[0]
    # deliberately over-smooth so the gradient forms visibly *thick* ridges,
    # making the 1-px thinning effect of NMS unmistakable.
    _, _, mag, ori = sobel_gradients(sample.image, smooth_sigma=2.2)
    nms = non_max_suppression(mag, ori)
    t = float(np.quantile(mag[mag > 0], 0.60))  # a single low threshold for both maps

    before = (mag >= t).astype(float)
    after = (nms >= t).astype(float)
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.0), constrained_layout=True)
    for ax, img, title in zip(
        axes,
        [min_max_scale(mag, 0, 255), before, after, min_max_scale(nms, 0, 255)],
        [
            "gradient magnitude |M|",
            f"|M| >= t  (BEFORE NMS)\nthick bands, density={before.mean():.3f}",
            f"NMS(|M|) >= t  (AFTER NMS)\n1-px ridges, density={after.mean():.3f}",
            "NMS response",
        ],
        strict=False,
    ):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Task 4 -- Ablation 2: non-maximum suppression thins gradient ridges to 1 px",
        fontsize=12,
        fontweight="bold",
    )
    stats = {
        "threshold": t,
        "edge_density_before_nms": float(before.mean()),
        "edge_density_after_nms": float(after.mean()),
        "thinning_ratio": float(before.mean() / (after.mean() + 1e-9)),
    }
    return save_figure(fig, cfg.out_dir / "task4_ablation2_nms.png", dpi=cfg.dpi), stats


def _ablation3(cfg: ExperimentConfig) -> tuple[Path, dict]:
    # a faint, contrast-varying crack: some stretches are strong, some barely
    # above the noise floor -> exactly where hysteresis linking earns its keep.
    from cvlab.datasets import synthetic_concrete

    if cfg.prefer == "real" and cfg.data_dir is not None:
        sample = get_samples(
            "concrete",
            n=1,
            size=cfg.image_size,
            seed=cfg.seed + 3,
            data_dir=cfg.data_dir,
            prefer="real",
        )[0]
    else:
        sample = synthetic_concrete(
            cfg.image_size, seed=cfg.seed + 3, n_cracks=1, crack_strength=0.6
        )
    noisy = add_gaussian_noise(sample.image, sigma=12.0, seed=cfg.seed)
    _, _, mag, ori = sobel_gradients(noisy, smooth_sigma=1.4)
    nms = non_max_suppression(mag, ori)
    t_lo, t_hi = quantile_thresholds(nms, 0.94, 0.98)

    single_low = (nms >= t_lo).astype(np.uint8) * 255
    single_high = (nms >= t_hi).astype(np.uint8) * 255
    strong, weak = double_threshold(nms, t_lo, t_hi)
    linked = hysteresis(strong, weak)

    gt = sample.gt_edges
    m_low = evaluate_edges(single_low, gt)
    m_high = evaluate_edges(single_high, gt)
    m_hyst = evaluate_edges(linked, gt)

    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.0), constrained_layout=True)
    imgs = [noisy, single_low, single_high, linked]
    titles = [
        "noisy concrete (faint crack)",
        f"single threshold @ t_low\nF1={_f(m_low.f1)}  R={_f(m_low.recall)}  {m_low.n_components} parts",
        f"single threshold @ t_high\nF1={_f(m_high.f1)}  R={_f(m_high.recall)}  {m_high.n_components} parts",
        f"double-threshold hysteresis\nF1={_f(m_hyst.f1)}  R={_f(m_hyst.recall)}  {m_hyst.n_components} parts",
    ]
    for ax, img, title in zip(axes, imgs, titles, strict=False):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Task 4 -- Ablation 3: hysteresis links weak crack pixels without background clutter",
        fontsize=12,
        fontweight="bold",
    )
    stats = {
        "single_low": m_low.as_dict(),
        "single_high": m_high.as_dict(),
        "hysteresis": m_hyst.as_dict(),
        "t_low": t_lo,
        "t_high": t_hi,
    }
    return save_figure(fig, cfg.out_dir / "task4_ablation3_hysteresis.png", dpi=cfg.dpi), stats


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def run(cfg: ExperimentConfig) -> list[Path]:
    banner("Task 4 -- visual ablation study")
    outputs, summary = [], {}
    for name, fn in (
        ("ablation1", _ablation1),
        ("ablation2", _ablation2),
        ("ablation3", _ablation3),
    ):
        paths, stats = fn(cfg)
        outputs.extend(paths if isinstance(paths, list) else [paths])
        summary[name] = stats
        print(f"  {name}: {list(stats.keys())}")
    outputs.append(dump_json(cfg, "task4_metrics", summary))
    print(f"  wrote {len(outputs)} artefacts to {cfg.out_dir}")
    return outputs


if __name__ == "__main__":  # pragma: no cover
    run(ExperimentConfig())
