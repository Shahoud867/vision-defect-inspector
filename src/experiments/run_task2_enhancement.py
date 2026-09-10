"""Task 2 -- Gaussian kernels, unsharp masking and 2-D pooling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvlab import (
    generate_gaussian_kernel_2d,
    max_pool_gradient,
    pool2d,
    unsharp_mask,
)
from cvlab.datasets import get_samples
from cvlab.pooling import average_pool_gradient
from cvlab.utils import min_max_scale
from cvlab.viz import apply_style, save_figure, show_grid

from ._common import ExperimentConfig, banner, dump_json

MAXPOOL_EXAMPLE = np.array([[8.0, 7.0], [12.0, 9.0]])


def _gaussian_panel(cfg: ExperimentConfig) -> Path:
    apply_style()
    combos = [(3, 0.8), (5, 1.0), (5, 1.5), (7, 2.0), (9, 3.0), (15, 5.0)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.0), constrained_layout=True)
    for ax, (ks, sg) in zip(axes.ravel(), combos, strict=False):
        k = generate_gaussian_kernel_2d(ks, sg)
        im = ax.imshow(k, cmap="viridis")
        ax.set_title(f"ksize={ks}, sigma={sg}\nsum={k.sum():.6f}  peak={k.max():.4f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Task 2.1 -- isotropic 2-D Gaussian kernels (unit gain)", fontsize=12, fontweight="bold"
    )
    return save_figure(fig, cfg.out_dir / "task2_gaussian_kernels.png", dpi=cfg.dpi)


def _unsharp_panel(cfg: ExperimentConfig) -> tuple[Path, dict]:
    metrics = {}
    rows = []
    titles = []
    for cat in ("pcb", "concrete"):
        sample = get_samples(
            cat, n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
        )[0]
        f = sample.image
        sharp = unsharp_mask(f, sigma=1.5, alpha=1.2)
        detail = min_max_scale(f - _blurhelp(f), 0, 255)
        rows += [f, detail, sharp]
        titles += [f"{cat}: F", f"{cat}: detail (F - F*G)", f"{cat}: F_sharp (a=1.2)"]
        metrics[cat] = {
            "input_std": float(f.std()),
            "sharpened_std": float(np.clip(sharp, 0, 255).std()),
            "high_freq_gain": float(np.clip(sharp, 0, 255).std() / (f.std() + 1e-9)),
        }
    fig = show_grid(
        rows,
        titles,
        ncols=3,
        suptitle="Task 2.2 -- unsharp masking (high-pass detail accentuation)",
    )
    return save_figure(fig, cfg.out_dir / "task2_unsharp_mask.png", dpi=cfg.dpi), metrics


def _blurhelp(f: np.ndarray) -> np.ndarray:
    from cvlab.filters import gaussian_blur

    return gaussian_blur(f, sigma=1.5)


def _alpha_sweep_panel(cfg: ExperimentConfig) -> Path:
    sample = get_samples(
        "concrete",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer=cfg.prefer,
    )[0]
    f = sample.image
    alphas = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]
    imgs = [unsharp_mask(f, sigma=1.5, alpha=a) for a in alphas]
    titles = [f"alpha = {a}" for a in alphas]
    fig = show_grid(
        imgs, titles, ncols=6, suptitle="Task 2.2 -- sharpening strength sweep (sigma=1.5)"
    )
    return save_figure(fig, cfg.out_dir / "task2_unsharp_alpha_sweep.png", dpi=cfg.dpi)


def _pooling_panel(cfg: ExperimentConfig) -> tuple[list[Path], dict]:
    from cvlab.canny import sobel_gradients

    sample = get_samples(
        "pcb", n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
    )[0]
    _, _, mag, _ = sobel_gradients(sample.image, smooth_sigma=1.0)
    feat = min_max_scale(mag, 0, 255)
    mx = pool2d(feat, (2, 2), 2, "max")
    av = pool2d(feat, (2, 2), 2, "average")
    mx4 = pool2d(feat, (4, 4), 4, "max")
    fig = show_grid(
        [feat, mx, av, mx4],
        [
            f"gradient feature map {feat.shape}",
            f"max-pool 2x2 s2  {mx.shape}",
            f"avg-pool 2x2 s2  {av.shape}",
            f"max-pool 4x4 s4  {mx4.shape}",
        ],
        ncols=4,
        suptitle="Task 2.3 -- feature subsampling via 2-D pooling",
    )
    pool_path = save_figure(fig, cfg.out_dir / "task2_pooling.png", dpi=cfg.dpi)

    # analytic max-pool gradient on the brief's 2x2 example
    routing = max_pool_gradient(MAXPOOL_EXAMPLE, (2, 2), 2)
    numeric = _numeric_maxpool_gradient(MAXPOOL_EXAMPLE)
    avg_routing = average_pool_gradient((2, 2), (2, 2), 2)
    payload = {
        "input": MAXPOOL_EXAMPLE.tolist(),
        "max_pool_output": float(pool2d(MAXPOOL_EXAMPLE, (2, 2), 2, "max")[0, 0]),
        "analytic_gradient": routing.tolist(),
        "numeric_gradient": numeric.tolist(),
        "gradients_match": bool(np.allclose(routing, numeric)),
        "average_pool_gradient": avg_routing.tolist(),
    }
    grad_path = _render_gradient_figure(cfg, routing, avg_routing)
    return [pool_path, grad_path], payload


def _numeric_maxpool_gradient(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy()
            xp[i, j] += eps
            xm = x.copy()
            xm[i, j] -= eps
            g[i, j] = (pool2d(xp, (2, 2), 2, "max")[0, 0] - pool2d(xm, (2, 2), 2, "max")[0, 0]) / (
                2 * eps
            )
    return g


def _render_gradient_figure(
    cfg: ExperimentConfig, routing: np.ndarray, avg_routing: np.ndarray
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        [MAXPOOL_EXAMPLE, routing, avg_routing],
        ["input block", "d(max)/d(input)", "d(avg)/d(input)"],
        strict=False,
    ):
        ax.imshow(data, cmap="magma")
        for (r, c), v in np.ndenumerate(data):
            ax.text(
                c,
                r,
                f"{v:g}",
                ha="center",
                va="center",
                color="white",
                fontsize=13,
                fontweight="bold",
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Task 2.3 -- analytical pooling gradients (2x2 block)", fontsize=12, fontweight="bold"
    )
    return save_figure(fig, cfg.out_dir / "task2_pooling_gradients.png", dpi=cfg.dpi)


def run(cfg: ExperimentConfig) -> list[Path]:
    banner("Task 2 -- enhancement & pooling")
    outputs = [_gaussian_panel(cfg)]

    unsharp_path, unsharp_metrics = _unsharp_panel(cfg)
    outputs.append(unsharp_path)
    outputs.append(_alpha_sweep_panel(cfg))

    pool_paths, pool_payload = _pooling_panel(cfg)
    outputs.extend(pool_paths)

    print("  max-pool gradient (brief example [[8,7],[12,9]]):")
    print("   ", np.array2string(np.array(pool_payload["analytic_gradient"]), prefix="    "))
    print(f"    matches finite-difference check: {pool_payload['gradients_match']}")

    outputs.append(
        dump_json(
            cfg,
            "task2_metrics",
            {
                "unsharp": unsharp_metrics,
                "pooling": pool_payload,
            },
        )
    )
    print(f"  wrote {len(outputs)} artefacts to {cfg.out_dir}")
    return outputs


if __name__ == "__main__":  # pragma: no cover
    run(ExperimentConfig())
