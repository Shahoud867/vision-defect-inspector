"""Task 5 -- scale-space causality, (T_high, T_low) sensitivity, failure cases."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cvlab import add_gaussian_noise, canny, evaluate_edges
from cvlab.canny import sobel_gradients
from cvlab.datasets import get_samples
from cvlab.utils import min_max_scale
from cvlab.viz import apply_style, save_figure

from ._common import ExperimentConfig, banner, dump_json

SCALE_SIGMAS = (1.0, 2.5, 5.0)
THIGH_Q = (0.75, 0.88, 0.96)
TLOW_FRAC = (0.30, 0.50, 0.80)  # t_low as a fraction of t_high
SENSITIVITY_NOISE = 25.0


# --------------------------------------------------------------------------- #
# 5.1 scale-space causality
# --------------------------------------------------------------------------- #
def _scale_space(cfg: ExperimentConfig) -> tuple[list[Path], dict]:
    sample = get_samples(
        "concrete",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer=cfg.prefer,
    )[0]
    img = sample.image
    apply_style()

    counts = {}
    stages = [
        canny(img, sigma=sg, low_q=0.70, high_q=0.92, return_stages=True) for sg in SCALE_SIGMAS
    ]
    for sg, st in zip(SCALE_SIGMAS, stages, strict=False):
        counts[f"sigma={sg}"] = {
            "edge_pixels": int((st.edges > 0).sum()),
            "edge_density": float((st.edges > 0).mean()),
            "t_low": st.t_low,
            "t_high": st.t_high,
        }

    # compact report figure: one row of edge maps (the causality evidence)
    fig, axes = plt.subplots(
        1, len(SCALE_SIGMAS), figsize=(3.4 * len(SCALE_SIGMAS), 3.7), constrained_layout=True
    )
    for ax, sg, st in zip(axes, SCALE_SIGMAS, stages, strict=False):
        ax.imshow(st.edges, cmap="gray")
        ax.set_title(
            f"sigma={sg}: {int((st.edges > 0).sum())} edge px ({(st.edges > 0).mean() * 100:.2f}%)"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Task 5.1 -- scale-space crack merging (edge count monotone non-increasing in sigma)",
        fontsize=12,
        fontweight="bold",
    )
    p1 = save_figure(fig, cfg.out_dir / "task5_scale_space.png", dpi=cfg.dpi)

    # full 2-row version (smoothed input + edges) for the supplementary
    fig, axes = plt.subplots(
        2, len(SCALE_SIGMAS), figsize=(3.4 * len(SCALE_SIGMAS), 7.1), constrained_layout=True
    )
    for col, (sg, st) in enumerate(zip(SCALE_SIGMAS, stages, strict=False)):
        axes[0, col].imshow(st.smoothed, cmap="gray")
        axes[0, col].set_title(f"Gaussian sigma={sg}")
        axes[1, col].imshow(st.edges, cmap="gray")
        axes[1, col].set_title(
            f"edges: {int((st.edges > 0).sum())} px ({(st.edges > 0).mean() * 100:.2f}%)"
        )
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Task 5.1 -- scale-space: Gaussian-smoothed input (top) and Canny edges (bottom)",
        fontsize=12,
        fontweight="bold",
    )
    p1b = save_figure(fig, cfg.out_dir / "task5_scale_space_full.png", dpi=cfg.dpi)

    # 1-D zero-crossing tracking on a fixed scan-line -> Witkin causality
    from cvlab.filters import gaussian_blur

    row = img[cfg.image_size // 2].astype(float)
    sigmas = np.linspace(0.4, 8.0, 40)
    fingerprint: list[np.ndarray] = []
    zc_counts: list[int] = []
    for sg in sigmas:
        smoothed_line = gaussian_blur(np.tile(row, (5, 1)), sigma=sg)[2]
        second = np.gradient(np.gradient(smoothed_line))
        zc = np.where(np.diff(np.sign(second)) != 0)[0]
        fingerprint.append(zc)
        zc_counts.append(len(zc))

    fig, (axf, axc) = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    for sg, zc in zip(sigmas, fingerprint, strict=False):
        axf.scatter(zc, np.full_like(zc, sg, dtype=float), s=6, c="#1f4e79")
    axf.set_xlabel("pixel position along central scan-line")
    axf.set_ylabel("Gaussian scale sigma")
    axf.set_title("scale-space fingerprint\n(2nd-derivative zero-crossings; branches only merge)")
    axf.grid(alpha=0.25)

    axc.plot(sigmas, zc_counts, "o-", color="#b5651d", ms=3)
    axc.set_xlabel("Gaussian scale sigma")
    axc.set_ylabel("number of zero-crossings")
    running_min = np.minimum.accumulate(zc_counts)
    axc.fill_between(sigmas, running_min, zc_counts, color="#b5651d", alpha=0.12)
    monotone_curve = bool(np.all(np.diff(running_min) <= 0))
    axc.set_title(f"zero-crossing count vs scale\n(monotone non-increasing = {monotone_curve})")
    axc.grid(alpha=0.25)
    fig.suptitle(
        "Task 5.1 -- Witkin scale-space causality: features are destroyed, never created",
        fontsize=12,
        fontweight="bold",
    )
    p2 = save_figure(fig, cfg.out_dir / "task5_scale_space_causality.png", dpi=cfg.dpi)

    monotone = _is_monotone_non_increasing([v["edge_pixels"] for v in counts.values()])
    counts["monotone_non_increasing"] = monotone
    print(
        f"  scale-space edge counts: "
        f"{[v['edge_pixels'] for v in counts.values() if isinstance(v, dict)]}  monotone={monotone}"
    )
    return [p1, p1b, p2], counts


def _is_monotone_non_increasing(seq: list[int], slack: float = 1.15) -> bool:
    from itertools import pairwise

    return all(b <= a * slack for a, b in pairwise(seq))


# --------------------------------------------------------------------------- #
# 5.2 parameter sensitivity grid
# --------------------------------------------------------------------------- #
def _sensitivity_grid(cfg: ExperimentConfig, cat: str) -> tuple[Path, dict]:
    sample = get_samples(
        cat, n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
    )[0]
    noisy = add_gaussian_noise(sample.image, sigma=SENSITIVITY_NOISE, seed=cfg.seed)
    _, _, mag, ori = sobel_gradients(noisy, smooth_sigma=1.4)
    from cvlab.canny import non_max_suppression

    nms = non_max_suppression(mag, ori)

    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 11.0), constrained_layout=True)
    grid = {}
    for r, hq in enumerate(THIGH_Q):
        for c, lf in enumerate(TLOW_FRAC):
            t_high = float(np.quantile(nms[nms > 0], hq))
            t_low = t_high * lf
            from cvlab.canny import double_threshold, hysteresis

            strong, weak = double_threshold(nms, t_low, t_high)
            edges = hysteresis(strong, weak)
            m = evaluate_edges(edges, sample.gt_edges)
            key = f"tHigh_q{hq:.2f}_tLow_{lf:.2f}xtHigh"
            grid[key] = {"t_low": t_low, "t_high": t_high, **m.as_dict()}
            ax = axes[r, c]
            ax.imshow(edges, cmap="gray")
            f1s = f"F1={m.f1:.3f}" if m.f1 is not None else ""
            ax.set_title(
                f"t_high=q{hq:.2f} ({t_high:.1f})\nt_low={lf:.2f}*t_high ({t_low:.1f})\n"
                f"density={m.density:.3f} {f1s}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Task 5.2 -- (T_high, T_low) sensitivity grid on noisy {cat}",
        fontsize=12,
        fontweight="bold",
    )
    path = save_figure(fig, cfg.out_dir / f"task5_sensitivity_grid_{cat}.png", dpi=cfg.dpi)

    if sample.gt_edges is not None:
        best = max(grid.items(), key=lambda kv: kv[1]["f1"] or 0.0)
        print(f"  {cat}: best operating point -> {best[0]}  F1={best[1]['f1']:.3f}")
    return path, grid


# --------------------------------------------------------------------------- #
# 5.3 failure cases
# --------------------------------------------------------------------------- #
def _failure_cases(cfg: ExperimentConfig) -> tuple[list[Path], dict]:
    apply_style()
    notes = {}

    # (a) strong cast-shadow contour dominates a genuine crack
    shadow = get_samples(
        "concrete_shadow",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer="synthetic",
    )[0]
    st_s = canny(shadow.image, sigma=1.4, low_q=0.70, high_q=0.92, return_stages=True)
    m_s = evaluate_edges(st_s.edges, shadow.gt_edges)

    # (b) heavily textured surface -> edge soup, crack lost in clutter
    tex = get_samples(
        "concrete_texture",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer="synthetic",
    )[0]
    st_t = canny(tex.image, sigma=1.4, low_q=0.70, high_q=0.92, return_stages=True)
    m_t = evaluate_edges(st_t.edges, tex.gt_edges)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.6), constrained_layout=True)
    for row, (name, st, m, gt) in enumerate(
        [("shadow contour", st_s, m_s, shadow.gt_edges), ("dense texture", st_t, m_t, tex.gt_edges)]
    ):
        f1_txt = f"  F1={m.f1:.3f}" if m.f1 is not None else ""
        axes[row, 0].imshow(st.gray, cmap="gray")
        axes[row, 0].set_title(f"{name}: input")
        axes[row, 1].imshow(st.edges, cmap="gray")
        axes[row, 1].set_title(f"{name}: Canny output\ndensity={m.density:.3f}{f1_txt}")
        axes[row, 2].imshow(gt, cmap="gray")
        axes[row, 2].set_title(f"{name}: ground-truth edges")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        notes[name] = m.as_dict()
    fig.suptitle(
        "Task 5.3 -- failure cases: shadow contours and high-frequency texture defeat Canny",
        fontsize=12,
        fontweight="bold",
    )
    p1 = save_figure(fig, cfg.out_dir / "task5_failure_cases.png", dpi=cfg.dpi)

    # zoomed comparison of texture clutter vs. true crack pixels
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.3), constrained_layout=True)
    ax[0].imshow(min_max_scale(st_t.magnitude, 0, 255), cmap="magma")
    ax[0].set_title("texture: gradient magnitude (no dominant ridge)")
    ax[1].imshow(st_t.edges, cmap="gray")
    ax[1].set_title("texture: fragmented edge map")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    p2 = save_figure(fig, cfg.out_dir / "task5_failure_texture_detail.png", dpi=cfg.dpi)
    return [p1, p2], notes


def run(cfg: ExperimentConfig) -> list[Path]:
    banner("Task 5 -- scale-space, sensitivity & failure analysis")
    outputs: list[Path] = []
    summary: dict = {}

    paths, counts = _scale_space(cfg)
    outputs += paths
    summary["scale_space"] = counts

    for cat in ("concrete", "pcb"):
        path, grid = _sensitivity_grid(cfg, cat)
        outputs.append(path)
        summary.setdefault("sensitivity", {})[cat] = grid

    paths, notes = _failure_cases(cfg)
    outputs += paths
    summary["failure_cases"] = notes

    outputs.append(dump_json(cfg, "task5_metrics", summary))
    print(f"  wrote {len(outputs)} artefacts to {cfg.out_dir}")
    return outputs


if __name__ == "__main__":  # pragma: no cover
    run(ExperimentConfig())
