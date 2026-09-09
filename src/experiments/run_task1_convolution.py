"""Task 1 -- demonstrate and verify the 2-D / 3-D convolution engines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cvlab import conv2d, conv3d
from cvlab.datasets import get_samples
from cvlab.utils import compute_output_size, min_max_scale
from cvlab.viz import save_figure, show_grid

from ._common import ExperimentConfig, banner, dump_json

# A small zoo of hand-designed kernels exercised by the 2-D engine.
KERNELS: dict[str, np.ndarray] = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
    "box_blur_3x3": np.ones((3, 3)) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0,
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
}


def _shape_verification_table() -> list[dict]:
    """Cross-check :func:`conv2d` output shapes against the discrete formula."""
    cases = [
        # (H, W, F, stride, padding)
        (32, 32, 3, 1, "same"),
        (32, 32, 3, 1, "valid"),
        (32, 32, 5, 1, "valid"),
        (32, 32, 3, 2, "valid"),
        (31, 45, 3, 2, "same"),
        (64, 64, 7, 3, "valid"),
        (28, 28, 5, 2, "same"),
    ]
    rows = []
    rng = np.random.default_rng(0)
    for H, W, F, s, pad in cases:
        img = rng.random((H, W))
        k = rng.random((F, F))
        out = conv2d(img, k, stride=s, padding=pad)
        p = (F - 1) // 2 if pad == "same" else 0
        exp = (compute_output_size(H, F, p, s), compute_output_size(W, F, p, s))
        rows.append(
            {
                "H": H,
                "W": W,
                "F": F,
                "stride": s,
                "padding": pad,
                "pad_width": p,
                "formula": list(exp),
                "actual": list(out.shape),
                "match": bool(tuple(out.shape) == exp),
            }
        )
    return rows


def run(cfg: ExperimentConfig) -> list[Path]:
    banner("Task 1 -- 2D & 3D convolution engines")
    outputs: list[Path] = []

    # ---- shape verification ------------------------------------------------ #
    rows = _shape_verification_table()
    all_ok = all(r["match"] for r in rows)
    for r in rows:
        print(
            f"  {r['H']}x{r['W']} * {r['F']}x{r['F']}  s={r['stride']} "
            f"pad={r['padding']:5s} -> formula {tuple(r['formula'])} "
            f"actual {tuple(r['actual'])}  {'OK' if r['match'] else 'FAIL'}"
        )
    print(f"  shape formula agreement: {'ALL PASS' if all_ok else 'MISMATCH'}")
    outputs.append(dump_json(cfg, "task1_shape_verification", {"cases": rows, "all_pass": all_ok}))

    # ---- 2-D kernel gallery --------------------------------------------------#
    sample = get_samples(
        "pcb", n=1, size=cfg.image_size, seed=cfg.seed, data_dir=cfg.data_dir, prefer=cfg.prefer
    )[0]
    gray = sample.image
    imgs = [gray]
    titles = [f"input ({sample.source})"]
    for name, k in KERNELS.items():
        conv = conv2d(gray, k, padding="same")
        imgs.append(min_max_scale(conv, 0, 255))
        titles.append(f"conv2d: {name}")
    fig = show_grid(
        imgs, titles, ncols=4, suptitle="Task 1.1 -- conv2d kernel bank (padding='same')"
    )
    outputs.append(save_figure(fig, cfg.out_dir / "task1_conv2d_kernels.png", dpi=cfg.dpi))

    # ---- strided down-sampling -------------------------------------------- #
    box = KERNELS["box_blur_3x3"]
    strided = [gray]
    st_titles = [f"input {gray.shape}"]
    for s in (1, 2, 3, 4):
        out = conv2d(gray, box, stride=s, padding="valid")
        strided.append(min_max_scale(out, 0, 255))
        st_titles.append(f"stride={s}  ->  {out.shape}")
    fig = show_grid(
        strided,
        st_titles,
        ncols=5,
        suptitle="Task 1.1 -- strided convolution as anti-aliased down-sampling",
    )
    outputs.append(save_figure(fig, cfg.out_dir / "task1_conv2d_strided.png", dpi=cfg.dpi))

    # ---- 3-D multi-channel layer ---------------------------------------- #
    color = get_samples(
        "concrete",
        n=1,
        size=cfg.image_size,
        seed=cfg.seed,
        data_dir=cfg.data_dir,
        prefer=cfg.prefer,
    )[0].image
    rgb = np.stack([color, np.roll(color, 5, axis=1), np.roll(color, 5, axis=0)], axis=-1)

    def rgb_kernel(spatial: np.ndarray) -> np.ndarray:
        return np.stack([spatial, spatial, spatial], axis=-1) / 3.0

    stack = np.stack(
        [
            rgb_kernel(KERNELS["box_blur_3x3"] * 9 / 9),
            rgb_kernel(KERNELS["sobel_x"]),
            rgb_kernel(KERNELS["sobel_x"].T),
            rgb_kernel(KERNELS["laplacian"]),
        ],
        axis=0,
    )
    activation = conv3d(rgb, stack, padding="same")  # (H, W, 4)
    panels = [min_max_scale(rgb, 0, 255).astype(np.uint8)] + [
        min_max_scale(activation[..., k], 0, 255) for k in range(activation.shape[-1])
    ]
    p_titles = [
        "RGB input (H,W,3)",
        "filter 0: blur",
        "filter 1: d/dx",
        "filter 2: d/dy",
        "filter 3: Laplacian",
    ]
    fig = show_grid(
        panels,
        p_titles,
        ncols=5,
        suptitle=f"Task 1.2 -- conv3d activation volume {activation.shape}",
    )
    outputs.append(save_figure(fig, cfg.out_dir / "task1_conv3d_activations.png", dpi=cfg.dpi))

    print(f"  wrote {len(outputs)} artefacts to {cfg.out_dir}")
    return outputs


if __name__ == "__main__":  # pragma: no cover
    run(ExperimentConfig())
