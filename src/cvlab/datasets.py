"""Sample-image provider with a deterministic synthetic fallback.

The assignment benchmarks are hosted on Kaggle (PCB defects, concrete surface
cracks).  Downloading them requires credentials, so the pipeline must *also* run
end-to-end with zero external assets.  :func:`get_samples` therefore:

1. loads real images from ``data/raw/<category>/`` when present, else
2. synthesises procedurally-generated stand-ins whose statistics mimic the real
   data (high-contrast orthogonal traces for PCB; illuminated, textured concrete
   with thin dark cracks).  Every synthetic image ships a ground-truth edge/crack
   mask, enabling genuine precision/recall evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .utils import load_gray

__all__ = ["Sample", "PROJECT_ROOT", "DATA_DIR", "get_samples", "list_synthetic"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    """A grayscale image plus optional ground truth and provenance."""

    name: str
    image: np.ndarray  # float64 (H, W) in [0, 255]
    category: str
    source: str  # 'real' | 'synthetic'
    gt_edges: np.ndarray | None = None  # bool (H, W) when source == 'synthetic'


# --------------------------------------------------------------------------- #
# Procedural texture primitives
# --------------------------------------------------------------------------- #
def _value_noise(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Cheap fractal value noise in [0, 1] via up-sampled white-noise octaves."""
    H, W = shape
    field = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        step = max(1, 2 ** (octaves - o))
        gh, gw = H // step + 2, W // step + 2
        coarse = rng.random((gh, gw))
        ys = np.linspace(0, gh - 1, H)
        xs = np.linspace(0, gw - 1, W)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        c00 = coarse[np.ix_(y0, x0)]
        c01 = coarse[np.ix_(y0, np.minimum(x0 + 1, gw - 1))]
        c10 = coarse[np.ix_(np.minimum(y0 + 1, gh - 1), x0)]
        c11 = coarse[np.ix_(np.minimum(y0 + 1, gh - 1), np.minimum(x0 + 1, gw - 1))]
        bilinear = (
            c00 * (1 - fy) * (1 - fx) + c01 * (1 - fy) * fx + c10 * fy * (1 - fx) + c11 * fy * fx
        )
        field += amp * bilinear
        total += amp
        amp *= 0.5
    return field / total


def _segment_distance(shape: tuple[int, int], p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the segment ``p0 -> p1``."""
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W]
    pts = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float64)
    d = p1 - p0
    denom = float(d @ d) or 1.0
    t = np.clip((pts - p0) @ d / denom, 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * d[None, :]
    return np.linalg.norm(pts - proj, axis=1).reshape(H, W)


def _draw_polyline(shape: tuple[int, int], vertices: np.ndarray, width: float) -> np.ndarray:
    """Soft-edged intensity profile (0..1) of a poly-line of given half-width."""
    dist = np.full(shape, np.inf)
    for a, b in pairwise(vertices):
        dist = np.minimum(dist, _segment_distance(shape, a, b))
    return np.exp(-(dist**2) / (2.0 * width**2))


def _crack_vertices(shape: tuple[int, int], rng: np.random.Generator, n_pts: int = 9) -> np.ndarray:
    """Random meandering path spanning the image (a plausible crack skeleton)."""
    H, W = shape
    if rng.random() < 0.5:
        ys = np.linspace(rng.uniform(0, H * 0.2), rng.uniform(H * 0.8, H), n_pts)
        xs = np.linspace(rng.uniform(0, W), rng.uniform(0, W), n_pts)
        xs += rng.normal(0, W * 0.06, n_pts)
    else:
        xs = np.linspace(rng.uniform(0, W * 0.2), rng.uniform(W * 0.8, W), n_pts)
        ys = np.linspace(rng.uniform(0, H), rng.uniform(0, H), n_pts)
        ys += rng.normal(0, H * 0.06, n_pts)
    xs = np.clip(xs, 1, W - 2)
    ys = np.clip(ys, 1, H - 2)
    return np.stack([ys, xs], axis=1)


# --------------------------------------------------------------------------- #
# Synthetic scenes
# --------------------------------------------------------------------------- #
def synthetic_pcb(size: int = 256, seed: int = 0) -> Sample:
    """High-contrast circuit-board mock-up with a few manufacturing defects."""
    rng = np.random.default_rng(seed)
    H = W = size
    img = np.full((H, W), 30.0)  # dark substrate
    gt = np.zeros((H, W), dtype=bool)

    def band(r0: int, r1: int, c0: int, c1: int) -> None:
        img[r0:r1, c0:c1] = 210.0
        gt[r0 - 1 : r0 + 1, c0:c1] = True
        gt[r1 - 1 : r1 + 1, c0:c1] = True
        gt[r0:r1, c0 - 1 : c0 + 1] = True
        gt[r0:r1, c1 - 1 : c1 + 1] = True

    trace_w = max(4, size // 32)
    for c in range(size // 8, W - size // 8, size // 6):
        band(size // 10, H - size // 10, c, c + trace_w)
    for r in range(size // 6, H - size // 6, size // 5):
        band(r, r + trace_w, size // 10, W - size // 10)

    # solder pads
    for _ in range(6):
        pr = rng.integers(size // 8, H - size // 8)
        pc = rng.integers(size // 8, W - size // 8)
        rr = size // 20
        ys, xs = np.mgrid[0:H, 0:W]
        disk = (ys - pr) ** 2 + (xs - pc) ** 2 <= rr**2
        img[disk] = 235.0
        ring = ((ys - pr) ** 2 + (xs - pc) ** 2 <= (rr + 1) ** 2) & ~(
            (ys - pr) ** 2 + (xs - pc) ** 2 <= (rr - 1) ** 2
        )
        gt |= ring

    # defects: an open circuit (gap) and a spur
    img[H // 2 - trace_w : H // 2 + trace_w, W // 3 : W // 3 + trace_w] = 30.0
    img[size // 6 : size // 6 + trace_w // 2, W // 2 : W // 2 + size // 8] = 210.0

    img += rng.normal(0, 1.5, (H, W))  # mild sensor grain
    return Sample("pcb_synth", np.clip(img, 0, 255), "pcb", "synthetic", gt)


def synthetic_concrete(
    size: int = 256,
    seed: int = 0,
    n_cracks: int = 2,
    shadow: bool = False,
    heavy_texture: bool = False,
    crack_strength: float = 1.0,
) -> Sample:
    """Illuminated concrete surface with fractal texture and thin dark cracks.

    ``crack_strength`` scales crack contrast; the crack depth is additionally
    modulated by a smooth low-frequency field so that a single crack contains
    both strong and faint stretches (this is what makes hysteresis linking beat
    a single global threshold in the Task 4 ablation).
    """
    rng = np.random.default_rng(seed)
    H = W = size

    texture = _value_noise((H, W), rng, octaves=5 if heavy_texture else 4)
    base = 150.0 + 40.0 * (texture - 0.5) * (2.2 if heavy_texture else 1.0)

    ys, xs = np.mgrid[0:H, 0:W]
    illum = 1.0 - 0.45 * (((xs - W * 0.35) ** 2 + (ys - H * 0.25) ** 2) / (H * W))
    img = base * illum

    if shadow:  # hard cast-shadow contour -> a strong non-crack edge
        line = xs > (0.6 * W + 0.25 * ys)
        img[line] *= 0.55

    gt = np.zeros((H, W), dtype=bool)
    for _ in range(n_cracks):
        verts = _crack_vertices((H, W), rng, n_pts=10)
        profile = _draw_polyline((H, W), verts, width=rng.uniform(0.9, 1.6))
        depth_mod = 0.25 + 0.75 * _value_noise((H, W), rng, octaves=2)
        img -= profile * rng.uniform(60, 95) * crack_strength * depth_mod
        gt |= profile > 0.35

    img += rng.normal(0, 2.5, (H, W))
    name = "concrete_synth" + ("_shadow" if shadow else "") + ("_texture" if heavy_texture else "")
    return Sample(name, np.clip(img, 0, 255), "concrete", "synthetic", gt)


_SYNTH_BUILDERS = {
    "pcb": lambda size, seed: synthetic_pcb(size, seed),
    "concrete": lambda size, seed: synthetic_concrete(size, seed),
    "concrete_shadow": lambda size, seed: synthetic_concrete(size, seed, n_cracks=1, shadow=True),
    "concrete_texture": lambda size, seed: synthetic_concrete(
        size, seed, n_cracks=1, heavy_texture=True
    ),
}


def list_synthetic() -> list[str]:
    return list(_SYNTH_BUILDERS)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _load_real(category: str, n: int, size: int | None, data_dir: Path) -> list[Sample]:
    folder = data_dir / category
    if not folder.is_dir():
        return []
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)[:n]
    out: list[Sample] = []
    for p in paths:
        img = load_gray(p)
        if size is not None:
            img = _center_crop_resize(img, size)
        out.append(Sample(p.stem, img, category, "real", None))
    return out


def _center_crop_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Square centre-crop then nearest-neighbour resample to ``size`` (no cv2)."""
    H, W = img.shape
    s = min(H, W)
    img = img[(H - s) // 2 : (H - s) // 2 + s, (W - s) // 2 : (W - s) // 2 + s]
    idx_y = np.linspace(0, s - 1, size).round().astype(int)
    idx_x = np.linspace(0, s - 1, size).round().astype(int)
    return img[np.ix_(idx_y, idx_x)]


def get_samples(
    category: str,
    n: int = 1,
    size: int | None = 256,
    seed: int = 0,
    data_dir: Path | str | None = None,
    prefer: str = "auto",
) -> list[Sample]:
    """Return up to ``n`` :class:`Sample` objects for ``category``.

    Parameters
    ----------
    category
        ``'pcb'``, ``'concrete'``, ``'concrete_shadow'`` or ``'concrete_texture'``.
    prefer
        ``'auto'`` (real if available else synthetic), ``'real'`` or ``'synthetic'``.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    real_key = category.split("_")[0]

    if prefer in ("auto", "real"):
        real = _load_real(real_key, n, size, data_dir)
        if real and (prefer == "real" or len(real) >= n):
            return real[:n]
        if real and prefer == "auto":
            # top up with synthetic
            synth = _synthesise(category, n - len(real), size or 256, seed)
            return (real + synth)[:n]
    if prefer == "real":
        raise FileNotFoundError(
            f"prefer='real' but no images found in {data_dir / real_key}. "
            "Run `python scripts/download_data.py` or drop images there."
        )
    return _synthesise(category, n, size or 256, seed)


def _synthesise(category: str, n: int, size: int, seed: int) -> list[Sample]:
    if n <= 0:
        return []
    builder = _SYNTH_BUILDERS.get(category)
    if builder is None:
        raise KeyError(f"Unknown synthetic category {category!r}; choose from {list_synthetic()}")
    samples = []
    for k in range(n):
        s = builder(size, seed + k)
        nm = f"{s.name}_{k:02d}" if n > 1 else s.name
        samples.append(Sample(nm, s.image, s.category, s.source, s.gt_edges))
    return samples
