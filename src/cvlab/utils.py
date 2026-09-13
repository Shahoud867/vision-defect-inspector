"""Shared utilities: type validation, padding, normalisation and image I/O.

Only :mod:`numpy` is used for numerical work.  Image file I/O optionally uses
OpenCV, and *strictly* only ``cv2.imread`` / ``cv2.imwrite`` / ``cv2.cvtColor``
as permitted by the assignment brief.  If OpenCV is unavailable the module falls
back to :mod:`matplotlib.image`, so the rest of the library never hard-depends on
``cv2``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "PadMode",
    "as_float",
    "ensure_2d",
    "ensure_3d",
    "compute_output_size",
    "resolve_padding",
    "pad_array",
    "normalize_to_uint8",
    "min_max_scale",
    "to_grayscale",
    "load_gray",
    "load_color",
    "save_image",
]

PadMode = Literal["constant", "reflect", "replicate", "wrap"]
_NpPadMode = Literal["constant", "reflect", "edge", "wrap"]

_NP_PAD_ALIASES: dict[str, _NpPadMode] = {
    "constant": "constant",
    "zero": "constant",
    "zeros": "constant",
    "reflect": "reflect",
    "replicate": "edge",
    "edge": "edge",
    "wrap": "wrap",
}


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def as_float(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as contiguous ``float64`` without copying when possible."""
    return np.ascontiguousarray(array, dtype=np.float64)


def ensure_2d(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate that ``image`` is a 2-D array and return it as ``float64``."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (H, W); got shape {arr.shape!r}")
    if 0 in arr.shape:
        raise ValueError(f"{name} must be non-empty; got shape {arr.shape!r}")
    return as_float(arr)


def ensure_3d(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate that ``image`` is a 3-D array ``(H, W, C)`` and return ``float64``.

    A 2-D array is promoted to a single-channel 3-D array for convenience.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2-D or 3-D (H, W, C); got shape {arr.shape!r}")
    if 0 in arr.shape:
        raise ValueError(f"{name} must be non-empty; got shape {arr.shape!r}")
    return as_float(arr)


# --------------------------------------------------------------------------- #
# Spatial size arithmetic
# --------------------------------------------------------------------------- #
def compute_output_size(in_size: int, filter_size: int, pad: int, stride: int) -> int:
    r"""Discrete spatial output size.

    .. math::
        O = \left\lfloor \frac{I - F + 2P}{S} \right\rfloor + 1
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")
    if filter_size < 1:
        raise ValueError(f"filter_size must be >= 1; got {filter_size}")
    if pad < 0:
        raise ValueError(f"pad must be >= 0; got {pad}")
    numerator = in_size - filter_size + 2 * pad
    if numerator < 0:
        raise ValueError(
            f"Invalid geometry: input={in_size}, filter={filter_size}, pad={pad}. "
            "Filter (minus padding) is larger than the input."
        )
    return numerator // stride + 1


def resolve_padding(
    padding: str | int | tuple[int, int],
    filter_h: int,
    filter_w: int,
) -> tuple[int, int]:
    """Translate a ``padding`` argument into explicit ``(pad_h, pad_w)`` widths.

    Accepts the string modes ``'same'`` / ``'valid'`` (assignment API) as well as
    an integer or an explicit ``(pad_h, pad_w)`` tuple (production convenience).
    ``'same'`` yields an identical output size **only when the stride is 1**, per
    the discrete convolution formula.
    """
    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "same":
            if filter_h % 2 == 0 or filter_w % 2 == 0:
                raise ValueError("'same' padding requires odd kernel dimensions")
            return (filter_h - 1) // 2, (filter_w - 1) // 2
        if mode == "valid":
            return 0, 0
        raise ValueError(f"padding string must be 'same' or 'valid'; got {padding!r}")
    if isinstance(padding, (int, np.integer)):
        if padding < 0:
            raise ValueError(f"integer padding must be >= 0; got {padding}")
        return int(padding), int(padding)
    if isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
        if ph < 0 or pw < 0:
            raise ValueError(f"padding widths must be >= 0; got {padding!r}")
        return int(ph), int(pw)
    raise TypeError(f"Unsupported padding specification: {padding!r}")


def pad_array(
    array: np.ndarray,
    pad_h: int,
    pad_w: int,
    mode: PadMode | str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """Pad the first two axes of ``array`` by ``pad_h`` / ``pad_w`` on both sides.

    ``mode`` accepts ``'constant'`` / ``'reflect'`` / ``'replicate'`` / ``'wrap'``
    (and the aliases ``'zero'``, ``'zeros'``, ``'edge'``).
    """
    if pad_h == 0 and pad_w == 0:
        return array
    np_mode = _NP_PAD_ALIASES.get(mode)
    if np_mode is None:
        raise ValueError(f"Unknown pad mode {mode!r}; expected one of {list(_NP_PAD_ALIASES)}")
    width = [(pad_h, pad_h), (pad_w, pad_w), *([(0, 0)] * (array.ndim - 2))]
    if np_mode == "constant":
        return np.pad(array, width, mode="constant", constant_values=value)
    return np.pad(array, width, mode=np_mode)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def min_max_scale(array: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Linearly rescale ``array`` so its min/max map to ``lo`` / ``hi``.

    A constant array is mapped to ``lo`` (avoids divide-by-zero).
    """
    arr = as_float(array)
    a_min = float(arr.min())
    a_max = float(arr.max())
    if a_max - a_min < np.finfo(np.float64).eps:
        return np.full_like(arr, lo)
    return (arr - a_min) / (a_max - a_min) * (hi - lo) + lo


def normalize_to_uint8(
    array: np.ndarray,
    mode: Literal["minmax", "clip"] = "minmax",
) -> np.ndarray:
    """Map a real-valued array to ``uint8`` in ``[0, 255]``.

    ``minmax``
        Full-range stretch of ``[min, max]`` to ``[0, 255]``.  Use for signed
        maps such as gradients where relative structure matters.
    ``clip``
        Clip to ``[0, 255]`` and round.  Use when values are already in intensity
        units and absolute brightness must be preserved.
    """
    arr = as_float(array)
    if mode == "minmax":
        scaled = min_max_scale(arr, 0.0, 255.0)
    elif mode == "clip":
        scaled = np.clip(arr, 0.0, 255.0)
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"mode must be 'minmax' or 'clip'; got {mode!r}")
    return np.rint(scaled).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Colour conversion
# --------------------------------------------------------------------------- #
def to_grayscale(
    image: np.ndarray, weights: tuple[float, float, float] | None = None
) -> np.ndarray:
    """Convert an RGB(A) image to single-channel grayscale (ITU-R BT.601 luma).

    Implemented directly in NumPy; ``cv2.cvtColor`` would be permitted but this
    keeps the core pipeline free of any OpenCV dependency.  A 2-D input is
    returned unchanged (as ``float64``).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return as_float(arr)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        w = np.array(weights if weights is not None else (0.299, 0.587, 0.114), dtype=np.float64)
        return as_float(arr[..., :3]) @ w
    if arr.ndim == 3 and arr.shape[2] == 1:
        return as_float(arr[..., 0])
    raise ValueError(f"Cannot convert image of shape {arr.shape!r} to grayscale")


# --------------------------------------------------------------------------- #
# Image file I/O  (cv2.imread / cv2.imwrite only; matplotlib fallback)
# --------------------------------------------------------------------------- #
def _imread(path: Path, *, color: bool) -> np.ndarray:
    try:
        import cv2

        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
        raw = cv2.imread(str(path), flag)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if color:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return raw
    except ModuleNotFoundError:
        import matplotlib.image as mpimg

        raw = mpimg.imread(str(path))
        # matplotlib returns float in [0, 1] for 8-bit PNGs, uint8 for some JPEGs.
        if np.issubdtype(raw.dtype, np.floating):
            raw = np.rint(raw * 255.0).astype(np.uint8)
        if not color and raw.ndim == 3:
            raw = np.rint(to_grayscale(raw)).astype(np.uint8)
        if color and raw.ndim == 2:
            raw = np.repeat(raw[:, :, None], 3, axis=2)
        if color and raw.ndim == 3 and raw.shape[2] == 4:
            raw = raw[:, :, :3]
        return raw


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image from disk as a ``float64`` grayscale array in ``[0, 255]``."""
    return ensure_2d(_imread(Path(path), color=False))


def load_color(path: str | Path) -> np.ndarray:
    """Read an image from disk as a ``float64`` RGB array ``(H, W, 3)`` in ``[0, 255]``."""
    return ensure_3d(_imread(Path(path), color=True))


def save_image(path: str | Path, array: np.ndarray) -> Path:
    """Write ``array`` (any real range) to ``path`` as an 8-bit PNG/JPEG.

    Floating-point input is clipped to ``[0, 255]`` and rounded; use
    :func:`normalize_to_uint8` beforehand if a full-range stretch is desired.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = array if array.dtype == np.uint8 else normalize_to_uint8(array, mode="clip")
    try:
        import cv2

        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(out), data):
            raise OSError(f"cv2.imwrite failed for {out}")
    except ModuleNotFoundError:
        import matplotlib.image as mpimg

        if data.ndim == 2:
            # vmin/vmax pinned so the gray colormap does NOT rescale the data
            mpimg.imsave(str(out), data, cmap="gray", vmin=0, vmax=255, format="png")
        else:
            mpimg.imsave(str(out), data, format="png")
    return out
