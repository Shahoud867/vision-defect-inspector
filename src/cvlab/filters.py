"""Task 2 (part 1-2) -- Gaussian smoothing kernels and unsharp-mask sharpening.

Everything is built on :func:`cvlab.convolution.conv2d`; no OpenCV/SciPy filtering
primitives are used.
"""

from __future__ import annotations

import numpy as np

from .convolution import conv2d
from .utils import ensure_2d

__all__ = [
    "SOBEL_X",
    "SOBEL_Y",
    "generate_gaussian_kernel_1d",
    "generate_gaussian_kernel_2d",
    "gaussian_blur",
    "unsharp_mask",
    "default_gaussian_ksize",
]

# Normalised Sobel operators (brief section 2.3, Stage 1).  The 1/8 factor makes
# the response an estimate of the local intensity derivative per pixel.
SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0

SOBEL_Y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 8.0


def _as_ksize_pair(ksize: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(ksize, (int, np.integer)):
        kh = kw = int(ksize)
    else:
        kh, kw = int(ksize[0]), int(ksize[1])
    if kh <= 0 or kw <= 0:
        raise ValueError(f"ksize must be positive; got {ksize!r}")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"ksize must be odd on both axes; got {(kh, kw)!r}")
    return kh, kw


def default_gaussian_ksize(sigma: float, truncate: float = 3.0) -> int:
    """Smallest odd kernel size that captures +/- ``truncate`` standard deviations."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    radius = int(np.ceil(truncate * sigma))
    return 2 * radius + 1


# --------------------------------------------------------------------------- #
# Gaussian kernels
# --------------------------------------------------------------------------- #
def generate_gaussian_kernel_2d(ksize: int | tuple[int, int], sigma: float) -> np.ndarray:
    r"""Isotropic 2-D Gaussian kernel, normalised so that :math:`\sum G = 1`.

    .. math::
        G_\sigma(x, y) = \frac{1}{2\pi\sigma^2}
            \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)

    The coordinate grid is centred on the middle tap via :func:`numpy.meshgrid`.
    The closed-form normaliser is applied for numerical fidelity to the formula,
    then a final exact renormalisation removes the truncation error introduced by
    the finite window.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    kh, kw = _as_ksize_pair(ksize)

    ay = (kh - 1) / 2.0
    ax = (kw - 1) / 2.0
    ys, xs = np.meshgrid(np.arange(kh) - ay, np.arange(kw) - ax, indexing="ij")

    norm = 1.0 / (2.0 * np.pi * sigma**2)
    kernel = norm * np.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()  # exact unity gain despite finite support
    return kernel


def generate_gaussian_kernel_1d(ksize: int, sigma: float) -> np.ndarray:
    """1-D Gaussian profile (unit sum).  Enables separable :math:`O(k)` blur."""
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    k, _ = _as_ksize_pair(ksize)
    xs = np.arange(k) - (k - 1) / 2.0
    line = np.exp(-(xs**2) / (2.0 * sigma**2))
    line /= line.sum()
    return line


# --------------------------------------------------------------------------- #
# Blur & sharpen
# --------------------------------------------------------------------------- #
def gaussian_blur(
    image: np.ndarray,
    sigma: float,
    ksize: int | tuple[int, int] | None = None,
    pad_mode: str = "reflect",
    separable: bool = True,
) -> np.ndarray:
    """Low-pass filter ``image`` with an isotropic Gaussian of scale ``sigma``.

    ``pad_mode='reflect'`` (rather than zero padding) is used by default so that
    borders are not artificially darkened -- a deliberate, documented deviation
    from the bare skeleton that materially improves edge maps near the frame.
    When ``separable`` and ``ksize`` is square the 2-D convolution is replaced by
    two cheaper 1-D passes (identical result up to floating-point rounding).
    """
    img = ensure_2d(image)
    if ksize is None:
        kh = kw = default_gaussian_ksize(sigma)
    else:
        kh, kw = _as_ksize_pair(ksize)

    if separable and kh == kw:
        line = generate_gaussian_kernel_1d(kh, sigma)
        tmp = conv2d(img, line[None, :], padding="same", pad_mode=pad_mode)
        return conv2d(tmp, line[:, None], padding="same", pad_mode=pad_mode)

    kernel = generate_gaussian_kernel_2d((kh, kw), sigma)
    return conv2d(img, kernel, padding="same", pad_mode=pad_mode)


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.5,
    alpha: float = 1.2,
    ksize: int | tuple[int, int] | None = None,
    pad_mode: str = "reflect",
) -> np.ndarray:
    r"""High-pass detail accentuation.

    .. math::
        F_{sharp} = F + \alpha \, (F - F * G_\sigma)

    The output is clipped to ``[0, 255]`` and returned as ``float64`` (cast to
    ``uint8`` at save time).  ``alpha = 0`` reproduces the input; larger ``alpha``
    boosts fine structure such as micro-cracks and PCB trace edges.
    """
    img = ensure_2d(image)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    blurred = gaussian_blur(img, sigma=sigma, ksize=ksize, pad_mode=pad_mode)
    detail = img - blurred
    sharpened = img + alpha * detail
    return np.clip(sharpened, 0.0, 255.0)
