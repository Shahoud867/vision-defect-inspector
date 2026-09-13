"""Task 4 -- synthetic sensor-noise models for the ablation study."""

from __future__ import annotations

import numpy as np

from .utils import ensure_2d

__all__ = ["add_gaussian_noise", "add_salt_pepper_noise"]


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    mean: float = 0.0,
    seed: int | np.random.Generator | None = None,
    clip: bool = True,
) -> np.ndarray:
    r"""Add i.i.d. additive Gaussian noise :math:`\mathcal{N}(\mu, \sigma_n^2)`.

    ``sigma`` and ``mean`` are expressed in 8-bit intensity units, so
    ``sigma=25`` perturbs a mid-grey pixel by roughly +/- 25 levels.  The result
    is ``float64``; set ``clip`` to keep it within ``[0, 255]``.
    """
    img = ensure_2d(image)
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    noisy = img + rng.normal(loc=mean, scale=sigma, size=img.shape)
    return np.clip(noisy, 0.0, 255.0) if clip else noisy


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Impulse (salt-and-pepper) corruption -- used for a Canny failure case."""
    img = ensure_2d(image).copy()
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1]; got {amount}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    u = rng.random(img.shape)
    img[u < amount * salt_vs_pepper] = 255.0
    img[u > 1.0 - amount * (1.0 - salt_vs_pepper)] = 0.0
    return img
