"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20232515)


@pytest.fixture
def checker_8x8() -> np.ndarray:
    """8x8 checkerboard of 0 / 255 blocks (2x2 cells)."""
    base = np.indices((8, 8)).sum(axis=0) % 2
    return (base * 255).astype(np.float64)


@pytest.fixture
def vertical_step() -> np.ndarray:
    """64x64 image, left half 0, right half 255 (a single vertical edge at col 32)."""
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 255.0
    return img


@pytest.fixture
def disk() -> np.ndarray:
    """96x96 image with a filled bright disk of radius 24 on a dark ground."""
    yy, xx = np.mgrid[0:96, 0:96]
    m = (yy - 48) ** 2 + (xx - 48) ** 2 <= 24**2
    return np.where(m, 220.0, 20.0)
