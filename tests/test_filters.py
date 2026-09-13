"""Task 2 (part 1-2) -- Gaussian kernels, blur and unsharp masking."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import (
    SOBEL_X,
    SOBEL_Y,
    conv2d,
    gaussian_blur,
    generate_gaussian_kernel_1d,
    generate_gaussian_kernel_2d,
    unsharp_mask,
)


@pytest.mark.parametrize("ksize", [3, 5, 7, 9, 15])
@pytest.mark.parametrize("sigma", [0.6, 1.0, 2.0, 3.5])
def test_gaussian_kernel_unit_sum(ksize, sigma):
    k = generate_gaussian_kernel_2d(ksize, sigma)
    assert k.shape == (ksize, ksize)
    assert np.isclose(k.sum(), 1.0, atol=1e-12)


def test_gaussian_kernel_symmetric_and_peaked():
    k = generate_gaussian_kernel_2d(7, 1.3)
    np.testing.assert_allclose(k, k[::-1, :], atol=1e-15)
    np.testing.assert_allclose(k, k[:, ::-1], atol=1e-15)
    np.testing.assert_allclose(k, k.T, atol=1e-15)
    assert np.unravel_index(k.argmax(), k.shape) == (3, 3)


def test_gaussian_kernel_matches_closed_form():
    sigma = 1.0
    k = generate_gaussian_kernel_2d(3, sigma)
    x = np.array([-1, 0, 1])
    xx, yy = np.meshgrid(x, x)
    raw = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    np.testing.assert_allclose(k, raw / raw.sum(), atol=1e-12)


def test_larger_sigma_flattens_kernel():
    peak_small = generate_gaussian_kernel_2d(15, 1.0).max()
    peak_large = generate_gaussian_kernel_2d(15, 4.0).max()
    assert peak_large < peak_small


def test_separable_equals_2d(rng):
    img = rng.random((32, 32)) * 255
    k2d = generate_gaussian_kernel_2d(9, 1.7)
    dense = conv2d(img, k2d, padding="same", pad_mode="reflect")
    sep = gaussian_blur(img, sigma=1.7, ksize=9, separable=True)
    np.testing.assert_allclose(dense, sep, atol=1e-9)


def test_gaussian_1d_unit_sum_and_symmetry():
    line = generate_gaussian_kernel_1d(9, 1.5)
    assert np.isclose(line.sum(), 1.0)
    np.testing.assert_allclose(line, line[::-1], atol=1e-15)


def test_blur_preserves_dc_level():
    flat = np.full((20, 20), 128.0)
    np.testing.assert_allclose(gaussian_blur(flat, sigma=2.0), 128.0, atol=1e-9)


def test_invalid_ksize_and_sigma():
    with pytest.raises(ValueError):
        generate_gaussian_kernel_2d(4, 1.0)  # even
    with pytest.raises(ValueError):
        generate_gaussian_kernel_2d(5, 0.0)  # non-positive sigma


# --------------------------------------------------------------------------- #
# unsharp mask
# --------------------------------------------------------------------------- #
def test_unsharp_alpha_zero_is_identity(rng):
    img = rng.random((16, 16)) * 255
    np.testing.assert_allclose(unsharp_mask(img, sigma=1.5, alpha=0.0), img, atol=1e-9)


def test_unsharp_output_bounded(rng):
    img = rng.random((40, 40)) * 255
    out = unsharp_mask(img, sigma=1.5, alpha=2.0)
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_unsharp_boosts_high_frequency_energy(disk):
    sharp = unsharp_mask(disk, sigma=1.5, alpha=1.5)
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    e_in = np.abs(conv2d(disk, lap, padding="same")).sum()
    e_out = np.abs(conv2d(sharp, lap, padding="same")).sum()
    assert e_out > e_in


def test_unsharp_negative_alpha_rejected(rng):
    with pytest.raises(ValueError):
        unsharp_mask(rng.random((8, 8)), alpha=-1.0)


# --------------------------------------------------------------------------- #
# Sobel operators
# --------------------------------------------------------------------------- #
def test_sobel_kernels_zero_sum():
    assert np.isclose(SOBEL_X.sum(), 0.0)
    assert np.isclose(SOBEL_Y.sum(), 0.0)


def test_sobel_on_constant_is_zero():
    flat = np.full((10, 10), 77.0)
    # reflect padding keeps a constant field constant, so the response is 0 everywhere
    np.testing.assert_allclose(
        conv2d(flat, SOBEL_X, padding="same", pad_mode="reflect"), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        conv2d(flat, SOBEL_Y, padding="same", pad_mode="reflect"), 0.0, atol=1e-12
    )
    # with zero padding only the interior is guaranteed zero
    np.testing.assert_allclose(conv2d(flat, SOBEL_X, padding="valid"), 0.0, atol=1e-12)


def test_sobel_vertical_edge_response(vertical_step):
    gx = conv2d(vertical_step, SOBEL_X, padding="same", pad_mode="reflect")
    gy = conv2d(vertical_step, SOBEL_Y, padding="same", pad_mode="reflect")
    # response concentrated at the column-32 discontinuity, horizontal gradient only
    assert np.abs(gx[:, 31:33]).mean() > 50.0
    assert np.abs(gy).max() < 1e-6
