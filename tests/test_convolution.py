"""Task 1 -- conv2d / conv3d correctness and geometry."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import conv2d, conv3d
from cvlab.utils import compute_output_size

IDENTITY = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float)
BOX = np.ones((3, 3)) / 9.0


# --------------------------------------------------------------------------- #
# conv2d
# --------------------------------------------------------------------------- #
def test_identity_same_is_noop(rng):
    img = rng.random((17, 23))
    np.testing.assert_allclose(conv2d(img, IDENTITY, padding="same"), img, atol=1e-12)


def test_valid_shrinks_by_kernel_minus_one(rng):
    img = rng.random((17, 23))
    out = conv2d(img, IDENTITY, padding="valid")
    assert out.shape == (15, 21)
    np.testing.assert_allclose(out, img[1:-1, 1:-1], atol=1e-12)


def test_box_filter_matches_manual_mean(rng):
    img = rng.random((9, 9))
    out = conv2d(img, BOX, padding="valid")
    assert np.isclose(out[0, 0], img[0:3, 0:3].mean())
    assert np.isclose(out[-1, -1], img[6:9, 6:9].mean())


@pytest.mark.parametrize(
    ("H", "W", "F", "stride", "padding"),
    [
        (32, 32, 3, 1, "same"),
        (32, 32, 3, 1, "valid"),
        (32, 32, 5, 2, "valid"),
        (31, 45, 3, 2, "same"),
        (64, 60, 7, 3, "valid"),
        (28, 28, 5, 2, "same"),
        (50, 50, 1, 1, "valid"),
    ],
)
def test_output_shape_matches_discrete_formula(rng, H, W, F, stride, padding):
    img = rng.random((H, W))
    k = rng.random((F, F))
    out = conv2d(img, k, stride=stride, padding=padding)
    p = (F - 1) // 2 if padding == "same" else 0
    assert out.shape == (
        compute_output_size(H, F, p, stride),
        compute_output_size(W, F, p, stride),
    )


def test_stride_two_subsamples_identity(rng):
    img = rng.random((10, 10))
    out = conv2d(img, IDENTITY, stride=2, padding="valid")
    assert out.shape == (4, 4)
    np.testing.assert_allclose(out, img[1:-1:2, 1:-1:2], atol=1e-12)


def test_linearity(rng):
    img_a, img_b = rng.random((12, 12)), rng.random((12, 12))
    k = rng.random((3, 3))
    lhs = conv2d(2.0 * img_a + 3.0 * img_b, k, padding="same")
    rhs = 2.0 * conv2d(img_a, k, padding="same") + 3.0 * conv2d(img_b, k, padding="same")
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_flip_kernel_gives_true_convolution(rng):
    img = rng.random((8, 8))
    k = rng.random((3, 3))
    cc = conv2d(img, k, padding="valid", flip_kernel=False)
    cv = conv2d(img, k[::-1, ::-1], padding="valid", flip_kernel=True)
    np.testing.assert_allclose(cc, cv, atol=1e-12)


def test_even_kernel_rejected_for_same(rng):
    with pytest.raises(ValueError, match="odd"):
        conv2d(rng.random((8, 8)), np.ones((2, 2)), padding="same")


def test_bad_padding_string(rng):
    with pytest.raises(ValueError, match=r"same.*valid"):
        conv2d(rng.random((8, 8)), IDENTITY, padding="reflect")


def test_pad_modes_change_border_only(rng):
    img = rng.random((10, 10))
    k = rng.random((3, 3))
    a = conv2d(img, k, padding="same", pad_mode="constant")
    b = conv2d(img, k, padding="same", pad_mode="reflect")
    np.testing.assert_allclose(a[1:-1, 1:-1], b[1:-1, 1:-1], atol=1e-12)
    assert not np.allclose(a[0, :], b[0, :])


# --------------------------------------------------------------------------- #
# conv3d
# --------------------------------------------------------------------------- #
def test_conv3d_activation_volume_shape(rng):
    vol = rng.random((20, 24, 3))
    stack = rng.random((5, 3, 3, 3))
    out = conv3d(vol, stack, padding="same")
    assert out.shape == (20, 24, 5)


def test_conv3d_channel_sum_manual():
    vol = np.zeros((3, 3, 2))
    vol[..., 0] = 1.0
    vol[..., 1] = 10.0
    k = np.ones((1, 3, 3, 2))  # one filter, sums the 3x3x2 patch
    out = conv3d(vol, k, padding="valid")
    assert out.shape == (1, 1, 1)
    assert np.isclose(out[0, 0, 0], 9 * 1.0 + 9 * 10.0)


def test_conv3d_matches_conv2d_single_channel(rng):
    img = rng.random((15, 15))
    k = rng.random((3, 3))
    a = conv2d(img, k, padding="same")
    b = conv3d(img[..., None], [k[..., None]], padding="same")[..., 0]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_conv3d_accepts_kernel_sequence(rng):
    vol = rng.random((10, 10, 3))
    kernels = [rng.random((3, 3, 3)) for _ in range(4)]
    out = conv3d(vol, kernels, padding="same")
    assert out.shape == (10, 10, 4)


def test_conv3d_channel_mismatch_raises(rng):
    with pytest.raises(ValueError, match="channel"):
        conv3d(rng.random((10, 10, 3)), rng.random((2, 3, 3, 4)), padding="same")


def test_conv3d_stride(rng):
    vol = rng.random((16, 16, 2))
    stack = rng.random((3, 3, 3, 2))
    out = conv3d(vol, stack, stride=2, padding="valid")
    assert out.shape == (7, 7, 3)


# --------------------------------------------------------------------------- #
# optional SciPy oracle (dev-only)
# --------------------------------------------------------------------------- #
@pytest.mark.oracle
def test_matches_scipy_correlate(rng):
    ndi = pytest.importorskip("scipy.ndimage")
    img = rng.random((40, 40))
    k = rng.random((5, 5))
    ours = conv2d(img, k, padding="same", pad_mode="constant")
    ref = ndi.correlate(img, k, mode="constant", cval=0.0)
    np.testing.assert_allclose(ours, ref, atol=1e-10)
