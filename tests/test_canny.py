"""Task 3 -- Canny stages and the assembled pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import canny, double_threshold, hysteresis, non_max_suppression, sobel_gradients
from cvlab.canny import quantile_thresholds


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def test_gradients_on_blank_are_zero():
    gx, gy, mag, _ = sobel_gradients(np.full((20, 20), 50.0))
    assert np.allclose(gx, 0) and np.allclose(gy, 0) and np.allclose(mag, 0)


def test_gradient_magnitude_and_orientation_on_vertical_edge(vertical_step):
    *_, mag, theta = sobel_gradients(vertical_step, smooth_sigma=None)
    col = mag.sum(axis=0)
    assert col.argmax() in (31, 32)
    ang = np.rad2deg(theta[:, 32]) % 180.0
    # horizontal gradient -> orientation near 0 or 180
    assert np.all((ang < 20) | (ang > 160))


# --------------------------------------------------------------------------- #
# Stage 2 -- NMS
# --------------------------------------------------------------------------- #
def test_nms_thins_thick_gradient_band():
    ramp = np.zeros((32, 32))
    for j in range(12, 20):
        ramp[:, j:] += 32.0
    _, _, mag, theta = sobel_gradients(ramp, smooth_sigma=None)
    nms = non_max_suppression(mag, theta)
    widths = [(nms[r] > 0).sum() for r in range(8, 24)]
    assert max(widths) <= 2  # 1-px ridge (allow a 2-px tie at the shoulder)


def test_nms_borders_zeroed(rng):
    mag = rng.random((16, 16)) + 1.0
    theta = rng.uniform(-np.pi, np.pi, (16, 16))
    nms = non_max_suppression(mag, theta)
    assert np.all(nms[0, :] == 0) and np.all(nms[-1, :] == 0)
    assert np.all(nms[:, 0] == 0) and np.all(nms[:, -1] == 0)


def test_nms_keeps_isolated_ridge_pixel():
    mag = np.zeros((5, 5))
    mag[2, 2] = 9.0
    theta = np.zeros((5, 5))  # horizontal gradient -> compare (2,1) and (2,3)
    nms = non_max_suppression(mag, theta)
    assert nms[2, 2] == 9.0


def test_nms_suppresses_non_maximal_neighbour():
    mag = np.zeros((5, 5))
    mag[2, 1], mag[2, 2], mag[2, 3] = 5.0, 9.0, 7.0
    theta = np.zeros((5, 5))
    nms = non_max_suppression(mag, theta)
    assert nms[2, 2] == 9.0 and nms[2, 1] == 0.0 and nms[2, 3] == 0.0


# --------------------------------------------------------------------------- #
# Stage 3 -- double threshold + hysteresis
# --------------------------------------------------------------------------- #
def test_double_threshold_partitions():
    nms = np.array([[0.0, 5.0], [12.0, 25.0]])
    strong, weak = double_threshold(nms, t_low=4.0, t_high=20.0)
    np.testing.assert_array_equal(strong, [[False, False], [False, True]])
    np.testing.assert_array_equal(weak, [[False, True], [True, False]])


def test_double_threshold_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        double_threshold(np.zeros((3, 3)), t_low=10.0, t_high=1.0)


def test_hysteresis_links_weak_bridge():
    strong = np.zeros((5, 7), bool)
    strong[2, 0] = strong[2, 6] = True
    weak = np.zeros((5, 7), bool)
    weak[2, 1:6] = True
    weak[0, 3] = True  # isolated -> must be dropped
    out = hysteresis(strong, weak, "bfs")
    assert np.all(out[2, :] == 255)
    assert out[0, 3] == 0


def test_hysteresis_bfs_and_iterative_agree(rng):
    mag = rng.random((40, 40)) * 30
    strong = mag > 24
    weak = (mag > 12) & ~strong
    a = hysteresis(strong, weak, "bfs")
    b = hysteresis(strong, weak, "iterative")
    np.testing.assert_array_equal(a, b)


def test_hysteresis_diagonal_connectivity():
    strong = np.zeros((5, 5), bool)
    strong[0, 0] = True
    weak = np.zeros((5, 5), bool)
    for d in range(1, 5):
        weak[d, d] = True  # purely diagonal chain -> 8-connected
    out = hysteresis(strong, weak, "iterative")
    assert np.all(np.diag(out) == 255)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def test_canny_blank_image_has_no_edges():
    assert canny(np.full((30, 30), 100.0)).sum() == 0


def test_canny_vertical_edge_localised(vertical_step):
    edges = canny(vertical_step, sigma=1.0)
    cols = np.where(edges.sum(axis=0) > 0)[0]
    assert cols.size > 0
    assert cols.min() >= 29 and cols.max() <= 34
    assert set(np.unique(edges)).issubset({0, 255})


def test_canny_disk_contour_is_closed(disk):
    # permissive absolute thresholds -> recover (almost) the whole 8-connected ring
    edges = canny(disk, t_low=6.0, t_high=18.0, sigma=1.2)
    n = (edges > 0).sum()
    # 8-connected digital circle of radius ~24 is ~170-180 px long
    assert 150 < n < 260
    yy, xx = np.mgrid[0:96, 0:96]
    r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    on_ring = ((edges > 0) & (r >= 20) & (r <= 28)).sum()
    assert on_ring / n > 0.95  # essentially every edge pixel lies on the contour


def test_canny_returns_stages_record(disk):
    st = canny(disk, sigma=1.2, return_stages=True)
    assert st.edges.shape == disk.shape
    assert st.magnitude.shape == disk.shape
    assert 0.0 <= st.t_low <= st.t_high


def test_canny_explicit_thresholds_respected(disk):
    st = canny(disk, t_low=3.0, t_high=9.0, sigma=1.2, return_stages=True)
    assert st.t_low == 3.0 and st.t_high == 9.0


def test_quantile_thresholds_ordered(disk):
    _, _, mag, theta = sobel_gradients(disk, smooth_sigma=1.2)
    nms = non_max_suppression(mag, theta)
    lo, hi = quantile_thresholds(nms, 0.5, 0.9)
    assert 0 < lo <= hi


@pytest.mark.parametrize("k", [1, 2, 3])
def test_canny_rotation_sanity(disk, k):
    """Edge count is roughly stable under 90-degree rotations of the disk."""
    base = (canny(disk, sigma=1.2) > 0).sum()
    rot = (canny(np.rot90(disk, k), sigma=1.2) > 0).sum()
    assert abs(base - rot) < 0.15 * base + 20
