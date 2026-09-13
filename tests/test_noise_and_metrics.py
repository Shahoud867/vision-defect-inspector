"""Task 4 helpers -- noise models and edge-map metrics."""

from __future__ import annotations

import numpy as np

from cvlab import add_gaussian_noise, add_salt_pepper_noise, evaluate_edges
from cvlab.metrics import count_components, edge_density, prf1


def test_gaussian_noise_statistics():
    img = np.full((400, 400), 128.0)
    noisy = add_gaussian_noise(img, sigma=15.0, seed=0, clip=False)
    assert abs(noisy.std() - 15.0) < 0.5
    assert abs(noisy.mean() - 128.0) < 0.5


def test_gaussian_noise_deterministic_with_seed():
    img = np.full((32, 32), 100.0)
    a = add_gaussian_noise(img, 20.0, seed=7)
    b = add_gaussian_noise(img, 20.0, seed=7)
    np.testing.assert_array_equal(a, b)


def test_gaussian_noise_clips_to_range():
    img = np.full((64, 64), 250.0)
    noisy = add_gaussian_noise(img, sigma=40.0, seed=1, clip=True)
    assert noisy.min() >= 0.0 and noisy.max() <= 255.0


def test_salt_pepper_hits_extremes():
    img = np.full((100, 100), 128.0)
    out = add_salt_pepper_noise(img, amount=0.1, seed=2)
    assert (out == 255).any() and (out == 0).any()
    corrupted = np.count_nonzero((out == 0) | (out == 255))
    assert 500 < corrupted < 1500  # ~10% of 10 000


def test_edge_density_and_components():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2, :] = 255
    m[7, 3:6] = 255
    assert np.isclose(edge_density(m), 13 / 100)
    n, mean_len = count_components(m)
    assert n == 2
    assert np.isclose(mean_len, 6.5)


def test_prf1_perfect_and_tolerance():
    gt = np.zeros((20, 20), bool)
    gt[10, :] = True
    p_exact, r_exact, f_exact = prf1(gt, gt, tolerance=0)
    assert (p_exact, r_exact, f_exact) == (1.0, 1.0, 1.0)

    shifted = np.zeros((20, 20), bool)
    shifted[11, :] = True  # 1 px off
    p0, r0, _ = prf1(shifted, gt, tolerance=0)
    p2, r2, f2 = prf1(shifted, gt, tolerance=2)
    assert p0 == 0.0 and r0 == 0.0
    assert p2 == 1.0 and r2 == 1.0 and f2 == 1.0


def test_evaluate_edges_bundle():
    gt = np.zeros((16, 16), bool)
    gt[8, :] = True
    pred = np.zeros((16, 16), np.uint8)
    pred[8, :] = 255
    m = evaluate_edges(pred, gt, tolerance=1)
    assert m.f1 == 1.0
    assert m.n_components == 1
    d = m.as_dict()
    assert set(d) == {"density", "n_components", "mean_component_len", "precision", "recall", "f1"}
