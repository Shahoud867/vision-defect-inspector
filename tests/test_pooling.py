"""Task 2 (part 3) -- pooling and analytical pooling gradients."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import max_pool_gradient, pool2d
from cvlab.pooling import average_pool_gradient

FEATURE_4x4 = np.array([[8, 7, 1, 2], [12, 9, 3, 4], [0, 1, 20, 21], [2, 3, 22, 23]], dtype=float)
BRIEF_BLOCK = np.array([[8.0, 7.0], [12.0, 9.0]])


def test_max_pool_known_values():
    out = pool2d(FEATURE_4x4, (2, 2), 2, "max")
    np.testing.assert_array_equal(out, [[12, 4], [3, 23]])


def test_average_pool_known_values():
    out = pool2d(FEATURE_4x4, (2, 2), 2, "average")
    np.testing.assert_allclose(out, [[9.0, 2.5], [1.5, 21.5]])


@pytest.mark.parametrize(
    ("H", "W", "p", "s", "exp"),
    [(4, 4, 2, 2, (2, 2)), (5, 5, 2, 2, (2, 2)), (7, 7, 3, 2, (3, 3)), (8, 6, 2, 1, (7, 5))],
)
def test_pool_output_shape(rng, H, W, p, s, exp):
    assert pool2d(rng.random((H, W)), (p, p), s, "max").shape == exp


def test_pool_3d_is_channelwise(rng):
    vol = rng.random((8, 8, 3))
    out = pool2d(vol, (2, 2), 2, "max")
    assert out.shape == (4, 4, 3)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], pool2d(vol[..., c], (2, 2), 2, "max"))


def test_maxpool_gradient_routes_to_argmax():
    """The brief's theoretical-reflection answer."""
    g = max_pool_gradient(BRIEF_BLOCK, (2, 2), 2)
    np.testing.assert_array_equal(g, [[0, 0], [1, 0]])


def test_maxpool_gradient_matches_finite_difference(rng):
    x = rng.random((2, 2))
    analytic = max_pool_gradient(x, (2, 2), 2)
    numeric = np.zeros_like(x)
    eps = 1e-6
    for i in range(2):
        for j in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            numeric[i, j] = (
                pool2d(xp, (2, 2), 2, "max")[0, 0] - pool2d(xm, (2, 2), 2, "max")[0, 0]
            ) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5)


def test_maxpool_gradient_scatters_upstream(rng):
    x = rng.random((4, 4))
    upstream = rng.random((2, 2))
    g = max_pool_gradient(x, (2, 2), 2, upstream=upstream)
    # non-overlapping windows over continuous noise -> 4 distinct argmax cells,
    # each carrying exactly one upstream value.
    assert np.isclose(g.sum(), upstream.sum())
    assert np.count_nonzero(g) == 4
    np.testing.assert_allclose(np.sort(g[g > 0]), np.sort(upstream.ravel()))


def test_average_pool_gradient_uniform():
    g = average_pool_gradient((2, 2), (2, 2), 2)
    np.testing.assert_allclose(g, np.full((2, 2), 0.25))


def test_average_pool_gradient_sums_to_output_count():
    g = average_pool_gradient((4, 4), (2, 2), 2)
    assert np.isclose(g.sum(), 4.0)  # 4 output cells, each distributing weight 1


def test_invalid_mode(rng):
    with pytest.raises(ValueError, match=r"max.*average"):
        pool2d(rng.random((4, 4)), (2, 2), 2, "median")
