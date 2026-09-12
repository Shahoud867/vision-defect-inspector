"""Synthetic dataset generators and utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from cvlab import canny
from cvlab.datasets import get_samples, list_synthetic
from cvlab.utils import (
    compute_output_size,
    min_max_scale,
    normalize_to_uint8,
    resolve_padding,
    to_grayscale,
)


@pytest.mark.parametrize("cat", ["pcb", "concrete", "concrete_shadow", "concrete_texture"])
def test_synthetic_sample_contract(cat):
    (s,) = get_samples(cat, n=1, size=128, seed=0, prefer="synthetic")
    assert s.image.shape == (128, 128)
    assert s.image.dtype == np.float64
    assert s.image.min() >= 0.0 and s.image.max() <= 255.0
    assert s.source == "synthetic"
    assert s.gt_edges is not None and s.gt_edges.shape == (128, 128)
    assert s.gt_edges.dtype == bool and s.gt_edges.any()


def test_synthetic_is_deterministic():
    a = get_samples("concrete", n=1, size=96, seed=3, prefer="synthetic")[0]
    b = get_samples("concrete", n=1, size=96, seed=3, prefer="synthetic")[0]
    np.testing.assert_array_equal(a.image, b.image)


def test_pcb_is_high_contrast():
    s = get_samples("pcb", n=1, size=128, seed=0, prefer="synthetic")[0]
    assert s.image.std() > 40.0  # bright traces on dark substrate


def test_concrete_crack_is_detectable():
    s = get_samples("concrete", n=1, size=160, seed=0, prefer="synthetic")[0]
    edges = canny(s.image, sigma=1.4, low_q=0.7, high_q=0.92)
    assert (edges > 0).mean() > 0.003  # the crack produces a real ridge


def test_prefer_real_without_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_samples("pcb", n=1, data_dir=tmp_path, prefer="real")


def test_list_synthetic():
    assert set(list_synthetic()) == {"pcb", "concrete", "concrete_shadow", "concrete_texture"}


def test_multiple_samples_have_distinct_names():
    samples = get_samples("concrete", n=3, size=64, seed=0, prefer="synthetic")
    assert len({s.name for s in samples}) == 3


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
def test_to_grayscale_weights_sum_and_passthrough(rng):
    rgb = rng.random((8, 8, 3)) * 255
    g = to_grayscale(rgb)
    assert g.shape == (8, 8)
    np.testing.assert_allclose(g, to_grayscale(g))  # 2-D passthrough


def test_normalize_to_uint8_minmax_full_range():
    x = np.array([[-5.0, 0.0], [5.0, 10.0]])
    out = normalize_to_uint8(x, mode="minmax")
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_normalize_to_uint8_clip_preserves_scale():
    x = np.array([[-10.0, 300.0], [128.0, 200.0]])
    out = normalize_to_uint8(x, mode="clip")
    assert out[0, 0] == 0 and out[0, 1] == 255 and out[1, 0] == 128


def test_min_max_scale_constant_array():
    np.testing.assert_array_equal(min_max_scale(np.full((4, 4), 7.0), 0, 1), 0.0)


@pytest.mark.parametrize(
    ("padding", "fh", "fw", "expected"),
    [("same", 3, 3, (1, 1)), ("same", 5, 7, (2, 3)), ("valid", 5, 5, (0, 0)), (2, 3, 3, (2, 2))],
)
def test_resolve_padding(padding, fh, fw, expected):
    assert resolve_padding(padding, fh, fw) == expected


def test_compute_output_size_formula():
    assert compute_output_size(32, 5, 2, 2) == 16
    assert compute_output_size(28, 3, 0, 1) == 26
    with pytest.raises(ValueError):
        compute_output_size(3, 9, 0, 1)  # kernel bigger than input


# --------------------------------------------------------------------------- #
# image I/O round-trip
# --------------------------------------------------------------------------- #
def test_save_load_grayscale_roundtrip(tmp_path, rng):
    from cvlab.utils import load_gray, normalize_to_uint8, save_image

    img = normalize_to_uint8(rng.random((40, 50)) * 255, mode="clip")
    path = save_image(tmp_path / "g.png", img)
    assert path.exists()
    back = load_gray(path)
    assert back.shape == (40, 50)
    # exact for the cv2 path; <=1 LSB for the matplotlib colormap fallback
    assert np.abs(back - img).max() <= 1


def test_save_load_color_roundtrip(tmp_path, rng):
    from cvlab.utils import load_color, normalize_to_uint8, save_image

    rgb = normalize_to_uint8(rng.random((24, 32, 3)) * 255, mode="clip")
    path = save_image(tmp_path / "c.png", rgb)
    back = load_color(path)
    assert back.shape == (24, 32, 3)
    assert np.abs(back - rgb).max() <= 1


def test_load_missing_file_raises(tmp_path):
    from cvlab.utils import load_gray

    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        load_gray(tmp_path / "does_not_exist.png")


def test_real_dataset_path_reads_from_disk(tmp_path, rng):
    """get_samples(prefer='real') loads user-supplied images when present."""
    from cvlab.utils import normalize_to_uint8, save_image

    folder = tmp_path / "pcb"
    folder.mkdir()
    for i in range(2):
        save_image(
            folder / f"img{i}.png", normalize_to_uint8(rng.random((64, 64)) * 255, mode="clip")
        )
    samples = get_samples("pcb", n=2, size=48, data_dir=tmp_path, prefer="real")
    assert len(samples) == 2
    assert all(s.source == "real" and s.image.shape == (48, 48) for s in samples)
