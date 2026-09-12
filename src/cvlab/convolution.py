"""Task 1 -- Generalised 2-D and 3-D convolution engines (pure NumPy).

Both engines implement spatial **cross-correlation** (sliding-window dot product
without kernel flipping).  Section 1.2 of the brief explicitly permits this and
it is the convention used by essentially every modern deep-learning framework.
Pass ``flip_kernel=True`` for a mathematically exact convolution.

The inner loop is fully vectorised with :func:`numpy.lib.stride_tricks.sliding_window_view`
followed by a single :func:`numpy.einsum` contraction -- no Python-level loop over
pixels -- so a 512x512 image with a 5x5 kernel runs in a few milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .utils import compute_output_size, ensure_2d, ensure_3d, pad_array, resolve_padding

__all__ = ["conv2d", "conv3d", "sliding_windows"]


def sliding_windows(padded: np.ndarray, k_h: int, k_w: int, stride: int) -> np.ndarray:
    """Return a strided view of shape ``(out_h, out_w, k_h, k_w, *padded.shape[2:])``.

    No data is copied: the result is a read-only view into ``padded``.
    """
    # numpy's type stub does not model a tuple ``axis`` here, though it is valid.
    windows = sliding_window_view(padded, (k_h, k_w), axis=(0, 1))  # type: ignore[call-overload]
    # windows.shape == (H-k_h+1, W-k_w+1, *channels, k_h, k_w)
    windows = np.moveaxis(windows, (-2, -1), (2, 3))
    if stride > 1:
        windows = windows[::stride, ::stride]
    return windows


# --------------------------------------------------------------------------- #
# 2-D convolution
# --------------------------------------------------------------------------- #
def conv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: str | int | tuple[int, int] = "same",
    pad_value: float = 0.0,
    pad_mode: str = "constant",
    flip_kernel: bool = False,
) -> np.ndarray:
    r"""2-D spatial convolution / cross-correlation from first principles.

    Parameters
    ----------
    image
        2-D array ``(H, W)``.  Cast to ``float64`` internally.
    kernel
        2-D array ``(k_h, k_w)``.  Odd dimensions are required for ``'same'``.
    stride
        Step of the sliding window along both axes (``>= 1``).
    padding
        ``'same'`` (output matches input when ``stride == 1``), ``'valid'``
        (no padding), an ``int``, or an explicit ``(pad_h, pad_w)`` tuple.
    pad_value
        Constant used when ``pad_mode == 'constant'``.
    pad_mode
        ``'constant'`` (brief default), ``'reflect'``, ``'replicate'`` or ``'wrap'``.
    flip_kernel
        If ``True`` perform true convolution (kernel flipped on both axes);
        otherwise cross-correlation.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of shape ``(H_out, W_out)`` where

        .. math::
            H_{out} = \left\lfloor \tfrac{H - k_h + 2P_h}{S} \right\rfloor + 1, \quad
            W_{out} = \left\lfloor \tfrac{W - k_w + 2P_w}{S} \right\rfloor + 1
    """
    img = ensure_2d(image)
    k = ensure_2d(kernel, name="kernel")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")

    k_h, k_w = k.shape
    if flip_kernel:
        k = k[::-1, ::-1]

    pad_h, pad_w = resolve_padding(padding, k_h, k_w)
    padded = pad_array(img, pad_h, pad_w, mode=pad_mode, value=pad_value)

    out_h = compute_output_size(img.shape[0], k_h, pad_h, stride)
    out_w = compute_output_size(img.shape[1], k_w, pad_w, stride)

    windows = sliding_windows(padded, k_h, k_w, stride)[:out_h, :out_w]
    # (out_h, out_w, k_h, k_w) . (k_h, k_w) -> (out_h, out_w)
    result = np.einsum("ijhw,hw->ij", windows, k, optimize=True)

    assert result.shape == (out_h, out_w), (result.shape, (out_h, out_w))
    return np.ascontiguousarray(result)


# --------------------------------------------------------------------------- #
# 3-D multi-channel convolution
# --------------------------------------------------------------------------- #
def _stack_kernels(kernel_stack: np.ndarray | Sequence[np.ndarray], channels: int) -> np.ndarray:
    """Coerce ``kernel_stack`` to a ``(K, k_h, k_w, C)`` float64 array."""
    if isinstance(kernel_stack, (list, tuple)):
        stack = np.stack([np.asarray(k, dtype=np.float64) for k in kernel_stack], axis=0)
    else:
        stack = np.asarray(kernel_stack, dtype=np.float64)
        if stack.ndim == 3:  # a single (k_h, k_w, C) kernel
            stack = stack[None, ...]
    if stack.ndim != 4:
        raise ValueError(
            "kernel_stack must be (K, k_h, k_w, C) or a sequence of (k_h, k_w, C) kernels; "
            f"got shape {stack.shape!r}"
        )
    if stack.shape[-1] != channels:
        raise ValueError(
            f"kernel channel count {stack.shape[-1]} != image channel count {channels}"
        )
    if stack.shape[1] % 2 == 0 or stack.shape[2] % 2 == 0:
        # allowed for 'valid'/int padding; resolve_padding re-checks for 'same'
        pass
    return stack


def conv3d(
    image: np.ndarray,
    kernel_stack: np.ndarray | Sequence[np.ndarray],
    stride: int = 1,
    padding: str | int | tuple[int, int] = "same",
    pad_value: float = 0.0,
    pad_mode: str = "constant",
    flip_kernel: bool = False,
) -> np.ndarray:
    r"""Multi-channel convolutional layer: ``K`` filters over a ``(H, W, C)`` volume.

    For every filter the engine multiplies element-wise across all ``C`` input
    channels, sums the ``k_h \times k_w \times C`` products into one scalar, and
    stacks the ``K`` resulting planes into an activation volume.

    Parameters
    ----------
    image
        ``(H, W, C)`` array.  A ``(H, W)`` array is treated as ``C = 1``.
    kernel_stack
        ``(K, k_h, k_w, C)`` array, or a sequence of ``K`` arrays of shape
        ``(k_h, k_w, C)``.
    stride, padding, pad_value, pad_mode, flip_kernel
        As in :func:`conv2d`.

    Returns
    -------
    numpy.ndarray
        ``float64`` activation volume of shape ``(H_out, W_out, K)``.
    """
    vol = ensure_3d(image)
    H, W, C = vol.shape
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")

    stack = _stack_kernels(kernel_stack, C)
    K, k_h, k_w, _ = stack.shape
    if flip_kernel:
        stack = stack[:, ::-1, ::-1, :]

    pad_h, pad_w = resolve_padding(padding, k_h, k_w)
    padded = pad_array(vol, pad_h, pad_w, mode=pad_mode, value=pad_value)

    out_h = compute_output_size(H, k_h, pad_h, stride)
    out_w = compute_output_size(W, k_w, pad_w, stride)

    # windows: (out_h, out_w, k_h, k_w, C)
    windows = sliding_windows(padded, k_h, k_w, stride)[:out_h, :out_w]
    # contract kernel window & channels against every filter -> (out_h, out_w, K)
    result = np.einsum("ijhwc,khwc->ijk", windows, stack, optimize=True)

    assert result.shape == (out_h, out_w, K), (result.shape, (out_h, out_w, K))
    return np.ascontiguousarray(result)
