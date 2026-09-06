"""cvlab -- from-scratch spatial filtering and Canny edge detection in pure NumPy.

Public API
----------
Task 1  :func:`conv2d`, :func:`conv3d`
Task 2  :func:`generate_gaussian_kernel_2d`, :func:`unsharp_mask`, :func:`pool2d`,
        :func:`max_pool_gradient`
Task 3  :func:`sobel_gradients`, :func:`non_max_suppression`, :func:`double_threshold`,
        :func:`hysteresis`, :func:`canny`
Task 4  :func:`add_gaussian_noise`
Helpers :mod:`cvlab.datasets`, :mod:`cvlab.metrics`, :mod:`cvlab.viz`, :mod:`cvlab.utils`
"""

from __future__ import annotations

from .canny import (
    CannyStages,
    canny,
    double_threshold,
    hysteresis,
    non_max_suppression,
    quantile_thresholds,
    sobel_gradients,
)
from .convolution import conv2d, conv3d
from .filters import (
    SOBEL_X,
    SOBEL_Y,
    default_gaussian_ksize,
    gaussian_blur,
    generate_gaussian_kernel_1d,
    generate_gaussian_kernel_2d,
    unsharp_mask,
)
from .metrics import EdgeMetrics, evaluate_edges
from .noise import add_gaussian_noise, add_salt_pepper_noise
from .pooling import average_pool_gradient, max_pool_gradient, pool2d
from .utils import normalize_to_uint8, to_grayscale

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # Task 1
    "conv2d",
    "conv3d",
    # Task 2
    "generate_gaussian_kernel_2d",
    "generate_gaussian_kernel_1d",
    "default_gaussian_ksize",
    "gaussian_blur",
    "unsharp_mask",
    "pool2d",
    "max_pool_gradient",
    "average_pool_gradient",
    "SOBEL_X",
    "SOBEL_Y",
    # Task 3
    "sobel_gradients",
    "non_max_suppression",
    "double_threshold",
    "hysteresis",
    "quantile_thresholds",
    "canny",
    "CannyStages",
    # Task 4
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    # helpers
    "to_grayscale",
    "normalize_to_uint8",
    "EdgeMetrics",
    "evaluate_edges",
]
