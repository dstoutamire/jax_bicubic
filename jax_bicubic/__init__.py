"""
C2-continuous bicubic interpolation with B-spline prefiltering for JAX.

This module provides smooth bicubic interpolation suitable for gradient-based
optimization. The key advantage over standard bicubic (Catmull-Rom) interpolation
is C2 continuity, which gives smoother gradients for optimization.

Main Functions:
    prefilter_2d     - Convert pixel values to B-spline coefficients
    bicubic_sample   - Sample at arbitrary coordinates

Kernels:
    bspline_cubic_kernel  - C2 kernel (use with prefiltered coefficients)
    catmull_rom_kernel    - C1 kernel (use with raw image)

Ellipse Utilities:
    ellipse_points   - Generate points along an ellipse
    draw_ellipse     - Render antialiased ellipse image

Example:
    >>> import jax.numpy as jnp
    >>> from jax_bicubic import prefilter_2d, bicubic_sample
    >>>
    >>> # Create test image
    >>> image = jnp.ones((64, 64), dtype=jnp.float32)
    >>>
    >>> # Prefilter for C2 interpolation
    >>> coeffs = prefilter_2d(image)
    >>>
    >>> # Sample at subpixel coordinates
    >>> coords = jnp.array([[10.5, 20.3], [30.7, 40.1]])
    >>> result = bicubic_sample(coeffs, coords)
    >>> print(result.values)
"""

from .prefilter import prefilter_2d, prefilter_1d, Z0, GAIN
from .kernels import bspline_cubic_kernel, catmull_rom_kernel
from .sample import bicubic_sample, SampleResult
from .ellipse import ellipse_points, draw_ellipse

__version__ = "0.1.0"

__all__ = [
    # Prefiltering
    "prefilter_2d",
    "prefilter_1d",
    "Z0",
    "GAIN",
    # Kernels
    "bspline_cubic_kernel",
    "catmull_rom_kernel",
    # Sampling
    "bicubic_sample",
    "SampleResult",
    # Ellipse utilities
    "ellipse_points",
    "draw_ellipse",
]
