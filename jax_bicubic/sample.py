"""
Bicubic Sampling Functions.

The sampling function evaluates the interpolant at arbitrary (x, y) coordinates.
It's designed for efficient batched evaluation along curves/paths.

Key design decisions for gradient-based optimization:

1. SEPARABLE EVALUATION: We compute weights separately for x and y, then
   combine with einsum. This is mathematically equivalent to evaluating
   the full 2D kernel but more efficient and numerically stable.

2. BOUNDARY HANDLING: Coordinates outside [0, W-1] x [0, H-1] are clamped.
   This ensures the function is defined everywhere, which is important
   when optimizing parameters that might push samples outside the image.

3. CONTRIBUTION MAP: Optional output showing which pixels influenced each
   sample, weighted by their contribution. Useful for:
   - Visualizing what the optimizer "sees"
   - Debugging unexpected gradients
   - Understanding the effective receptive field

4. JAX COMPATIBILITY: All operations are JAX primitives, so the function
   is fully differentiable via jax.grad() and compatible with jax.jit().
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

from .kernels import bspline_cubic_kernel


@dataclass
class SampleResult:
    """Result of bicubic sampling."""
    values: jnp.ndarray  # [N] sampled values
    contribution_map: Optional[jnp.ndarray] = None  # [H, W] pixel contributions


def _bicubic_sample_single(
    coeffs: jnp.ndarray,
    x: float,
    y: float,
    kernel_fn
) -> float:
    """Sample at a single point using explicit loops (correct gradients)."""
    H, W = coeffs.shape
    x0 = jnp.floor(x)
    y0 = jnp.floor(y)
    fx = x - x0
    fy = y - y0
    x0i = x0.astype(jnp.int32)
    y0i = y0.astype(jnp.int32)

    result = 0.0
    for oy in [-1, 0, 1, 2]:
        wy = kernel_fn(fy - oy)
        for ox in [-1, 0, 1, 2]:
            wx = kernel_fn(fx - ox)
            iy = jnp.clip(y0i + oy, 0, H - 1)
            ix = jnp.clip(x0i + ox, 0, W - 1)
            result = result + wy * wx * coeffs[iy, ix]
    return result


def bicubic_sample(
    coeffs: jnp.ndarray,
    coords: jnp.ndarray,
    kernel_fn=bspline_cubic_kernel,
    compute_contributions: bool = False
) -> SampleResult:
    """
    Sample from coefficient image at subpixel coordinates.

    Args:
        coeffs: [H, W] array (prefiltered for B-spline, raw for Catmull-Rom)
        coords: [N, 2] array of (x, y) coordinates (x=col, y=row)
        kernel_fn: Interpolation kernel function
        compute_contributions: If True, compute per-pixel contribution map

    Returns:
        SampleResult with sampled values and optional contribution map
    """
    H, W = coeffs.shape
    N = coords.shape[0]

    # Use vmap over single-point sampling for correct gradient flow
    values = jax.vmap(
        lambda xy: _bicubic_sample_single(coeffs, xy[0], xy[1], kernel_fn)
    )(coords)

    # Compute contribution map if requested (for visualization only)
    contribution_map = None
    if compute_contributions:
        contribution_map = jnp.zeros((H, W), dtype=jnp.float32)

        for n in range(N):
            x, y = coords[n, 0], coords[n, 1]
            x0 = int(jnp.floor(x))
            y0 = int(jnp.floor(y))
            fx = float(x - x0)
            fy = float(y - y0)

            for oy in [-1, 0, 1, 2]:
                wy = float(kernel_fn(fy - oy))
                for ox in [-1, 0, 1, 2]:
                    wx = float(kernel_fn(fx - ox))
                    iy = int(jnp.clip(y0 + oy, 0, H - 1))
                    ix = int(jnp.clip(x0 + ox, 0, W - 1))
                    contribution_map = contribution_map.at[iy, ix].add(
                        jnp.abs(wy * wx)
                    )

    return SampleResult(values=values, contribution_map=contribution_map)
