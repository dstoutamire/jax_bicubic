"""
Interpolation Kernels for Bicubic Sampling.

Both kernels use a 4-point stencil in each dimension (16 pixels total in 2D).
The key difference is their smoothness at the boundaries between pixels:

  KERNEL              FORMULA AT |t|<=1           CONTINUITY
  ----------------------------------------------------------------
  B-spline            2/3 - t^2 + |t|^3/2          C2
  Catmull-Rom         1 - 5t^2/2 + 3|t|^3/2        C1

The B-spline kernel REQUIRES prefiltered coefficients to be interpolating.
The Catmull-Rom kernel works directly on pixel values.

For gradient-based optimization:
- B-spline gives smoother gradients (better for second-order optimizers)
- Catmull-Rom is simpler but has kinks in the gradient at pixel boundaries
"""

import jax.numpy as jnp


def bspline_cubic_kernel(t: jnp.ndarray) -> jnp.ndarray:
    """
    Cubic B-spline kernel (requires prefiltered coefficients for interpolation).
    C2 continuous.
    """
    t = jnp.abs(t)
    t2 = t * t
    t3 = t2 * t

    return jnp.where(
        t <= 1,
        2.0/3.0 - t2 + t3/2.0,
        jnp.where(
            t <= 2,
            (2.0 - t) ** 3 / 6.0,
            0.0
        )
    )


def catmull_rom_kernel(t: jnp.ndarray) -> jnp.ndarray:
    """
    Catmull-Rom (Keys) kernel. No prefiltering needed, but only C1 continuous.
    """
    t = jnp.abs(t)
    t2 = t * t
    t3 = t2 * t

    a = -0.5
    return jnp.where(
        t <= 1,
        (a + 2) * t3 - (a + 3) * t2 + 1,
        jnp.where(
            t <= 2,
            a * t3 - 5 * a * t2 + 8 * a * t - 4 * a,
            0.0
        )
    )
