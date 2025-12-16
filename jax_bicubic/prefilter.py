"""
B-Spline Prefilter for Cubic Interpolation.

The prefilter converts pixel values to B-spline coefficients. This is the
key step that enables C2 interpolation while maintaining the interpolating
property (passing through original pixel values at integer coordinates).

Mathematical background:
- Cubic B-spline basis beta(t) has support [-2, 2] and is C2 continuous
- Naive B-spline evaluation sum(image[i,j] * beta(x-i) * beta(y-j)) is C2 but SMOOTHS
- We need coefficients c[i,j] such that sum(c[i,j] * beta(x-i) * beta(y-j)) INTERPOLATES
- This requires solving: c * beta = image, or c = image * beta^-1
- The inverse filter beta^-1 is IIR with poles at z0 = -2+sqrt(3) and 1/z0

Implementation:
- Decompose into causal (forward) and anticausal (backward) passes
- Each pass is a simple first-order IIR: y[n] = x[n] + z0*y[n-1]
- Mirror boundary conditions for symmetry
- Gain factor normalizes the result
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

# Cubic B-spline pole: z0 = -2 + sqrt(3) ~ -0.268
# This is the root of the z-transform of the B-spline basis function
Z0 = -2.0 + jnp.sqrt(3.0)

# Normalization gain: (1 - z0)^2 / (-z0)
# Derived from the DC gain of the causal and anticausal filters
GAIN = (1 - Z0) ** 2 / (-Z0)


def _causal_filter_1d(x: jnp.ndarray) -> jnp.ndarray:
    """Causal IIR filter: y[n] = x[n] + z0 * y[n-1]"""
    def step(y_prev, xi):
        y = xi + Z0 * y_prev
        return y, y

    # Mirror boundary condition
    y0 = x[0] / (1 - Z0)
    _, y = lax.scan(step, y0, x)
    return y


def _anticausal_filter_1d(x: jnp.ndarray) -> jnp.ndarray:
    """Anticausal IIR filter: y[n] = z0 * (y[n+1] - x[n])"""
    def step(y_next, xi):
        y = Z0 * (y_next - xi)
        return y, y

    # Mirror boundary condition
    y0 = Z0 / (Z0 * Z0 - 1) * (x[-1] + Z0 * x[-2])
    _, y = lax.scan(step, y0, x, reverse=True)
    return y


def prefilter_1d(x: jnp.ndarray) -> jnp.ndarray:
    """Apply cubic B-spline prefilter along axis 0."""
    return GAIN * _anticausal_filter_1d(_causal_filter_1d(x))


def prefilter_2d(image: jnp.ndarray) -> jnp.ndarray:
    """
    Prefilter 2D image for cubic B-spline interpolation.

    Converts pixel values to B-spline coefficients.
    Cost: O(H*W), approximately 8 ops per pixel.

    Args:
        image: [H, W] float32 array

    Returns:
        [H, W] array of B-spline coefficients
    """
    # Filter along columns (axis 0), then rows (axis 1)
    coeffs = jax.vmap(prefilter_1d, in_axes=1, out_axes=1)(image)
    coeffs = jax.vmap(prefilter_1d, in_axes=0, out_axes=0)(coeffs)
    return coeffs
