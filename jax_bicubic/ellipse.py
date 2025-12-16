"""
Ellipse Utilities for Optimization Demos.

These utilities support the optimization demo. The ellipse is parameterized
by (cx, cy, a, b, theta) where:
  - (cx, cy) is the center
  - a, b are semi-major and semi-minor axes
  - theta is rotation angle in radians

Note on symmetries (important for optimization):
  - Ellipse is invariant under theta -> theta + pi (180 degree rotation)
  - Ellipse is invariant under (a,b,theta) -> (b,a,theta+pi/2) (swap axes)
  - These create multiple equivalent global minima in the loss landscape
"""

from typing import Tuple
import jax.numpy as jnp


def ellipse_points(
    cx: float, cy: float,
    a: float, b: float,
    theta: float,
    n_points: int = 100
) -> jnp.ndarray:
    """
    Generate equally arc-length spaced points along an ellipse.

    Uses numerical integration to compute arc lengths, then interpolates
    to find parameter values that give equal spacing along the curve.
    This provides uniform sampling density, which is important for
    accurate line integral approximation and stable gradients.

    Args:
        cx, cy: Center coordinates
        a, b: Semi-major and semi-minor axes
        theta: Rotation angle in radians
        n_points: Number of points to generate

    Returns:
        [n_points, 2] array of (x, y) coordinates, equally spaced by arc length
    """
    # Use many more points for accurate arc-length computation
    n_dense = 1000
    t_dense = jnp.linspace(0, 2 * jnp.pi, n_dense + 1)

    # Compute arc length differential: ds/dt = sqrt((dx/dt)^2 + (dy/dt)^2)
    # For parametric ellipse: x = a*cos(t), y = b*sin(t)
    # dx/dt = -a*sin(t), dy/dt = b*cos(t)
    # ds/dt = sqrt(a^2*sin^2(t) + b^2*cos^2(t))
    ds_dt = jnp.sqrt((a * jnp.sin(t_dense))**2 + (b * jnp.cos(t_dense))**2)

    # Cumulative arc length using trapezoidal rule
    dt = t_dense[1] - t_dense[0]
    arc_lengths = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum((ds_dt[:-1] + ds_dt[1:]) / 2 * dt)
    ])

    # Total arc length
    total_length = arc_lengths[-1]

    # Target arc lengths for equally spaced points (excluding endpoint to avoid duplicate)
    target_lengths = jnp.linspace(0, total_length, n_points, endpoint=False)

    # Interpolate to find t values for each target arc length
    t_equal = jnp.interp(target_lengths, arc_lengths, t_dense)

    # Parametric ellipse centered at origin, aligned with axes
    x_local = a * jnp.cos(t_equal)
    y_local = b * jnp.sin(t_equal)

    # Rotate
    cos_th = jnp.cos(theta)
    sin_th = jnp.sin(theta)
    x_rot = cos_th * x_local - sin_th * y_local
    y_rot = sin_th * x_local + cos_th * y_local

    # Translate
    x = x_rot + cx
    y = y_rot + cy

    return jnp.stack([x, y], axis=-1)


def draw_ellipse(
    shape: Tuple[int, int],
    cx: float, cy: float,
    a: float, b: float,
    theta: float,
    line_width: float = 1.5,
    foreground: float = 0.0,
    background: float = 1.0
) -> jnp.ndarray:
    """
    Draw an antialiased ellipse (dark on light background).

    Args:
        shape: (H, W) image dimensions
        cx, cy: Center coordinates
        a, b: Semi-major and semi-minor axes
        theta: Rotation angle in radians
        line_width: Width of the ellipse line
        foreground: Intensity of the ellipse line (default: 0.0 = black)
        background: Intensity of the background (default: 1.0 = white)

    Returns:
        [H, W] float32 array with the drawn ellipse
    """
    H, W = shape
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')

    # Transform to ellipse-local coordinates
    dx = xx - cx
    dy = yy - cy
    cos_th = jnp.cos(-theta)
    sin_th = jnp.sin(-theta)
    x_local = cos_th * dx - sin_th * dy
    y_local = sin_th * dx + cos_th * dy

    # Distance from ellipse in normalized coordinates
    # (x/a)^2 + (y/b)^2 = 1 on ellipse
    r = jnp.sqrt((x_local / a) ** 2 + (y_local / b) ** 2)

    # Approximate distance to ellipse boundary
    dist_from_boundary = jnp.abs(r - 1.0) * jnp.minimum(a, b)

    # Antialiased line
    alpha = jnp.clip(1.0 - dist_from_boundary / line_width, 0.0, 1.0)

    image = background * (1 - alpha) + foreground * alpha
    return image.astype(jnp.float32)
