"""Test that sampling at integer coordinates returns original values."""

import numpy as np
import jax.numpy as jnp

from jax_bicubic import prefilter_2d, bicubic_sample


def test_interpolation_at_integers():
    """Test that sampling at integer coordinates returns original values."""
    # Create test image
    np.random.seed(42)
    image = jnp.array(np.random.rand(10, 10).astype(np.float32))

    # Prefilter
    coeffs = prefilter_2d(image)

    # Sample at all integer coordinates (avoiding boundary)
    yy, xx = jnp.meshgrid(jnp.arange(1, 9), jnp.arange(1, 9), indexing='ij')
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1).astype(jnp.float32)

    result = bicubic_sample(coeffs, coords)
    expected = image[1:9, 1:9].ravel()

    max_error = jnp.max(jnp.abs(result.values - expected))
    print(f"Max error at integer coordinates: {max_error:.2e}")

    assert max_error < 1e-5, f"Interpolation error too large: {max_error}"


def test_interpolation_larger_image():
    """Test interpolation on a larger image."""
    np.random.seed(123)
    image = jnp.array(np.random.rand(50, 50).astype(np.float32))

    coeffs = prefilter_2d(image)

    # Sample at integer coordinates in the interior
    yy, xx = jnp.meshgrid(jnp.arange(5, 45), jnp.arange(5, 45), indexing='ij')
    coords = jnp.stack([xx.ravel(), yy.ravel()], axis=-1).astype(jnp.float32)

    result = bicubic_sample(coeffs, coords)
    expected = image[5:45, 5:45].ravel()

    max_error = jnp.max(jnp.abs(result.values - expected))
    assert max_error < 1e-5, f"Interpolation error too large: {max_error}"


def test_interpolation_smooth_image():
    """Test interpolation on a smooth synthetic image."""
    x = jnp.linspace(-2, 2, 32)
    y = jnp.linspace(-2, 2, 32)
    xx, yy = jnp.meshgrid(x, y)
    image = jnp.sin(xx) * jnp.cos(yy)
    image = image.astype(jnp.float32)

    coeffs = prefilter_2d(image)

    # Sample at integer coordinates
    yy_idx, xx_idx = jnp.meshgrid(jnp.arange(2, 30), jnp.arange(2, 30), indexing='ij')
    coords = jnp.stack([xx_idx.ravel(), yy_idx.ravel()], axis=-1).astype(jnp.float32)

    result = bicubic_sample(coeffs, coords)
    expected = image[2:30, 2:30].ravel()

    max_error = jnp.max(jnp.abs(result.values - expected))
    assert max_error < 1e-5, f"Interpolation error too large: {max_error}"
