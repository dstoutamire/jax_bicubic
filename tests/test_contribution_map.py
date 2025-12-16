"""Test that contribution map is generated correctly."""

import jax.numpy as jnp

from jax_bicubic import prefilter_2d, bicubic_sample


def test_contribution_map():
    """Test that contribution map is generated correctly."""
    image = jnp.ones((20, 20), dtype=jnp.float32)
    coeffs = prefilter_2d(image)

    # Sample at a few points
    coords = jnp.array([[10.5, 10.5], [5.3, 7.8]], dtype=jnp.float32)

    result = bicubic_sample(coeffs, coords, compute_contributions=True)

    assert result.contribution_map is not None
    assert result.contribution_map.shape == (20, 20)

    # Contributions should be non-zero near sample points
    assert result.contribution_map[10, 10] > 0
    assert result.contribution_map[8, 5] > 0

    # Contributions should be zero far from sample points
    assert result.contribution_map[0, 0] == 0
    assert result.contribution_map[19, 19] == 0

    print(f"Total contribution: {jnp.sum(result.contribution_map):.3f}")


def test_contribution_map_single_point():
    """Test contribution map for a single sample point."""
    image = jnp.ones((10, 10), dtype=jnp.float32)
    coeffs = prefilter_2d(image)

    # Sample at center
    coords = jnp.array([[5.0, 5.0]], dtype=jnp.float32)

    result = bicubic_sample(coeffs, coords, compute_contributions=True)

    assert result.contribution_map is not None

    # Should have contributions in a 4x4 neighborhood around (5, 5)
    # Check that the contribution is concentrated near the sample point
    center_contrib = result.contribution_map[4:7, 4:7].sum()
    total_contrib = result.contribution_map.sum()

    # Most contribution should be near center
    assert center_contrib / total_contrib > 0.8


def test_contribution_map_disabled():
    """Test that contribution map is None when not requested."""
    image = jnp.ones((10, 10), dtype=jnp.float32)
    coeffs = prefilter_2d(image)
    coords = jnp.array([[5.0, 5.0]], dtype=jnp.float32)

    result = bicubic_sample(coeffs, coords, compute_contributions=False)

    assert result.contribution_map is None
