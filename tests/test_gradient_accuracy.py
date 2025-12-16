"""
Verify that autodiff gradients match finite-difference gradients.

This is crucial for optimization: if the gradients are wrong, the
optimizer will move in the wrong direction and fail to converge.

Tests:
1. Gradient of sampled value w.r.t. sample coordinates
2. Gradient of mean value along ellipse w.r.t. ellipse parameters
"""

import jax
import jax.numpy as jnp

from jax_bicubic import prefilter_2d, bicubic_sample, ellipse_points, draw_ellipse


def test_gradient_accuracy_single_sample():
    """Test gradient of single sample w.r.t. coordinates."""
    # Create test image with smooth variation
    shape = (32, 32)
    yy, xx = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij')
    image = jnp.sin(xx * 0.3) * jnp.cos(yy * 0.3)
    image = image.astype(jnp.float32)
    coeffs = prefilter_2d(image)

    def sample_at(xy):
        """Sample at a single point."""
        coords = xy.reshape(1, 2)
        result = bicubic_sample(coeffs, coords)
        return result.values[0]

    test_point = jnp.array([15.3, 12.7])
    eps = 1e-4

    # Autodiff gradient
    grad_fn = jax.grad(sample_at)
    autodiff_grad = grad_fn(test_point)

    # Finite difference gradient
    fd_grad_x = (sample_at(test_point + jnp.array([eps, 0.0])) -
                 sample_at(test_point - jnp.array([eps, 0.0]))) / (2 * eps)
    fd_grad_y = (sample_at(test_point + jnp.array([0.0, eps])) -
                 sample_at(test_point - jnp.array([0.0, eps]))) / (2 * eps)
    fd_grad = jnp.array([fd_grad_x, fd_grad_y])

    print(f"Autodiff gradient:    [{autodiff_grad[0]:.6f}, {autodiff_grad[1]:.6f}]")
    print(f"Finite-diff gradient: [{fd_grad[0]:.6f}, {fd_grad[1]:.6f}]")

    grad_error = jnp.max(jnp.abs(autodiff_grad - fd_grad))
    print(f"Max difference: {grad_error:.2e}")

    assert grad_error < 1e-3, f"Sample gradient error too large: {grad_error}"


def test_gradient_accuracy_ellipse():
    """Test gradient of ellipse mean value w.r.t. parameters."""
    # Create image with a dark ellipse
    target_image = draw_ellipse(
        (64, 64), 32.0, 28.0, 18.0, 10.0, 0.4,
        line_width=2.0, foreground=0.0, background=1.0
    )
    target_coeffs = prefilter_2d(target_image)

    def ellipse_objective(params):
        """Mean intensity along parameterized ellipse."""
        cx, cy, a, b, theta = params
        a = jnp.maximum(a, 1.0)
        b = jnp.maximum(b, 1.0)
        coords = ellipse_points(cx, cy, a, b, theta, n_points=50)
        result = bicubic_sample(target_coeffs, coords)
        return jnp.mean(result.values)

    test_params = jnp.array([32.0, 28.0, 15.0, 12.0, 0.2])
    eps = 1e-4

    # Autodiff gradient
    grad_fn = jax.grad(ellipse_objective)
    autodiff_grad = grad_fn(test_params)

    # Finite difference gradient
    fd_grad = jnp.zeros(5)
    for i in range(5):
        delta = jnp.zeros(5).at[i].set(eps)
        fd_grad = fd_grad.at[i].set(
            (ellipse_objective(test_params + delta) -
             ellipse_objective(test_params - delta)) / (2 * eps)
        )

    print(f"Parameter: ['cx', 'cy', 'a', 'b', 'theta']")
    print(f"Autodiff:    [{', '.join(f'{g:.6f}' for g in autodiff_grad)}]")
    print(f"Finite-diff: [{', '.join(f'{g:.6f}' for g in fd_grad)}]")

    param_errors = jnp.abs(autodiff_grad - fd_grad)
    rel_errors = param_errors / (jnp.abs(fd_grad) + 1e-10)
    print(f"Max abs error: {jnp.max(param_errors):.2e}")
    print(f"Max rel error: {jnp.max(rel_errors):.2e}")

    max_error = jnp.max(param_errors)
    assert max_error < 2e-3, f"Ellipse gradient error too large: {max_error}"


def test_gradient_multiple_points():
    """Test gradient when sampling at multiple points."""
    shape = (32, 32)
    yy, xx = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij')
    image = jnp.sin(xx * 0.2) + jnp.cos(yy * 0.3)
    image = image.astype(jnp.float32)
    coeffs = prefilter_2d(image)

    def sample_mean(coords_flat):
        """Sample at multiple points and return mean."""
        coords = coords_flat.reshape(-1, 2)
        result = bicubic_sample(coeffs, coords)
        return jnp.mean(result.values)

    # Test with 3 sample points
    test_coords = jnp.array([10.5, 12.3, 15.2, 18.7, 20.1, 22.4])
    eps = 1e-4

    grad_fn = jax.grad(sample_mean)
    autodiff_grad = grad_fn(test_coords)

    # Finite difference
    fd_grad = jnp.zeros(6)
    for i in range(6):
        delta = jnp.zeros(6).at[i].set(eps)
        fd_grad = fd_grad.at[i].set(
            (sample_mean(test_coords + delta) -
             sample_mean(test_coords - delta)) / (2 * eps)
        )

    grad_error = jnp.max(jnp.abs(autodiff_grad - fd_grad))
    print(f"Multi-point gradient max error: {grad_error:.2e}")

    assert grad_error < 1e-3, f"Multi-point gradient error too large: {grad_error}"
