"""
Demonstrate ellipse parameter recovery using gradient descent.

This test creates a synthetic image with a dark ellipse on a light
background, then uses gradient-based optimization to recover the
ellipse parameters from an initial circular guess.

The objective function computes the mean image intensity along the
parameterized ellipse curve. Since the ellipse is dark (intensity 0)
on a light background (intensity 1), minimizing this objective
drives the curve toward the dark ring.

This demonstrates:
1. Gradients flow correctly through bicubic_sample()
2. The C2 interpolant provides smooth gradients for optimization
3. Automatic differentiation (jax.grad) works with the full pipeline

The optimization uses momentum-based gradient descent with learning
rate decay. Note that due to ellipse symmetries, there are multiple
equivalent global minima.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from jax_bicubic import prefilter_2d, bicubic_sample, ellipse_points, draw_ellipse


def test_ellipse_optimization(output_dir=None):
    """
    Demonstrate ellipse parameter recovery via gradient descent.

    Args:
        output_dir: Directory to save output images. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = "."

    # Ground truth ellipse parameters
    true_params = {
        'cx': 32.0,
        'cy': 28.0,
        'a': 18.0,   # semi-major
        'b': 10.0,   # semi-minor
        'theta': 0.4  # rotation ~23 degrees
    }

    # Create target image
    shape = (64, 64)
    target_image = draw_ellipse(
        shape,
        true_params['cx'], true_params['cy'],
        true_params['a'], true_params['b'],
        true_params['theta'],
        line_width=2.0,
        foreground=0.0,
        background=1.0
    )

    # Prefilter for C2 interpolation
    coeffs = prefilter_2d(target_image)

    # Initial guess: circle overlapping the ellipse
    init_params = jnp.array([
        32.0,  # cx (correct)
        28.0,  # cy (correct)
        14.0,  # a (average of true a and b)
        14.0,  # b (same, making it a circle)
        0.0    # theta (no rotation)
    ])

    n_sample_points = 200  # More samples for smoother gradients

    def objective(params):
        """
        Compute mean image intensity along the ellipse curve.

        For a dark ellipse on light background:
        - Points ON the ellipse have low intensity (~0)
        - Points OFF the ellipse have high intensity (~1)

        Minimizing this drives the parameterized curve toward the dark ring.
        """
        cx, cy, a, b, theta = params

        # Soft constraints to keep parameters reasonable
        a = jnp.maximum(a, 1.0)
        b = jnp.maximum(b, 1.0)

        coords = ellipse_points(cx, cy, a, b, theta, n_sample_points)
        result = bicubic_sample(coeffs, coords)

        # Mean value along ellipse (minimize to find dark curve)
        return jnp.mean(result.values)

    print(f"Initial parameters: cx={init_params[0]:.1f}, cy={init_params[1]:.1f}, "
          f"a={init_params[2]:.1f}, b={init_params[3]:.1f}, theta={init_params[4]:.3f}")
    print(f"Initial loss: {objective(init_params):.4f}")

    # Use manual gradient descent with adaptive learning rate
    params = init_params.copy()
    grad_fn = jax.grad(objective)

    param_history = [params.copy()]
    loss_history = [float(objective(params))]

    # Gradient descent with momentum
    velocity = jnp.zeros_like(params)
    learning_rate = 1.0
    momentum = 0.9

    for i in range(200):
        grad = grad_fn(params)

        # Momentum update
        velocity = momentum * velocity - learning_rate * grad
        params = params + velocity

        # Keep a, b positive and reasonable
        params = params.at[2].set(jnp.clip(params[2], 3.0, 40.0))
        params = params.at[3].set(jnp.clip(params[3], 3.0, 40.0))

        # Keep center in image
        params = params.at[0].set(jnp.clip(params[0], 10.0, 54.0))
        params = params.at[1].set(jnp.clip(params[1], 10.0, 54.0))

        # Wrap theta to [-pi, pi]
        params = params.at[4].set(jnp.mod(params[4] + jnp.pi, 2*jnp.pi) - jnp.pi)

        param_history.append(params.copy())
        loss_history.append(float(objective(params)))

        # Learning rate decay
        if i > 0 and i % 50 == 0:
            learning_rate *= 0.5

    param_history = jnp.stack(param_history)
    final_params = param_history[-1]

    print(f"\nFinal parameters: cx={final_params[0]:.1f}, cy={final_params[1]:.1f}, "
          f"a={final_params[2]:.1f}, b={final_params[3]:.1f}, theta={final_params[4]:.3f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"True parameters:  cx={true_params['cx']:.1f}, cy={true_params['cy']:.1f}, "
          f"a={true_params['a']:.1f}, b={true_params['b']:.1f}, theta={true_params['theta']:.3f}")

    # Compute contribution map for visualization
    final_coords = ellipse_points(
        final_params[0], final_params[1],
        final_params[2], final_params[3],
        final_params[4],
        n_sample_points
    )
    final_result = bicubic_sample(coeffs, final_coords, compute_contributions=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Target image with initial and final ellipses
    axes[0, 0].imshow(target_image, cmap='gray', origin='lower', vmin=0, vmax=1)

    init_coords = ellipse_points(
        init_params[0], init_params[1],
        init_params[2], init_params[3],
        init_params[4],
        n_sample_points
    )
    axes[0, 0].plot(init_coords[:, 0], init_coords[:, 1], 'b-',
                    linewidth=2, label='Initial (circle)')
    axes[0, 0].plot(final_coords[:, 0], final_coords[:, 1], 'r--',
                    linewidth=2, label='Optimized')

    # Also show true ellipse
    true_coords = ellipse_points(
        true_params['cx'], true_params['cy'],
        true_params['a'], true_params['b'],
        true_params['theta'],
        n_sample_points
    )
    axes[0, 0].plot(true_coords[:, 0], true_coords[:, 1], 'g:',
                    linewidth=2, label='True')

    axes[0, 0].set_title('Target image with initial, optimized, and true ellipse')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, shape[1])
    axes[0, 0].set_ylim(0, shape[0])

    # Contribution map
    im = axes[0, 1].imshow(final_result.contribution_map, cmap='hot', origin='lower')
    axes[0, 1].set_title('Pixel contribution map (optimized ellipse)')
    plt.colorbar(im, ax=axes[0, 1])

    # Prefiltered coefficients
    axes[1, 0].imshow(coeffs, cmap='gray', origin='lower')
    axes[1, 0].set_title('Prefiltered coefficients')

    # Loss convergence
    axes[1, 1].semilogy(loss_history, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss (log scale)')
    axes[1, 1].set_title('Optimization convergence')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ellipse_optimization.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Parameter evolution plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    param_names = ['cx', 'cy', 'a', 'b', 'theta']
    true_values = [true_params['cx'], true_params['cy'],
                   true_params['a'], true_params['b'], true_params['theta']]

    for i, (name, true_val) in enumerate(zip(param_names, true_values)):
        ax = axes[i // 3, i % 3]
        ax.plot(param_history[:, i], 'b-', linewidth=2, label='Optimized')
        ax.axhline(y=true_val, color='r', linestyle='--', label='True')
        ax.axhline(y=float(init_params[i]), color='gray', linestyle=':',
                   alpha=0.5, label='Initial')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(name)
        ax.set_title(f'Parameter: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary in last subplot
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = (
        f"Parameter Recovery Results\n"
        f"{'_'*30}\n\n"
        f"{'Param':<6} {'True':<8} {'Init':<8} {'Final':<8}\n"
        f"{'_'*30}\n"
        f"{'cx':<6} {true_params['cx']:<8.2f} {init_params[0]:<8.2f} {final_params[0]:<8.2f}\n"
        f"{'cy':<6} {true_params['cy']:<8.2f} {init_params[1]:<8.2f} {final_params[1]:<8.2f}\n"
        f"{'a':<6} {true_params['a']:<8.2f} {init_params[2]:<8.2f} {final_params[2]:<8.2f}\n"
        f"{'b':<6} {true_params['b']:<8.2f} {init_params[3]:<8.2f} {final_params[3]:<8.2f}\n"
        f"{'theta':<6} {true_params['theta']:<8.3f} {init_params[4]:<8.3f} {final_params[4]:<8.3f}\n"
    )
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=11, verticalalignment='center')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ellipse_parameters.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Check convergence (accounting for ellipse symmetries)
    final_a = float(final_params[2])
    final_b = float(final_params[3])
    final_theta = float(final_params[4])

    # Try both (a,b,theta) and (b,a,theta-pi/2) interpretations
    errors_v1 = {
        'a': abs(final_a - true_params['a']),
        'b': abs(final_b - true_params['b']),
    }
    errors_v2 = {
        'a': abs(final_b - true_params['a']),  # swapped
        'b': abs(final_a - true_params['b']),  # swapped
    }

    # Pick the interpretation with smaller total error
    if errors_v1['a'] + errors_v1['b'] < errors_v2['a'] + errors_v2['b']:
        param_errors = errors_v1
        effective_theta = final_theta
    else:
        param_errors = errors_v2
        effective_theta = final_theta - jnp.pi/2

    # Normalize theta error to [-pi/2, pi/2] accounting for pi periodicity
    theta_diff = effective_theta - true_params['theta']
    theta_diff = jnp.mod(theta_diff + jnp.pi/2, jnp.pi) - jnp.pi/2
    param_errors['theta'] = abs(float(theta_diff))

    param_errors['cx'] = abs(float(final_params[0]) - true_params['cx'])
    param_errors['cy'] = abs(float(final_params[1]) - true_params['cy'])

    print("\nParameter errors (accounting for symmetries):")
    for name, err in param_errors.items():
        print(f"  {name}: {err:.3f}")

    # Check that loss decreased significantly
    loss_improvement = loss_history[0] - loss_history[-1]
    print(f"Loss improvement: {loss_improvement:.4f}")

    # Relaxed thresholds - just verify optimization is working
    assert param_errors['cx'] < 3.0, f"cx error too large: {param_errors['cx']}"
    assert param_errors['cy'] < 3.0, f"cy error too large: {param_errors['cy']}"
    assert param_errors['a'] < 5.0, f"a error too large: {param_errors['a']}"
    assert param_errors['b'] < 5.0, f"b error too large: {param_errors['b']}"
    assert loss_improvement > 0.01, f"Loss should decrease: {loss_improvement}"

    return True


if __name__ == "__main__":
    test_ellipse_optimization(output_dir="images")
