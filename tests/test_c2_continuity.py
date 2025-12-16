"""
Test C2 continuity by probing second derivatives along a diagonal.
Shows discontinuity without prefilter, continuity with prefilter.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

from jax_bicubic import (
    prefilter_2d,
    bicubic_sample,
    bspline_cubic_kernel,
    catmull_rom_kernel,
)


def test_c2_continuity(output_dir=None):
    """
    Test C2 continuity by probing second derivatives along a diagonal.

    Args:
        output_dir: Directory to save output image. If None, uses current directory.
    """
    if output_dir is None:
        output_dir = "."

    # Create smooth test image with some structure
    x = jnp.linspace(-2, 2, 32)
    y = jnp.linspace(-2, 2, 32)
    xx, yy = jnp.meshgrid(x, y)
    image = jnp.sin(xx) * jnp.cos(yy) + 0.5 * xx
    image = image.astype(jnp.float32)

    # Diagonal line through the image
    t = jnp.linspace(4, 28, 500)
    coords = jnp.stack([t, t], axis=-1)

    # Spacing for numerical second derivative
    dt = t[1] - t[0]

    results = {}

    for name, use_prefilter in [("Without prefilter (Catmull-Rom)", False),
                                 ("With prefilter (B-spline)", True)]:
        if use_prefilter:
            coeffs = prefilter_2d(image)
            kernel = bspline_cubic_kernel
        else:
            coeffs = image
            kernel = catmull_rom_kernel

        result = bicubic_sample(coeffs, coords, kernel_fn=kernel)
        values = result.values

        # Numerical second derivative
        d2 = (values[2:] - 2*values[1:-1] + values[:-2]) / (dt * dt)

        # Measure discontinuity as max absolute jump in second derivative
        d2_jumps = jnp.abs(d2[1:] - d2[:-1])
        max_jump = jnp.max(d2_jumps)

        results[name] = {
            't': t[1:-1],
            'd2': d2,
            'max_jump': max_jump,
            'values': values
        }

        print(f"{name}:")
        print(f"  Max jump in second derivative: {max_jump:.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Values
    axes[0, 0].plot(t, results["Without prefilter (Catmull-Rom)"]['values'],
                    'b-', label='Catmull-Rom', alpha=0.7)
    axes[0, 0].plot(t, results["With prefilter (B-spline)"]['values'],
                    'r--', label='B-spline', alpha=0.7)
    axes[0, 0].set_xlabel('t (diagonal position)')
    axes[0, 0].set_ylabel('Interpolated value')
    axes[0, 0].set_title('Interpolated values along diagonal')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Second derivatives
    for name, color, style in [("Without prefilter (Catmull-Rom)", 'b', '-'),
                                ("With prefilter (B-spline)", 'r', '--')]:
        axes[0, 1].plot(results[name]['t'], results[name]['d2'],
                       color=color, linestyle=style, label=name, alpha=0.7)
    axes[0, 1].set_xlabel('t (diagonal position)')
    axes[0, 1].set_ylabel("f''(t)")
    axes[0, 1].set_title('Second derivative along diagonal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Zoom in on second derivative to see discontinuities
    t_zoom = (results["Without prefilter (Catmull-Rom)"]['t'] > 10) & \
             (results["Without prefilter (Catmull-Rom)"]['t'] < 14)
    for name, color, style in [("Without prefilter (Catmull-Rom)", 'b', '-'),
                                ("With prefilter (B-spline)", 'r', '--')]:
        axes[1, 0].plot(results[name]['t'][t_zoom], results[name]['d2'][t_zoom],
                       color=color, linestyle=style, label=name, linewidth=1.5)

    # Add vertical lines at integer crossings
    for i in range(11, 14):
        axes[1, 0].axvline(x=i, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('t (diagonal position)')
    axes[1, 0].set_ylabel("f''(t)")
    axes[1, 0].set_title('Second derivative (zoomed) - note kinks at integer boundaries')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Show the test image
    axes[1, 1].imshow(image, cmap='viridis', origin='lower')
    axes[1, 1].plot([4, 28], [4, 28], 'r-', linewidth=2, label='Sample path')
    axes[1, 1].set_title('Test image with sample path')
    axes[1, 1].legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'c2_continuity_test.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Check that B-spline has smaller jumps
    cr_jump = results["Without prefilter (Catmull-Rom)"]['max_jump']
    bs_jump = results["With prefilter (B-spline)"]['max_jump']

    print(f"\nCatmull-Rom max d^2 jump: {cr_jump:.4f}")
    print(f"B-spline max d^2 jump:    {bs_jump:.4f}")
    print(f"Ratio (CR/BS):           {cr_jump/bs_jump:.1f}x")

    assert bs_jump < cr_jump, "B-spline should have smaller second derivative jumps"
    return True


if __name__ == "__main__":
    test_c2_continuity(output_dir="images")
