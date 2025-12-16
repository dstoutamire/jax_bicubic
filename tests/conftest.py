"""Pytest fixtures for jax_bicubic tests."""

import jax
import pytest

# Use CPU for reproducibility
jax.config.update('jax_platform_name', 'cpu')
