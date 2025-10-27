"""Pytest configuration and fixtures for zmaj-lm tests."""

import jax
import pytest


@pytest.fixture(scope="session")
def jax_devices() -> list[jax.Device]:
    """Return all available JAX devices."""
    return jax.devices()


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Return True if GPU devices are available."""
    return any(d.platform == "gpu" for d in jax.devices())


@pytest.fixture
def rng_key() -> jax.Array:
    """Return a JAX random key for testing."""
    return jax.random.PRNGKey(0)
