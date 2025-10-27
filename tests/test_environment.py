"""Tests for verifying the environment setup, including JAX CUDA installation."""

import jax
import jax.numpy as jnp
import pytest


def has_gpu() -> bool:
    """Check if GPU devices are available."""
    return any(d.platform == "gpu" for d in jax.devices())


def test_jax_installation() -> None:
    """Test that JAX is properly installed and can perform basic operations."""
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    result = x + y
    expected = jnp.array([5.0, 7.0, 9.0])
    assert jnp.allclose(result, expected), "JAX basic operations failed"


def test_jax_cuda_available() -> None:
    """Test that JAX can detect CUDA devices."""
    devices = jax.devices()
    assert len(devices) > 0, "No JAX devices found"

    # Check if any GPU devices are available
    gpu_devices = [d for d in devices if d.platform == "gpu"]

    if len(gpu_devices) == 0:
        pytest.skip(f"No GPU devices found. Running on CPU only. Available devices: {devices}")


@pytest.mark.skipif(not has_gpu(), reason="No GPU available")
def test_jax_cuda_computation() -> None:
    """Test that JAX can perform computations on GPU."""

    # Perform a simple computation on GPU
    @jax.jit
    def simple_computation(x: jax.Array) -> jax.Array:
        return jnp.sin(x) ** 2 + jnp.cos(x) ** 2

    x = jnp.linspace(0, 2 * jnp.pi, 1000)
    result = simple_computation(x)

    # Should be close to 1.0 everywhere (sin^2 + cos^2 = 1)
    assert jnp.allclose(result, 1.0, atol=1e-6), "GPU computation produced incorrect results"


def test_jax_cuda_device_info() -> None:
    """Print CUDA device information for debugging."""
    devices = jax.devices()
    print(f"\nTotal JAX devices: {len(devices)}")

    for i, device in enumerate(devices):
        print(f"Device {i}: {device}")
        print(f"  Platform: {device.platform}")
        print(f"  Device kind: {device.device_kind}")

    gpu_devices = [d for d in devices if d.platform == "gpu"]
    print(f"\nGPU devices: {len(gpu_devices)}")
