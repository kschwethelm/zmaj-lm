"""Pytest configuration and fixtures for zmaj-lm tests."""

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the appropriate device (GPU if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def has_gpu() -> bool:
    """Return True if GPU devices are available."""
    return torch.cuda.is_available()


@pytest.fixture
def rng_generator() -> torch.Generator:
    """Return a PyTorch random generator with fixed seed for deterministic tests."""
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator
