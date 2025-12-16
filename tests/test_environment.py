"""Environment setup and PyTorch installation tests."""

import torch


def test_torch_installation() -> None:
    """Test that PyTorch is installed and importable."""
    assert torch.__version__ is not None


def test_cuda_available() -> None:
    """Test whether CUDA is available (informational, not required)."""
    cuda_available = torch.cuda.is_available()
    # This is informational - tests should pass on CPU too
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")


def test_tensor_creation() -> None:
    """Test basic tensor creation works."""
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)
    assert x.dtype == torch.float32


def test_basic_operations() -> None:
    """Test basic PyTorch operations work."""
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    assert z.shape == (2, 3)

    # Test matrix multiplication
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    c = torch.matmul(a, b)
    assert c.shape == (2, 4)


def test_autograd() -> None:
    """Test that autograd works."""
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2
    y.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([4.0]))
