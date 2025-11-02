import jax


def split_heads(x: jax.Array, n_heads: int) -> jax.Array:
    """Split last dimension of `x` into `n_heads` heads.

    Args:
        x: Input array of shape (batch, seq_len, d_model).
        n_heads: Number of heads to split into.

    Returns:
        Array of shape (batch, seq_len, n_heads, d_head) where d_head = d_model // n_heads.
    """
    batch, seq_len, d_model = x.shape
    assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    d_head = d_model // n_heads
    x = x.reshape(batch, seq_len, n_heads, d_head)
    return x


def split_heads_transposed(x: jax.Array, n_heads: int) -> jax.Array:
    """Split last dimension of `x` into `n_heads` heads and transpose for attention.

    Args:
        x: Input array of shape (batch, seq_len, d_model).
        n_heads: Number of heads to split into.

    Returns:
        Array of shape (batch, n_heads, seq_len, d_head) where d_head = d_model // n_heads.
        The transpose places n_heads before seq_len for efficient batched attention.
    """
    return split_heads(x, n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: jax.Array) -> jax.Array:
    """Merge heads dimension back into last dimension.

    Args:
        x: Input array of shape (batch, seq_len, n_heads, d_head).

    Returns:
        Array of shape (batch, seq_len, d_model) where d_model = n_heads * d_head.
    """
    batch, seq_len, n_heads, d_head = x.shape
    d_model = n_heads * d_head
    x = x.reshape(batch, seq_len, d_model)
    return x


def merge_heads_transposed(x: jax.Array) -> jax.Array:
    """Merge heads from transposed layout back into last dimension.

    Args:
        x: Input array of shape (batch, n_heads, seq_len, d_head).

    Returns:
        Array of shape (batch, seq_len, d_model) where d_model = n_heads * d_head.
    """
    batch, n_heads, seq_len, d_head = x.shape
    # Transpose back: [B, H, L, D] → [B, L, H, D]
    x = x.transpose(0, 2, 1, 3)
    # Merge: [B, L, H, D] → [B, L, H*D]
    d_model = n_heads * d_head
    x = x.reshape(batch, seq_len, d_model)
    return x


def assert_shape(
    x: jax.Array, expected_shape: tuple[int | None, ...], name: str = "tensor"
) -> None:
    """Assert that `x` has the expected shape.

    Args:
        x: Input array.
        expected_shape: Expected shape, with `None` for dimensions that can vary.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If the shape does not match.
    """
    if len(x.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has incorrect number of dimensions. "
            f"Expected {len(expected_shape)}, got {len(x.shape)}."
        )
    for i, (dim, expected_dim) in enumerate(zip(x.shape, expected_shape)):
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(
                f"{name} has incorrect shape at dimension {i}. Expected {expected_dim}, got {dim}."
            )


def shape_str(x: jax.Array, name: str = "") -> str:
    """Format shape as readable string for logging.

    Args:
        x: Input array.
        name: Optional name for the tensor.

    Returns:
        Formatted string with shape and dtype information.
    """
    prefix = f"{name}: " if name else ""
    return f"{prefix}{x.shape} {x.dtype}"
