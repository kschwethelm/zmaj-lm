"""Attention mask utilities for Transformer models."""

import jax
import jax.numpy as jnp


def create_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bool_) -> jax.Array:
    """Create a causal attention mask for sequences of length `seq_len`

    Args:
        seq_len: Length of the sequence.
        dtype: Data type of the returned mask.

    Returns:
        A (1, seq_len, seq_len) JAX array representing the causal mask.
    """
    mask = jnp.tril(jnp.ones(shape=(1, seq_len, seq_len), dtype=dtype))
    return mask


def create_padding_mask(lengths: jax.Array, max_len: int) -> jax.Array:
    """Create a padding mask based on sequence lengths.

    Args:
        lengths: JAX array of shape (batch_size,) containing the lengths of each sequence.
        max_len: Maximum sequence length.

    Returns:
        A (batch_size, max_len) JAX boolean array where True indicates valid tokens and False indicates padding.
    """
    pos_idx = jnp.arange(max_len)[None, :]  # Shape (1, max_len)
    mask = pos_idx < lengths[:, None]  # Shape (batch_size, max_len)
    return mask


def combine_masks(*masks: jax.Array) -> jax.Array:
    """Combine multiple attention masks using logical AND.

    Args:
        *masks: Variable number of JAX arrays representing attention masks.

    Returns:
        A JAX array representing the combined attention mask.
    """
    if not masks:
        raise ValueError("At least one mask must be provided to combine_masks.")

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = jnp.logical_and(combined_mask, mask)
    return combined_mask.astype(masks[0].dtype)


def mask_to_bias(
    mask: jax.Array, dtype: jnp.dtype = jnp.float32, mask_value: float = -1e10
) -> jax.Array:
    """Convert a boolean attention mask to an attention bias.

    Args:
        mask: Attention mask
        dtype: Data type of the returned bias.
        mask_value: Value to use for masked positions.

    Returns:
        A JAX array representing the attention bias.
    """
    return jnp.where(mask, 0.0, mask_value).astype(dtype)


def create_decoder_mask(
    seq_len: int, lengths: jax.Array | None, dtype: jnp.dtype = jnp.bool_
) -> jax.Array:
    """Create a combined causal and padding mask for decoder self-attention.

    Args:
        seq_len: Length of the sequence.
        lengths: JAX array of shape (batch_size,) containing the lengths of each sequence, or None.
        dtype: Data type of the returned mask.

    Returns:
        A (batch_size, seq_len, seq_len) JAX array representing the combined mask
    """
    causal_mask = create_causal_mask(seq_len, dtype=dtype)  # Shape (1, seq_len, seq_len)

    if lengths is None:
        return causal_mask  # Shape (1, seq_len, seq_len)

    padding_mask = create_padding_mask(lengths, seq_len)  # Shape (batch_size, seq_len)
    padding_mask = padding_mask[:, None, :]  # Shape (batch_size, 1, seq_len)

    combined_mask = combine_masks(causal_mask, padding_mask)  # Shape (batch_size, seq_len, seq_len)
    return combined_mask
