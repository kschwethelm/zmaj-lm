"""Attention mask utilities for Transformer models."""

import jax
import jax.numpy as jnp


def create_causal_mask(seq_len: int, dtype: jnp.dtype = jnp.bool_) -> jax.Array:
    """Create a causal attention mask for autoregressive generation.

    Tokens at position i can only attend to positions j where j <= i (lower triangular).
    This prevents the model from attending to future positions during training.

    Args:
        seq_len: Length of the sequence.
        dtype: Data type of the returned mask.

    Returns:
        A (1, seq_len, seq_len) JAX array representing the causal mask.
    """
    mask = jnp.tril(jnp.ones(shape=(1, seq_len, seq_len), dtype=dtype))
    return mask


def create_padding_mask(
    lengths: jax.Array, max_len: int, dtype: jnp.dtype = jnp.bool_
) -> jax.Array:
    """Create a padding mask based on sequence lengths.

    Args:
        lengths: JAX array of shape (batch_size,) containing the lengths of each sequence.
        max_len: Maximum sequence length.
        dtype: Data type of the returned mask.

    Returns:
        A (batch_size, max_len) JAX array where True indicates valid tokens and False indicates padding.
    """
    pos_idx = jnp.arange(max_len)[None, :]  # Shape (1, max_len)
    mask = pos_idx < lengths[:, None]  # Shape (batch_size, max_len)
    return mask.astype(dtype)


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


def create_packing_mask(batch_size: int, seq_len: int, dtype: jnp.dtype = jnp.bool_) -> jax.Array:
    """Create an all-ones attention mask for packed sequences without padding.

    For packed sequences where all tokens are valid, all positions should attend.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Length of each sequence.
        dtype: Data type of the returned mask.

    Returns:
        A (batch_size, seq_len) JAX array of all ones.
    """
    return jnp.ones((batch_size, seq_len), dtype=dtype)


def create_block_diagonal_mask(doc_ids: jax.Array, dtype: jnp.dtype = jnp.bool_) -> jax.Array:
    """Create block-diagonal attention mask to prevent cross-document attention.

    Used with packed sequences to prevent attention from crossing document boundaries.
    Tokens can only attend to other tokens from the same document.

    Args:
        doc_ids: Document IDs of shape (batch_size, seq_len) indicating which
                 document each token belongs to.
        dtype: Data type of the returned mask.

    Returns:
        Mask of shape (batch_size, seq_len, seq_len) where mask[b, i, j] is True
        if tokens i and j belong to the same document, False otherwise.
    """
    # Create pairwise comparison: doc_ids[b, i] == doc_ids[b, j]
    # Broadcasting: (batch_size, seq_len, 1) == (batch_size, 1, seq_len)
    mask = doc_ids[:, :, jnp.newaxis] == doc_ids[:, jnp.newaxis, :]
    return mask.astype(dtype)


def create_decoder_mask(
    seq_len: int, attention_mask: jax.Array | None = None, dtype: jnp.dtype = jnp.bool_
) -> jax.Array:
    """Create a combined causal and attention mask for decoder self-attention.

    Args:
        seq_len: Length of the sequence.
        attention_mask: Optional attention mask. Can be:
                        - None: Only causal masking is applied
                        - (batch_size, seq_len): Padding mask, True for valid tokens
                        - (batch_size, seq_len, seq_len): Packing/block-diagonal mask
        dtype: Data type of the returned mask.

    Returns:
        A (batch_size, seq_len, seq_len) or (1, seq_len, seq_len) JAX array representing
        the combined mask. Shape depends on whether attention_mask is provided.
    """
    causal_mask = create_causal_mask(seq_len, dtype=dtype)  # Shape (1, seq_len, seq_len)

    if attention_mask is None:
        return causal_mask

    # Handle 2D padding mask: (batch_size, seq_len) -> (batch_size, 1, seq_len)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, :]  # (batch_size, 1, seq_len)

    combined_mask = combine_masks(
        causal_mask, attention_mask
    )  # Shape (batch_size, seq_len, seq_len)
    return combined_mask
