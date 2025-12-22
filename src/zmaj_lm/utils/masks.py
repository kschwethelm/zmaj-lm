"""Attention mask utilities for Transformer models."""

import torch


def create_causal_mask(
    seq_len: int, device: torch.device | str, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Create a causal attention mask for autoregressive generation.

    Tokens at position i can only attend to positions j where j <= i (lower triangular).
    This prevents the model from attending to future positions during training.

    Args:
        seq_len: Length of the sequence.
        dtype: Data type of the returned mask.
        device: Device on which to create the mask.

    Returns:
        A (1, seq_len, seq_len) PyTorch tensor representing the causal mask.
    """
    mask = torch.tril(torch.ones(1, seq_len, seq_len, dtype=dtype, device=device))
    return mask


def create_padding_mask(
    lengths: torch.Tensor, max_len: int, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Create a padding mask based on sequence lengths.

    Args:
        lengths: PyTorch tensor of shape (batch_size,) containing the lengths of each sequence.
        max_len: Maximum sequence length.
        dtype: Data type of the returned mask.

    Returns:
        A (batch_size, max_len) PyTorch tensor where True indicates valid tokens and False indicates padding.
    """
    pos_idx = torch.arange(max_len, device=lengths.device)[None, :]  # Shape (1, max_len)
    mask = pos_idx < lengths[:, None]  # Shape (batch_size, max_len)
    return mask.to(dtype)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    """Combine multiple attention masks using logical AND.

    Args:
        *masks: Variable number of PyTorch tensors representing attention masks.

    Returns:
        A PyTorch tensor representing the combined attention mask.
    """
    if not masks:
        raise ValueError("At least one mask must be provided to combine_masks.")

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = torch.logical_and(combined_mask, mask)
    return combined_mask.to(masks[0].dtype)


def mask_to_bias(
    mask: torch.Tensor, dtype: torch.dtype = torch.float32, mask_value: float = -1e10
) -> torch.Tensor:
    """Convert a boolean attention mask to an attention bias.

    Args:
        mask: Attention mask
        dtype: Data type of the returned bias.
        mask_value: Value to use for masked positions.

    Returns:
        A PyTorch tensor representing the attention bias.
    """
    return torch.where(mask, 0.0, mask_value).to(dtype)


def create_packing_mask(
    batch_size: int, seq_len: int, device: torch.device | str, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Create an all-ones attention mask for packed sequences without padding.

    For packed sequences where all tokens are valid, all positions should attend.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Length of each sequence.
        dtype: Data type of the returned mask.
        device: Device on which to create the mask.

    Returns:
        A (batch_size, seq_len) PyTorch tensor of all ones.
    """
    return torch.ones(batch_size, seq_len, dtype=dtype, device=device)


def create_block_diagonal_mask(
    doc_ids: torch.Tensor, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
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
    mask = doc_ids[:, :, None] == doc_ids[:, None, :]
    return mask.to(dtype)


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bool,
    causal: bool = True,
) -> torch.Tensor:
    """Create a sliding window attention mask.

    For causal sliding window (like Mistral):
    - Position i attends to positions [max(0, i - window_size + 1), i]
    - Creates a diagonal band in the lower triangular region

    For bidirectional sliding window (like Longformer):
    - Position i attends to positions [max(0, i - window_size), min(seq_len, i + window_size + 1)]
    - Creates a diagonal band around the main diagonal

    Args:
        seq_len: Length of the sequence.
        window_size: Size of the attention window. Each position can attend to
                     at most window_size tokens (including itself for causal).
        device: Device on which to create the mask.
        dtype: Data type of the returned mask.
        causal: If True, use causal sliding window. If False, use bidirectional.

    Returns:
        A (1, seq_len, seq_len) PyTorch tensor representing the sliding window mask.
        True indicates that attention is allowed, False indicates it's masked out.
    """
    # Create position indices
    row_idx = torch.arange(seq_len, device=device)[:, None]  # (seq_len, 1)
    col_idx = torch.arange(seq_len, device=device)[None, :]  # (1, seq_len)

    if causal:
        # Causal: position i attends to [max(0, i - window_size + 1), i]
        # This is equivalent to: (i - window_size + 1) <= j <= i
        mask = (col_idx >= row_idx - window_size + 1) & (col_idx <= row_idx)
    else:
        # Bidirectional: position i attends to [i - window_size, i + window_size]
        # This is equivalent to: |i - j| <= window_size
        mask = torch.abs(row_idx - col_idx) <= window_size

    # Add batch dimension: (seq_len, seq_len) -> (1, seq_len, seq_len)
    mask = mask.unsqueeze(0).to(dtype)
    return mask


def create_decoder_mask(
    seq_len: int,
    device: torch.device | str,
    attention_mask: torch.Tensor | None = None,
    window_size: int | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Create a combined causal and attention mask for decoder self-attention.

    Args:
        seq_len: Length of the sequence.
        attention_mask: Optional attention mask. Can be:
                        - None: Only causal masking is applied
                        - (batch_size, seq_len): Padding mask, True for valid tokens
                        - (batch_size, seq_len, seq_len): Packing/block-diagonal mask
        window_size: Optional sliding window size. If None, uses full causal attention.
                     If provided, applies sliding window attention with the specified window size.
        dtype: Data type of the returned mask.
        device: Device on which to create the mask.

    Returns:
        A (batch_size, seq_len, seq_len) or (1, seq_len, seq_len) PyTorch tensor representing
        the combined mask. Shape depends on whether attention_mask is provided.
    """
    # Create base causal mask (either full or sliding window)
    if window_size is None:
        causal_mask = create_causal_mask(
            seq_len, dtype=dtype, device=device
        )  # Shape (1, seq_len, seq_len)
    else:
        causal_mask = create_sliding_window_mask(
            seq_len, window_size, dtype=dtype, device=device, causal=True
        )  # Shape (1, seq_len, seq_len)

    if attention_mask is None:
        return causal_mask

    # Handle 2D padding mask: (batch_size, seq_len) -> (batch_size, 1, seq_len)
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, :]  # (batch_size, 1, seq_len)

    combined_mask = combine_masks(
        causal_mask, attention_mask
    )  # Shape (batch_size, seq_len, seq_len)
    return combined_mask
