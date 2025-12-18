import torch
import torch.nn as nn
import torch.nn.functional as F

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.utils.masks import mask_to_bias
from zmaj_lm.utils.shapes import merge_heads_transposed, split_heads_transposed


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_rate: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax((QK^T / √d_k) + M) V

    Args:
        query: torch.Tensor of shape (batch, n_heads, seq_len_q, d_head)
        key: torch.Tensor of shape (batch, n_heads, seq_len_k, d_head)
        value: torch.Tensor of shape (batch, n_heads, seq_len_v, d_head)
        mask: torch.Tensor broadcastable to (batch, 1, seq_len_q, seq_len_k), default is None
        dropout_rate: float, dropout rate to apply on attention weights
        training: bool, whether the model is in training mode

    Returns:
        output: Attention output of shape (batch, n_heads, seq_len_q, d_head)
        attention_weights: Attention probabilities (batch, n_heads, seq_len_q, seq_len_k)
    """
    d_head = query.shape[-1]
    # torch.einsum('bhqd,bhkd->bhqk', q, k) for the matrix multiplication Q @ K^T
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) / torch.sqrt(
        torch.tensor(d_head, dtype=query.dtype, device=query.device)
    )

    if mask is not None:
        bias = mask_to_bias(mask, dtype=scores.dtype)
        scores = scores + bias

    attention_weights = F.softmax(scores, dim=-1)  # normalize over keys

    if training and dropout_rate > 0.0:
        attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)

    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.

    Applies learned linear projections to create queries, keys, and values,
    then computes scaled dot-product attention across multiple heads in parallel.
    """

    def __init__(self, config: TransformerConfig, return_attention_weights: bool = False) -> None:
        """Initialize the multi-head attention layer.

        Args:
            config: Transformer configuration
            return_attention_weights: Whether to return attention weights

        Note: For efficiency, Q/K/V projections could be fused into a single
        dense layer, but we keep them separate for clarity and modularity.
        """
        super().__init__()
        self.config = config
        self.return_attention_weights = return_attention_weights

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.use_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len) or (1, seq_len, seq_len)

        Returns:
            If return_attention_weights is False:
                Output tensor of shape (batch, seq_len, hidden_dim)
            If return_attention_weights is True:
                Tuple of (output, attention_weights) where attention_weights has shape
                (batch, n_heads, seq_len, seq_len)
        """
        # Project inputs to Q, K, V
        query = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        key = self.k_proj(x)  # (batch, seq_len, hidden_dim)
        value = self.v_proj(x)  # (batch, seq_len, hidden_dim)

        # Split into multiple attention heads
        query = split_heads_transposed(query, self.config.num_heads)
        key = split_heads_transposed(key, self.config.num_heads)
        value = split_heads_transposed(value, self.config.num_heads)
        # Shape after split: (batch, n_heads, seq_len, head_dim)

        # Add heads dimension to mask if needed: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Compute attention
        attn_output, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout_rate=self.config.attention_dropout_rate,
            training=self.training,
        )

        # Merge attention heads
        attn_output = merge_heads_transposed(attn_output)  # (batch, seq_len, hidden_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        if self.return_attention_weights:
            return output, attention_weights
        return output
