import torch
import torch.nn as nn
import torch.nn.functional as F

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.utils.shapes import merge_heads_transposed, split_heads_transposed


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
        attn_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=(self.config.attention_dropout_rate if self.training else 0.0),
        )

        # Merge attention heads
        attn_output = merge_heads_transposed(attn_output)  # (batch, seq_len, hidden_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        return output
