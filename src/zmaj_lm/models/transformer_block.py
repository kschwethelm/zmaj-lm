import torch
import torch.nn as nn

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.attention import MultiHeadAttention
from zmaj_lm.models.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A single Transformer block with pre-norm architecture.

    Implements the pre-LayerNorm transformer block used in modern LLMs (GPT-2/3, etc.):
    - Layer normalization before each sub-layer (pre-norm)
    - Multi-head self-attention with residual connection
    - Feed-forward network with residual connection
    - Optional residual dropout for regularization
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the transformer block components.

        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config=config)
        self.feedforward = FeedForward(config=config)
        self.layernorm_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.layernorm_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.residual_dropout = nn.Dropout(p=config.residual_dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the transformer block with pre-norm architecture.

        Architecture:
            x = x + dropout(attention(norm(x)))
            x = x + dropout(ffn(norm(x)))

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask, broadcastable to (batch, n_heads, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # Attention block with residual connection
        residual = x
        x = self.layernorm_1(x)
        x = self.attention(x, mask=mask)
        x = self.residual_dropout(x)
        x = x + residual

        # FFN block with residual connection
        residual = x
        x = self.layernorm_2(x)
        x = self.feedforward(x)
        x = self.residual_dropout(x)
        x = x + residual

        return x
