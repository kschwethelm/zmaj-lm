import flax.linen as nn
import jax

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

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize the transformer block components."""
        self.attention = MultiHeadAttention(config=self.config)
        self.feedforward = FeedForward(config=self.config)
        self.layernorm_1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        self.layernorm_2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps)
        self.residual_dropout = nn.Dropout(
            rate=self.config.residual_dropout_rate, rng_collection="dropout"
        )

    def __call__(
        self, x: jax.Array, mask: jax.Array | None = None, deterministic: bool = False
    ) -> jax.Array:
        """Apply the transformer block with pre-norm architecture.

        Architecture:
            x = x + dropout(attention(norm(x)))
            x = x + dropout(ffn(norm(x)))

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask, broadcastable to (batch, n_heads, seq_len, seq_len)
            deterministic: If True, disable dropout (for inference)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # Attention block with residual connection
        residual = x
        x = self.layernorm_1(x)
        x = self.attention(x, mask=mask, deterministic=deterministic)
        x = self.residual_dropout(x, deterministic=deterministic)
        x = x + residual

        # FFN block with residual connection
        residual = x
        x = self.layernorm_2(x)
        x = self.feedforward(x, deterministic=deterministic)
        x = self.residual_dropout(x, deterministic=deterministic)
        x = x + residual

        return x
