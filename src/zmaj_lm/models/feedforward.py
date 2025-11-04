import flax.linen as nn
import jax

from zmaj_lm.config.model_config import TransformerConfig


class FeedForward(nn.Module):
    config: TransformerConfig

    def setup(self) -> None:
        self.dense_1 = nn.Dense(self.config.mlp_dim, use_bias=self.config.use_bias)
        self.dense_2 = nn.Dense(self.config.hidden_dim, use_bias=self.config.use_bias)
        self.activation = nn.gelu
        self.dropout = nn.Dropout(rate=self.config.dropout_rate, rng_collection="dropout")

    def __call__(self, x: jax.Array, deterministic: bool = False) -> jax.Array:
        """Apply the feedforward network.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            deterministic: If True, disable dropout (for inference)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense_2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x
