import flax.linen as nn
import jax
import jax.numpy as jnp

from zmaj_lm.config.model_config import TransformerConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.

    Implements the sinusoidal positional encoding as described in
    "Attention is All You Need" (Vaswani et al., 2017).

    This encoding allows the model to learn the position of tokens
    in a sequence without using learned parameters.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Precompute sinusoidal positional encodings."""
        # Create angles for all positions and dimensions
        position = jnp.arange(self.config.max_seq_len)  # [seq_len,]
        dim_indices = jnp.arange(0, self.config.hidden_dim, 2)  # [hidden_dim/2,]
        div_term = jnp.power(10000.0, dim_indices / self.config.hidden_dim)  # [hidden_dim/2,]
        angles = position[:, None] / div_term[None, :]  # [seq_len, hidden_dim/2]

        # Interleave sin and cos into the positional encodings
        encodings = jnp.zeros((self.config.max_seq_len, self.config.hidden_dim))
        encodings = encodings.at[:, 0::2].set(jnp.sin(angles))
        encodings = encodings.at[:, 1::2].set(jnp.cos(angles))

        self.encodings = encodings

    def __call__(self, x: jax.Array) -> jax.Array:
        """Add sinusoidal positional encodings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim) with positional encodings added.
        """
        seq_len = x.shape[1]
        assert seq_len <= self.config.max_seq_len, (
            f"Input sequence length {seq_len} exceeds maximum "
            f"supported length {self.config.max_seq_len}"
        )
        pos_enc = self.encodings[:seq_len, :]  # [seq_len, hidden_dim]
        return x + pos_enc[None, :, :]  # Broadcast to (batch, seq_len, hidden_dim)


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encoding.

    Implements learned positional embeddings as used in GPT-style models.
    Each position in the input sequence has a corresponding learned embedding
    vector that is added to the token embeddings.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize learned positional embeddings."""
        self.pos_embedding = nn.Embed(
            num_embeddings=self.config.max_seq_len,
            features=self.config.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Add learned positional embeddings to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim) with positional embeddings added.
        """
        seq_len = x.shape[1]
        assert seq_len <= self.config.max_seq_len, (
            f"Input sequence length {seq_len} exceeds maximum "
            f"supported length {self.config.max_seq_len}"
        )
        position_ids = jnp.arange(seq_len)  # [seq_len]
        pos_emb = self.pos_embedding(position_ids)  # [seq_len, hidden_dim]
        return x + pos_emb[None, :, :]  # Broadcast to (batch, seq_len, hidden_dim)


def get_positional_encoding_module(
    config: TransformerConfig,
) -> SinusoidalPositionalEncoding | LearnedPositionalEncoding:
    """Factory function to get the appropriate positional encoding module.

    Args:
        config: TransformerConfig instance with model configuration.

    Returns:
        An instance of either SinusoidalPositionalEncoding or LearnedPositionalEncoding
        based on the pos_encoding_type specified in the config.
    """
    if config.pos_encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(config=config)
    else:
        return LearnedPositionalEncoding(config=config)
