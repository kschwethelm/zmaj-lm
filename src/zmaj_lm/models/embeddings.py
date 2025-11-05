import flax.linen as nn
import jax

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.positional_encoding import get_positional_encoding_module


class TokenEmbedding(nn.Module):
    """Token embedding layer with positional encoding.

    Converts input token IDs to dense vectors and adds positional information.
    Used as the input layer for transformer models.
    """

    config: TransformerConfig

    def setup(self) -> None:
        # Token embeddings: vocab_size -> hidden_dim
        self.token_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )

        self.pos_encoding = get_positional_encoding_module(self.config)
        self.dropout = nn.Dropout(rate=self.config.dropout_rate, rng_collection="dropout")

    def encode(self, input_ids: jax.Array, deterministic: bool = False) -> jax.Array:
        """Convert token IDs to embeddings with positional encoding.

        embeddings = embedding_matrix[token_ids] + positional_encoding

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            deterministic: If True, disable dropout

        Returns:
            Embeddings of shape (batch, seq_len, hidden_dim)
        """
        x = self.token_embed(input_ids)  # (batch, seq_len, hidden_dim)
        x = self.pos_encoding(x)  # Add positional encoding
        x = self.dropout(x, deterministic=deterministic)  # Apply dropout
        return x

    def decode(self, hidden_states: jax.Array) -> jax.Array:
        """Project hidden states back to vocabulary logits using tied weights.

        Uses the same embedding matrix transposed (weight tying) to reduce
        parameters and improve performance.

        logits = hidden_states @ self.token_embed.embeddings.T

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.token_embed.attend(hidden_states)  # (batch, seq_len, vocab_size)
