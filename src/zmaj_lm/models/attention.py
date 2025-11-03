import flax.linen as nn
import jax
import jax.numpy as jnp

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.utils.masks import mask_to_bias
from zmaj_lm.utils.shapes import merge_heads_transposed, split_heads_transposed


def scaled_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None = None,
    dropout_rate: float = 0.0,
    dropout_rng: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute scaled dot-product attention.

    Formula: Attention(Q, K, V) = softmax((QK^T / √d_k) + M) V

    Args:
        query: jax.Array of shape (batch, n_heads, seq_len_q, d_head)
        key: jax.Array of shape (batch, n_heads, seq_len_k, d_head)
        value: jax.Array of shape (batch, n_heads, seq_len_v, d_head)
        mask: jax.Array broadcastable to (batch, 1, seq_len_q, seq_len_k), default is None
        dropout_rate: float, dropout rate to apply on attention weights
        dropout_rng: jax.Array, random number generator for dropout <- required if dropout_rate > 0 and training

    Returns:
        output: Attention output of shape (batch, n_heads, seq_len_q, d_head)
        attention_weights: Attention probabilities (batch, n_heads, seq_len_q, seq_len_k)
    """
    d_head = query.shape[-1]
    # jnp.einsum('bhqd,bhkd->bhqk', q, k) for the matrix multiplication Q @ K^T
    scores = jnp.einsum("bhqd,bhkd->bhqk", query, key) / jnp.sqrt(d_head)

    if mask is not None:
        bias = mask_to_bias(mask, dtype=scores.dtype)
        scores += bias

    attention_weights = jax.nn.softmax(scores, axis=-1)  # normalize over keys

    if dropout_rate > 0.0 and dropout_rng is not None:
        keep_prob = 1.0 - dropout_rate
        dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attention_weights.shape)
        attention_weights = (
            attention_weights * dropout_mask / keep_prob
        )  # Scale to maintain expectation

    output = jnp.einsum("bhqk,bhkd->bhqd", attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.

    Applies learned linear projections to create queries, keys, and values,
    then computes scaled dot-product attention across multiple heads in parallel.
    """

    config: TransformerConfig
    return_attention_weights: bool = False

    def setup(self) -> None:
        """Initialize the linear projection layers.

        Note: For efficiency, Q/K/V projections could be fused into a single
        dense layer, but we keep them separate for clarity and modularity.
        """
        self.q_proj = nn.Dense(self.config.hidden_dim, use_bias=self.config.use_bias, name="query")
        self.k_proj = nn.Dense(self.config.hidden_dim, use_bias=self.config.use_bias, name="key")
        self.v_proj = nn.Dense(self.config.hidden_dim, use_bias=self.config.use_bias, name="value")
        self.out_proj = nn.Dense(self.config.hidden_dim, use_bias=self.config.use_bias, name="out")

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask, broadcastable to (batch, n_heads, seq_len, seq_len)
            deterministic: If True, disable dropout (for inference)

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

        # Prepare dropout RNG key if needed
        dropout_rng = None
        if not deterministic and self.config.attention_dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Compute attention
        attn_output, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout_rate=self.config.attention_dropout_rate if not deterministic else 0.0,
            dropout_rng=dropout_rng,
        )

        # Merge attention heads
        attn_output = merge_heads_transposed(attn_output)  # (batch, seq_len, hidden_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        if self.return_attention_weights:
            return output, attention_weights
        return output
