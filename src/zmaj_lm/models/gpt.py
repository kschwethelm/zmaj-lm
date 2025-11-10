import flax.linen as nn
import jax

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.embeddings import TokenEmbedding
from zmaj_lm.models.transformer_block import TransformerBlock
from zmaj_lm.utils.masks import create_decoder_mask


class GPTModel(nn.Module):
    """GPT-style Transformer Language Model.

    Consists of:
    - Token embedding layer with learned positional encodings
    - Stack of Transformer blocks with pre-norm architecture
    - Final layer normalization
    - Output projection tied to input embeddings

    Designed for autoregressive language modeling tasks.
    """

    config: TransformerConfig

    def setup(self) -> None:
        """Initialize model components."""
        self.embedding = TokenEmbedding(config=self.config)
        self.transformer_blocks = [
            TransformerBlock(config=self.config) for _ in range(self.config.num_layers)
        ]
        self.final_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps)

    def __call__(
        self,
        input_ids: jax.Array,
        deterministic: bool = False,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of the GPT model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            deterministic: If True, disable dropout (for inference)
            attention_mask: Optional attention mask of shape (batch, seq_len) where True/1 indicates
                          valid tokens and False/0 indicates padding

        Returns:
            Logits over vocabulary of shape (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.shape[1]
        mask = create_decoder_mask(seq_len, attention_mask=attention_mask)  # (1, seq_len, seq_len)

        x = self.embedding.encode(
            input_ids, deterministic=deterministic
        )  # (batch, seq_len, hidden_dim)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, deterministic=deterministic)  # (batch, seq_len, hidden_dim)

        x = self.final_layernorm(x)  # (batch, seq_len, hidden_dim)
        logits = self.embedding.decode(x)  # (batch, seq_len, vocab_size)

        return logits
