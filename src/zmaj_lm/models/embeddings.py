import torch
import torch.nn as nn

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.positional_encoding import get_positional_encoding_module


class TokenEmbedding(nn.Module):
    """Token embedding layer with positional encoding.

    Converts input token IDs to dense vectors and adds positional information.
    Used as the input layer for transformer models.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize token embedding layer.

        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config

        # Token embeddings: vocab_size -> hidden_dim
        self.token_embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
        )
        # Initialize with normal distribution (stddev=0.02)
        nn.init.normal_(self.token_embed.weight, std=0.02)

        self.pos_encoding = get_positional_encoding_module(config)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings with positional encoding.

        embeddings = embedding_matrix[token_ids] + positional_encoding

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, hidden_dim)
        """
        x = self.token_embed(input_ids)  # (batch, seq_len, hidden_dim)
        x = self.pos_encoding(x)  # Add positional encoding
        x = self.dropout(x)  # Apply dropout
        return x

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states back to vocabulary logits using tied weights.

        Uses the same embedding matrix transposed (weight tying) to reduce
        parameters and improve performance.

        logits = hidden_states @ self.token_embed.weight.T

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return torch.matmul(
            hidden_states, self.token_embed.weight.T
        )  # (batch, seq_len, vocab_size)
