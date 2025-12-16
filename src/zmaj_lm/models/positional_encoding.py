import torch
import torch.nn as nn

from zmaj_lm.config.model_config import TransformerConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.

    Implements the sinusoidal positional encoding as described in
    "Attention is All You Need" (Vaswani et al., 2017).

    This encoding allows the model to learn the position of tokens
    in a sequence without using learned parameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize sinusoidal positional encodings.

        Args:
            config: TransformerConfig instance with model configuration.
        """
        super().__init__()
        self.config = config

        # Create angles for all positions and dimensions
        position = torch.arange(self.config.max_seq_len, dtype=torch.float32)  # [seq_len,]
        dim_indices = torch.arange(
            0, self.config.hidden_dim, 2, dtype=torch.float32
        )  # [hidden_dim/2,]
        div_term = torch.pow(10000.0, dim_indices / self.config.hidden_dim)  # [hidden_dim/2,]
        angles = position[:, None] / div_term[None, :]  # [seq_len, hidden_dim/2]

        # Interleave sin and cos into the positional encodings
        encodings = torch.zeros((self.config.max_seq_len, self.config.hidden_dim))
        encodings[:, 0::2] = torch.sin(angles)
        encodings[:, 1::2] = torch.cos(angles)

        # Register as buffer (not a parameter, but part of the state)
        self.register_buffer("encodings", encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return x + pos_enc.unsqueeze(0)  # Broadcast to (batch, seq_len, hidden_dim)


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encoding.

    Implements learned positional embeddings as used in GPT-style models.
    Each position in the input sequence has a corresponding learned embedding
    vector that is added to the token embeddings.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize learned positional embeddings.

        Args:
            config: TransformerConfig instance with model configuration.
        """
        super().__init__()
        self.config = config

        self.pos_embedding = nn.Embedding(
            num_embeddings=self.config.max_seq_len,
            embedding_dim=self.config.hidden_dim,
        )
        # Initialize with normal distribution (stddev=0.02) as in GPT-2
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        position_ids = torch.arange(seq_len, device=x.device)  # [seq_len]
        pos_emb = self.pos_embedding(position_ids)  # [seq_len, hidden_dim]
        return x + pos_emb.unsqueeze(0)  # Broadcast to (batch, seq_len, hidden_dim)


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
