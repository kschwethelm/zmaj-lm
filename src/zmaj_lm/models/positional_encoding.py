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

    encodings: torch.Tensor

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


class NoPositionalEncoding(nn.Module):
    """No Positional Encoding.

    A no-op module that applies no positional encoding. Some architectures
    (e.g., certain recurrent models or models with alternative position
    awareness mechanisms) don't require explicit positional encodings.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize with no-op.

        Args:
            config: TransformerConfig instance with model configuration.
        """
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through input unchanged.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Input tensor unchanged
        """
        return x


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Implements rotary positional embeddings as described in "RoFormer: Enhanced
    Transformer with Rotary Position Embedding" (Su et al., 2021).

    Unlike additive positional encodings, RoPE encodes position information by
    rotating query and key vectors in the attention mechanism. This naturally
    incorporates relative position information and allows better extrapolation
    to sequence lengths beyond those seen during training.

    This implementation uses complex number representation for efficiency.

    IMPORTANT - Dimension Pairing Strategy:
    This implementation uses SLICED pairing: pairs first half with second half
    [x0, x_d/2], [x1, x_d/2+1], [x2, x_d/2+2], etc.

    Used by: LLaMA, LLaMA 2, LLaMA 3, Mistral, Gemma (most modern models)

    Note: PaLM, GPT-J, and RoFormer (original paper) use INTERLEAVED pairing.
    This implementation is compatible with loading pretrained LLaMA-family weights.
    """

    freqs_complex: torch.Tensor

    def __init__(
        self, config: TransformerConfig, head_dim: int | None = None, max_seq_len: int | None = None
    ) -> None:
        """Initialize RoPE with precomputed rotation frequencies.

        Args:
            config: TransformerConfig instance with model configuration
            head_dim: Head dimension (overrides config.head_dim if provided)
            max_seq_len: Maximum sequence length (overrides config.max_seq_len if provided)
        """
        super().__init__()
        self.config = config
        self.head_dim = head_dim if head_dim is not None else config.head_dim
        self.rope_theta = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
        self.max_seq_len = max_seq_len if max_seq_len is not None else config.max_seq_len

        # Precompute rotation frequencies for max_seq_len
        freqs_complex = self._compute_freqs(self.max_seq_len)
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("freqs_complex", freqs_complex, persistent=False)

    def _compute_freqs(self, seq_len: int) -> torch.Tensor:
        """Compute rotation frequencies for given sequence length.

        Args:
            seq_len: Maximum sequence length to precompute frequencies for

        Returns:
            Complex frequency tensor of shape [seq_len, head_dim/2]
        """
        # Compute theta_i = rope_theta^(-2i/d) for i in [0, head_dim/2)
        # These are the rotation frequencies for each dimension pair
        dim_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        theta = self.rope_theta ** (-dim_indices / self.head_dim)  # [head_dim/2]

        # Create position indices for all possible positions
        positions = torch.arange(seq_len, dtype=torch.float32)  # [seq_len]

        # Compute all angles: position * theta for each (position, theta) pair
        # Shape: [seq_len, head_dim/2]
        angles = positions[:, None] * theta[None, :]

        # Convert to complex numbers for efficient rotation: e^(i*theta) = cos(theta) + i*sin(theta)
        # This represents the rotation matrix in complex form
        return torch.polar(torch.ones_like(angles), angles)  # [seq_len, head_dim/2]

    def extend_sequence_length(self, new_max_len: int) -> None:
        """Extend the cached frequencies to support longer sequences.

        Useful for extrapolation experiments where you want to test the model
        on sequences longer than max_seq_len used during training.

        Args:
            new_max_len: New maximum sequence length (must be >= current max_len)
        """
        if new_max_len <= self.freqs_complex.shape[0]:
            return  # Already long enough

        # Recompute frequencies for the new length
        new_freqs = self._compute_freqs(new_max_len).to(self.freqs_complex.device)
        self.register_buffer("freqs_complex", new_freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding to query or key tensor.

        This is a no-op placeholder - actual rotation is applied in the attention
        mechanism via apply_rotary_pos_emb(). This forward method exists to maintain
        interface compatibility with other positional encoding modules.

        Args:
            x: Input tensor (not used, just passed through)

        Returns:
            Input tensor unchanged
        """
        return x

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, n_heads, seq_len, head_dim)

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        seq_len = q.shape[2]
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}"
        )

        # Get rotation frequencies for this sequence length
        freqs = self.freqs_complex[:seq_len, :]  # [seq_len, head_dim/2]

        # SLICED pairing: pair first half of dimensions with second half
        # Split head_dim into two halves: [x0, x1, ..., x_(d/2-1)] and [x_(d/2), ..., x_(d-1)]
        half_dim = self.head_dim // 2
        q_1 = q[..., :half_dim]  # First half: [batch, n_heads, seq_len, head_dim/2]
        q_2 = q[..., half_dim:]  # Second half: [batch, n_heads, seq_len, head_dim/2]
        k_1 = k[..., :half_dim]
        k_2 = k[..., half_dim:]

        # Stack as pairs: [batch, n_heads, seq_len, head_dim/2, 2]
        # IMPORTANT: .contiguous() is required because view_as_complex needs stride 1 on last dim
        q_reshaped = torch.stack([q_1, q_2], dim=-1).contiguous()
        k_reshaped = torch.stack([k_1, k_2], dim=-1).contiguous()

        # View as complex numbers: [batch, n_heads, seq_len, head_dim/2]
        q_complex = torch.view_as_complex(q_reshaped.float())
        k_complex = torch.view_as_complex(k_reshaped.float())

        # Apply rotation by complex multiplication
        # Broadcasting: freqs is [seq_len, head_dim/2], tensors are [batch, n_heads, seq_len, head_dim/2]
        q_rotated = q_complex * freqs.unsqueeze(0).unsqueeze(0)
        k_rotated = k_complex * freqs.unsqueeze(0).unsqueeze(0)

        # Convert back to real representation: [batch, n_heads, seq_len, head_dim/2, 2]
        q_real = torch.view_as_real(q_rotated)
        k_real = torch.view_as_real(k_rotated)

        # Unstack and concatenate: [first_half, second_half] -> [batch, n_heads, seq_len, head_dim]
        q_out = torch.cat([q_real[..., 0], q_real[..., 1]], dim=-1)
        k_out = torch.cat([k_real[..., 0], k_real[..., 1]], dim=-1)

        return q_out.type_as(q), k_out.type_as(k)


def get_positional_encoding_module(
    config: TransformerConfig,
) -> (
    SinusoidalPositionalEncoding
    | LearnedPositionalEncoding
    | RotaryPositionalEncoding
    | NoPositionalEncoding
):
    """Factory function to get the appropriate positional encoding module.

    Args:
        config: TransformerConfig instance with model configuration.

    Returns:
        An instance of SinusoidalPositionalEncoding, LearnedPositionalEncoding,
        RotaryPositionalEncoding, or NoPositionalEncoding based on the
        pos_encoding_type specified in the config.
    """
    if config.pos_encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(config=config)
    elif config.pos_encoding_type == "rope":
        return RotaryPositionalEncoding(config=config)
    elif config.pos_encoding_type == "none":
        return NoPositionalEncoding(config=config)
    else:
        return LearnedPositionalEncoding(config=config)
