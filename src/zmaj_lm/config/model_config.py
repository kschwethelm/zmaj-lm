from typing import Literal

from pydantic import BaseModel, computed_field, model_validator


class TransformerBlockConfig(BaseModel):
    """Configuration for a single Transformer block.

    Contains all hyperparameters for the attention and feedforward layers
    within one transformer block, including dropout rates, normalization,
    and architectural choices.
    """

    hidden_dim: int = 256
    num_heads: int = 4  # number of attention heads (query heads)
    num_kv_heads: int | None = None  # number of key/value heads (None = num_heads for MHA)
    mlp_dim: int | None = None  # typically 4 * hidden_dim, None = auto

    dropout_rate: float = 0.1  # dropout rate (0.0 to 0.1 typically)
    layer_norm_eps: float = 1e-5
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    use_bias: bool = True  # whether to use bias in linear layers
    attention_dropout_rate: float | None = None  # if None, use dropout_rate
    residual_dropout_rate: float | None = None  # if None, use dropout_rate
    window_size: int | None = None  # sliding window size (None = full attention)

    pos_encoding_type: Literal["learned", "sinusoidal", "rope", "none"] = "learned"
    rope_theta: float = (
        10000.0  # RoPE base frequency (10000 standard, larger values for long context)
    )
    activation: Literal["gelu", "gelu_tanh", "silu", "relu", "geglu", "swiglu"] = "gelu"

    @model_validator(mode="after")
    def validate_num_heads(self) -> "TransformerBlockConfig":
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self

    @model_validator(mode="after")
    def validate_mlp_dim(self) -> "TransformerBlockConfig":
        if self.mlp_dim is None:
            object.__setattr__(self, "mlp_dim", 4 * self.hidden_dim)
        return self

    @model_validator(mode="after")
    def validate_attention_dropout_rate(self) -> "TransformerBlockConfig":
        if self.attention_dropout_rate is None:
            object.__setattr__(self, "attention_dropout_rate", self.dropout_rate)
        return self

    @model_validator(mode="after")
    def validate_residual_dropout_rate(self) -> "TransformerBlockConfig":
        if self.residual_dropout_rate is None:
            object.__setattr__(self, "residual_dropout_rate", self.dropout_rate)
        return self

    @model_validator(mode="after")
    def validate_num_kv_heads(self) -> "TransformerBlockConfig":
        if self.num_kv_heads is None:
            object.__setattr__(self, "num_kv_heads", self.num_heads)
        return self

    @model_validator(mode="after")
    def validate_kv_heads_divisibility(self) -> "TransformerBlockConfig":
        if self.num_kv_heads is not None and self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        return self

    @model_validator(mode="after")
    def validate_window_size(self) -> "TransformerBlockConfig":
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key/value head."""
        assert self.num_kv_heads is not None  # Set by validator
        return self.num_heads // self.num_kv_heads


class TransformerConfig(BaseModel):
    """Configuration for a GPT-style decoder-only Transformer model.

    This config defines model-level hyperparameters and delegates block-level
    configuration to TransformerBlockConfig. Supports both homogeneous
    (single block_config) and heterogeneous (list of block_configs) architectures.
    """

    # Model-level configuration
    vocab_size: int = 50257  # GPT-2 tokenizer size
    max_seq_len: int = 1024
    num_layers: int = 4  # number of transformer blocks

    # Block configuration (single config or list for heterogeneous layers)
    block_config: TransformerBlockConfig | list[TransformerBlockConfig] = TransformerBlockConfig()

    @model_validator(mode="after")
    def validate_block_config_length(self) -> "TransformerConfig":
        if isinstance(self.block_config, list) and len(self.block_config) != self.num_layers:
            raise ValueError(
                f"block_config list length ({len(self.block_config)}) "
                f"must match num_layers ({self.num_layers})"
            )
        return self

    def get_block_config(self, layer_idx: int) -> TransformerBlockConfig:
        """Get configuration for a specific transformer block.

        Args:
            layer_idx: Index of the layer (0-indexed)

        Returns:
            TransformerBlockConfig for the specified layer
        """
        if isinstance(self.block_config, list):
            return self.block_config[layer_idx]
        return self.block_config

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hidden_dim(self) -> int:
        """Convenience property to access hidden_dim from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].hidden_dim
        return self.block_config.hidden_dim

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        """Convenience property to access head_dim from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].head_dim
        return self.block_config.head_dim

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_kv_groups(self) -> int:
        """Convenience property to access num_kv_groups from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].num_kv_groups
        return self.block_config.num_kv_groups

    @computed_field  # type: ignore[prop-decorator]
    @property
    def window_size(self) -> int | None:
        """Convenience property to access window_size from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].window_size
        return self.block_config.window_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def layer_norm_eps(self) -> float:
        """Convenience property to access layer_norm_eps from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].layer_norm_eps
        return self.block_config.layer_norm_eps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pos_encoding_type(self) -> Literal["learned", "sinusoidal", "rope", "none"]:
        """Convenience property to access pos_encoding_type from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].pos_encoding_type
        return self.block_config.pos_encoding_type

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dropout_rate(self) -> float:
        """Convenience property to access dropout_rate from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].dropout_rate
        return self.block_config.dropout_rate

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rope_theta(self) -> float:
        """Convenience property to access rope_theta from first block_config."""
        if isinstance(self.block_config, list):
            return self.block_config[0].rope_theta
        return self.block_config.rope_theta

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mlp_dim(self) -> int:
        """Convenience property to access mlp_dim from first block_config."""
        if isinstance(self.block_config, list):
            assert self.block_config[0].mlp_dim is not None  # Set by validator
            return self.block_config[0].mlp_dim
        assert self.block_config.mlp_dim is not None  # Set by validator
        return self.block_config.mlp_dim
