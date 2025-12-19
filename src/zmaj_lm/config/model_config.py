from typing import Literal

from pydantic import BaseModel, computed_field, model_validator


class TransformerConfig(BaseModel):
    """Configuration for a GPT-style decoder-only Transformer model.

    This config defines the architecture hyperparameters for a causal
    language model with multi-head self-attention and feedforward layers.
    """

    vocab_size: int = 50257  # GPT-2 tokenizer size
    max_seq_len: int = 1024

    hidden_dim: int = 256
    num_layers: int = 4  # number of transformer blocks
    num_heads: int = 4  # number of attention heads
    mlp_dim: int | None = None  # typically 4 * hidden_dim, None = auto

    dropout_rate: float = 0.1  # dropout rate (0.0 to 0.1 typically)
    layer_norm_eps: float = 1e-5
    use_bias: bool = True  # whether to use bias in linear layers
    attention_dropout_rate: float | None = None  # if None, use dropout_rate
    residual_dropout_rate: float | None = None  # if None, use dropout_rate

    pos_encoding_type: Literal["learned", "sinusoidal"] = "learned"  # TODO: RoPE, NoPE, RNoPE
    activation: Literal["gelu", "gelu_tanh", "silu", "relu", "geglu", "swiglu"] = "gelu"

    @model_validator(mode="after")
    def validate_num_heads(self) -> "TransformerConfig":
        # Ensure hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self

    @model_validator(mode="after")
    def validate_mlp_dim(self) -> "TransformerConfig":
        # Ensure mlp_dim is set correctly
        if self.mlp_dim is None:
            object.__setattr__(self, "mlp_dim", 4 * self.hidden_dim)
        return self

    @model_validator(mode="after")
    def validate_attention_dropout_rate(self) -> "TransformerConfig":
        # Use dropout_rate if attention_dropout_rate is not specified
        if self.attention_dropout_rate is None:
            object.__setattr__(self, "attention_dropout_rate", self.dropout_rate)
        return self

    @model_validator(mode="after")
    def validate_residual_dropout_rate(self) -> "TransformerConfig":
        # Use dropout_rate if residual_dropout_rate is not specified
        if self.residual_dropout_rate is None:
            object.__setattr__(self, "residual_dropout_rate", self.dropout_rate)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads
