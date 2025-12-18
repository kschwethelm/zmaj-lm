from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from zmaj_lm.config.model_config import TransformerConfig


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the feedforward network.

        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_dim, config.mlp_dim, bias=config.use_bias)
        self.dense_2 = nn.Linear(config.mlp_dim, config.hidden_dim, bias=config.use_bias)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function based on the config.

        Args:
            activation: Name of the activation function

        Returns:
            Activation function
        """
        if activation == "gelu":
            return F.gelu
        elif activation == "gelu_tanh":
            return lambda x: F.gelu(x, approximate="tanh")
        elif activation == "silu":
            return F.silu
        elif activation == "relu":
            return F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feedforward network.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
