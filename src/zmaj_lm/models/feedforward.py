import torch
import torch.nn as nn

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.activations import get_activation


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the feedforward network.

        Args:
            config: Transformer configuration
        """
        super().__init__()
        self.config = config
        self.activation = get_activation(config.activation)

        # For gated activations (GLU variants), dense_1 projects to chunk_size*mlp_dim
        # For standard activations (chunk_size=1), dense_1 projects to mlp_dim
        dense_1_output_dim = config.mlp_dim * self.activation.chunk_size
        self.dense_1 = nn.Linear(config.hidden_dim, dense_1_output_dim, bias=config.use_bias)

        # dense_2 always takes mlp_dim as input (after activation/chunking)
        self.dense_2 = nn.Linear(config.mlp_dim, config.hidden_dim, bias=config.use_bias)

        self.dropout = nn.Dropout(p=config.dropout_rate)

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
