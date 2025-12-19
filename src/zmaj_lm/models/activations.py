import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    """Base class for activation functions.

    Attributes:
        chunk_size: Input dimension multiplier. For gated activations (GLU variants),
                    chunk_size=2 means the layer expects 2*dim input and outputs dim.
                    For standard activations, chunk_size=1.
    """

    chunk_size: int = 1


class GELU(Activation):
    """Gaussian Error Linear Unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class GELUTanh(Activation):
    """GELU with tanh approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class SiLU(Activation):
    """Sigmoid Linear Unit (also known as Swish) activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class ReLU(Activation):
    """Rectified Linear Unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class GeGLU(Activation):
    """Gated Linear Unit with GELU activation.

    Splits input in half: value and gate, returns value * GELU(gate).
    """

    chunk_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation.

        Args:
            x: Input tensor of shape (batch, seq_len, 2*dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)


class SwiGLU(Activation):
    """Gated Linear Unit with SiLU/Swish activation.

    Used in models like LLaMA, Mistral. Splits input in half: value and gate,
    returns value * SiLU(gate).
    """

    chunk_size = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor of shape (batch, seq_len, 2*dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


# Activation registry mapping names to activation classes
ACTIVATIONS: dict[str, type[Activation]] = {
    "gelu": GELU,
    "gelu_tanh": GELUTanh,
    "silu": SiLU,
    "relu": ReLU,
    "geglu": GeGLU,
    "swiglu": SwiGLU,
}


def get_activation(name: str) -> Activation:
    """Get an activation function instance by name.

    Args:
        name: Name of the activation function (e.g., 'gelu', 'swiglu')

    Returns:
        Instantiated activation module

    Raises:
        ValueError: If the activation name is not recognized
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: {name}. Available activations: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]()
