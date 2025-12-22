import torch

from zmaj_lm.config.model_config import TransformerBlockConfig
from zmaj_lm.models.transformer_block import TransformerBlock
from zmaj_lm.utils.masks import create_causal_mask


class TestTransformerBlock:
    """Test suite for TransformerBlock module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that TransformerBlock preserves input shape."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16

        block = TransformerBlock(config=block_config).to(device)
        block.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = block(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_with_causal_mask(self, device: torch.device) -> None:
        """Test TransformerBlock with causal mask."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8

        block = TransformerBlock(config=block_config).to(device)
        block.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)
        mask = create_causal_mask(seq_len, device=device)

        with torch.no_grad():
            output = block(x, mask=mask)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode is consistent and training mode varies with dropout."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            residual_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        block = TransformerBlock(config=block_config).to(device)
        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        # Eval mode should give same output every time
        block.eval()
        with torch.no_grad():
            output1 = block(x)
            output2 = block(x)
        assert torch.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        block.train()
        torch.manual_seed(42)
        output3 = block(x)
        torch.manual_seed(43)
        output4 = block(x)
        assert not torch.allclose(output3, output4)

    def test_gradient_flow(self, device: torch.device) -> None:
        """Test that gradients flow through residual connections."""
        block_config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,  # Disable dropout for gradient test
        )

        batch, seq_len = 2, 4

        block = TransformerBlock(config=block_config).to(device)
        block.train()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device, requires_grad=True)

        # Forward pass
        output = block(x)

        # Simple loss function
        loss = (output**2).sum()

        # Backward pass
        loss.backward()

        # Check that all parameters have non-zero gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0.0), f"Zero gradient for {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient for {name}"
            assert not torch.any(torch.isinf(param.grad)), f"Inf gradient for {name}"

    def test_compilation(self, device: torch.device) -> None:
        """Test that TransformerBlock can be compiled."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8

        block = TransformerBlock(config=block_config).to(device)
        block.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        # Compile the module
        compiled_block = torch.compile(block)

        with torch.no_grad():
            output_compiled = compiled_block(x)
            output_regular = block(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)
