import torch

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.feedforward import FeedForward


class TestFeedForward:
    """Test suite for FeedForward (MLP) module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that FFN preserves input shape (batch, seq_len, hidden_dim)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16

        ffn = FeedForward(config=config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_parameter_count(self, device: torch.device) -> None:
        """Test that parameter count matches expected for FFN."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
            use_bias=True,
        )

        ffn = FeedForward(config=config).to(device)

        # First layer: (hidden_dim -> mlp_dim) + mlp_dim bias
        # Second layer: (mlp_dim -> hidden_dim) + hidden_dim bias
        expected_total = (
            config.hidden_dim * config.mlp_dim
            + config.mlp_dim
            + config.mlp_dim * config.hidden_dim
            + config.hidden_dim
        )

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_total

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode produces consistent outputs and training mode applies dropout."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        ffn = FeedForward(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Eval mode should give same output every time
        ffn.eval()
        with torch.no_grad():
            output1 = ffn(x)
            output2 = ffn(x)
        assert torch.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        ffn.train()
        torch.manual_seed(42)
        output3 = ffn(x)
        torch.manual_seed(43)
        output4 = ffn(x)
        assert not torch.allclose(output3, output4)

    def test_activation_nonlinearity(self, device: torch.device) -> None:
        """Test that FFN applies nonlinear activation (GELU)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
        )

        ffn = FeedForward(config=config).to(device)
        ffn.eval()

        # Use a simple input pattern
        x = torch.ones((1, 4, config.hidden_dim), device=device)

        with torch.no_grad():
            output = ffn(x)

            # If activation is working, output should not be a simple linear transformation
            # We verify this by checking that FFN(2*x) != 2*FFN(x) (non-linear behavior)
            x_scaled = 2.0 * x
            output_scaled_input = ffn(x_scaled)
            output_scaled = 2.0 * output

        # Due to GELU nonlinearity, these should NOT be equal
        assert not torch.allclose(output_scaled_input, output_scaled, rtol=0.01)

    def test_compilation(self, device: torch.device) -> None:
        """Test that FFN can be compiled."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8

        ffn = FeedForward(config=config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Compile the module
        compiled_ffn = torch.compile(ffn)

        with torch.no_grad():
            output_compiled = compiled_ffn(x)
            output_regular = ffn(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)
