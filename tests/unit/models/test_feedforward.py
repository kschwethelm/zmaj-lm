import pytest
import torch

from zmaj_lm.config.model_config import TransformerBlockConfig
from zmaj_lm.models.feedforward import FeedForward


class TestFeedForward:
    """Test suite for FeedForward (MLP) module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that FFN preserves input shape (batch, seq_len, hidden_dim)."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_parameter_count(self, device: torch.device) -> None:
        """Test that parameter count matches expected for FFN."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            mlp_dim=1024,
            use_bias=True,
        )

        ffn = FeedForward(config=block_config).to(device)

        # First layer: (hidden_dim -> mlp_dim) + mlp_dim bias
        # Second layer: (mlp_dim -> hidden_dim) + hidden_dim bias
        expected_total = (
            block_config.hidden_dim * block_config.mlp_dim
            + block_config.mlp_dim
            + block_config.mlp_dim * block_config.hidden_dim
            + block_config.hidden_dim
        )

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_total

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode produces consistent outputs and training mode applies dropout."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        ffn = FeedForward(config=block_config).to(device)
        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

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
        block_config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
        )

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        # Use a simple input pattern
        x = torch.ones((1, 4, block_config.hidden_dim), device=device)

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
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        # Compile the module
        compiled_ffn = torch.compile(ffn)

        with torch.no_grad():
            output_compiled = compiled_ffn(x)
            output_regular = ffn(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "relu"])
    def test_activation_functions(self, activation: str, device: torch.device) -> None:
        """Test that different activation functions work correctly."""
        block_config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
            activation=activation,
        )

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

        # Verify activation is applied (nonlinearity check)
        with torch.no_grad():
            x_scaled = 2.0 * x
            output_scaled_input = ffn(x_scaled)
            output_scaled = 2.0 * output

        # All activations should be nonlinear
        assert not torch.allclose(output_scaled_input, output_scaled, rtol=0.01)

    def test_gelu_variants_difference(self, device: torch.device) -> None:
        """Test that GELU and GELU tanh produce different outputs."""
        block_config_gelu = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
            activation="gelu",
        )

        block_config_gelu_tanh = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
            activation="gelu_tanh",
        )

        ffn_gelu = FeedForward(config=block_config_gelu).to(device)
        ffn_gelu_tanh = FeedForward(config=block_config_gelu_tanh).to(device)

        # Copy weights to ensure difference is only from activation
        ffn_gelu_tanh.load_state_dict(ffn_gelu.state_dict(), strict=False)

        ffn_gelu.eval()
        ffn_gelu_tanh.eval()

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, block_config_gelu.hidden_dim, device=device)

        with torch.no_grad():
            output_gelu = ffn_gelu(x)
            output_gelu_tanh = ffn_gelu_tanh(x)

        # Outputs should be different but close (both are GELU approximations)
        assert not torch.allclose(output_gelu, output_gelu_tanh, rtol=1e-6, atol=1e-8)
        assert torch.allclose(output_gelu, output_gelu_tanh, rtol=5e-2, atol=1e-2)

    def test_gated_activation_output_shape(self, device: torch.device) -> None:
        """Test that gated FFN preserves input shape."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            mlp_dim=1024,
            activation="swiglu",
        )

        batch, seq_len = 2, 16

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_gated_activation_parameter_count(self, device: torch.device) -> None:
        """Test that gated FFN parameter count with chunking."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            mlp_dim=1024,
            use_bias=True,
            activation="swiglu",
        )

        ffn = FeedForward(config=block_config).to(device)

        # With gated activation (chunk-based):
        # dense_1: (hidden_dim -> 2*mlp_dim) + 2*mlp_dim bias
        # dense_2: (mlp_dim -> hidden_dim) + hidden_dim bias (input is chunked to mlp_dim)
        expected_total = (
            block_config.hidden_dim * (2 * block_config.mlp_dim)
            + (2 * block_config.mlp_dim)
            + block_config.mlp_dim * block_config.hidden_dim
            + block_config.hidden_dim
        )

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_total

    def test_gated_vs_non_gated_outputs_differ(self, device: torch.device) -> None:
        """Test that gated and non-gated FFN produce different outputs."""
        block_config_standard = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.0,
            activation="gelu",
        )

        block_config_gated = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.0,
            activation="geglu",
        )

        ffn_standard = FeedForward(config=block_config_standard).to(device)
        ffn_gated = FeedForward(config=block_config_gated).to(device)

        ffn_standard.eval()
        ffn_gated.eval()

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, block_config_standard.hidden_dim, device=device)

        with torch.no_grad():
            output_standard = ffn_standard(x)
            output_gated = ffn_gated(x)

        # Outputs should differ significantly due to gating mechanism
        assert not torch.allclose(output_standard, output_gated, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize("activation", ["geglu", "swiglu"])
    def test_gated_activation_variants(self, activation: str, device: torch.device) -> None:
        """Test that gated activation works with different activation functions (GeGLU, SwiGLU)."""
        block_config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
            activation=activation,
        )

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = ffn(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

        # Verify gating is applied (nonlinearity check)
        with torch.no_grad():
            x_scaled = 2.0 * x
            output_scaled_input = ffn(x_scaled)
            output_scaled = 2.0 * output

        # Gated activations should be nonlinear
        assert not torch.allclose(output_scaled_input, output_scaled, rtol=0.01)

    def test_swiglu_compilation(self, device: torch.device) -> None:
        """Test that SwiGLU can be compiled."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            mlp_dim=512,
            activation="swiglu",
        )

        batch, seq_len = 2, 8

        ffn = FeedForward(config=block_config).to(device)
        ffn.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        # Compile the module
        compiled_ffn = torch.compile(ffn)

        with torch.no_grad():
            output_compiled = compiled_ffn(x)
            output_regular = ffn(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)
