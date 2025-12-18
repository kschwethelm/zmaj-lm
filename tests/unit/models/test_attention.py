import torch

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.attention import MultiHeadAttention
from zmaj_lm.utils.masks import create_causal_mask


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that MHA produces correct output shape."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_with_mask(self, device: torch.device) -> None:
        """Test MHA with causal mask."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
        mask = create_causal_mask(seq_len, device=device)

        with torch.no_grad():
            output = mha(x, mask=mask)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode produces consistent outputs."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            attention_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Eval mode should give same output every time
        mha.eval()
        with torch.no_grad():
            output1 = mha(x)
            output2 = mha(x)
        assert torch.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        mha.train()
        torch.manual_seed(42)
        output3 = mha(x)
        torch.manual_seed(43)
        output4 = mha(x)
        assert not torch.allclose(output3, output4)

    def test_parameter_count(self, device: torch.device) -> None:
        """Test that parameter count matches expected for MHA."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=True,
        )

        mha = MultiHeadAttention(config=config).to(device)

        # Count parameters
        # Q, K, V projections: each has (hidden_dim, hidden_dim) + hidden_dim bias
        # Output projection: (hidden_dim, hidden_dim) + hidden_dim bias
        expected_params_per_proj = config.hidden_dim * config.hidden_dim + config.hidden_dim
        expected_total = 4 * expected_params_per_proj

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_parameter_count_no_bias(self, device: torch.device) -> None:
        """Test parameter count without bias."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=False,
        )

        mha = MultiHeadAttention(config=config).to(device)

        # Without bias: 4 projections * (hidden_dim * hidden_dim)
        expected_total = 4 * config.hidden_dim * config.hidden_dim

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_different_sequence_lengths(self, device: torch.device) -> None:
        """Test that MHA works with different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch = 2

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        # Test with different sequence lengths
        for seq_len in [1, 4, 8, 32]:
            x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
            with torch.no_grad():
                output = mha(x)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_single_head(self, device: torch.device) -> None:
        """Test edge case with single attention head."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=1,  # Single head
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_compilation(self, device: torch.device) -> None:
        """Test that MHA can be compiled with torch.compile."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Compile the module
        compiled_mha = torch.compile(mha)

        with torch.no_grad():
            output_compiled = compiled_mha(x)
            output_regular = mha(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)
