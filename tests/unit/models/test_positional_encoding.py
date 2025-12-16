import pytest
import torch

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.positional_encoding import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    get_positional_encoding_module,
)


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that sinusoidal encoding preserves input shape."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="sinusoidal",
        )

        batch, seq_len = 2, 128

        pos_enc = SinusoidalPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        output = pos_enc(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_deterministic_output(self, device: torch.device) -> None:
        """Test that sinusoidal encodings are deterministic (no randomness)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="sinusoidal",
        )

        batch, seq_len = 2, 64

        pos_enc = SinusoidalPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Should give identical output every time (no stochasticity)
        output1 = pos_enc(x)
        output2 = pos_enc(x)

        assert torch.allclose(output1, output2, rtol=1e-7, atol=1e-7)

    def test_variable_sequence_lengths(self, device: torch.device) -> None:
        """Test that encoding works for different sequence lengths up to max_seq_len."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="sinusoidal",
        )

        batch = 2

        pos_enc = SinusoidalPositionalEncoding(config=config).to(device)

        # Test various sequence lengths
        for seq_len in [16, 32, 128, 256, 512]:
            x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
            output = pos_enc(x)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_position_encoding_properties(self, device: torch.device) -> None:
        """Test mathematical properties of sinusoidal encoding."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pos_encoding_type="sinusoidal",
        )

        batch, seq_len = 1, 128

        pos_enc = SinusoidalPositionalEncoding(config=config).to(device)
        # Use zeros to isolate the positional encodings
        x = torch.zeros((batch, seq_len, config.hidden_dim), device=device)

        output = pos_enc(x)

        # Extract just the positional encodings (since x was zeros)
        encodings = output[0]  # [seq_len, hidden_dim]

        # Check that different positions have different encodings
        pos_0 = encodings[0]
        pos_1 = encodings[1]
        assert not torch.allclose(pos_0, pos_1)

        # Check that encodings are bounded (sin/cos outputs)
        assert torch.all(torch.abs(encodings) <= 2.0)  # Should be roughly in [-1, 1]

    def test_exceeds_max_seq_len_raises(self, device: torch.device) -> None:
        """Test that sequence length exceeding max_seq_len raises an error."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pos_encoding_type="sinusoidal",
        )

        batch, seq_len = 2, 256  # Exceeds max_seq_len=128

        pos_enc = SinusoidalPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with pytest.raises(AssertionError):
            pos_enc(x)


class TestLearnedPositionalEncoding:
    """Test suite for LearnedPositionalEncoding module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that learned encoding preserves input shape."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="learned",
        )

        batch, seq_len = 2, 128

        pos_enc = LearnedPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        output = pos_enc(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_parameter_count(self, device: torch.device) -> None:
        """Test that learned embeddings have correct number of parameters."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="learned",
        )

        pos_enc = LearnedPositionalEncoding(config=config).to(device)

        # Expected: max_seq_len positions × hidden_dim features
        expected_params = config.max_seq_len * config.hidden_dim

        total_params = sum(p.numel() for p in pos_enc.parameters())
        assert total_params == expected_params

    def test_different_positions_different_embeddings(self, device: torch.device) -> None:
        """Test that different positions have different learned embeddings."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pos_encoding_type="learned",
        )

        batch, seq_len = 1, 128

        pos_enc = LearnedPositionalEncoding(config=config).to(device)
        # Use zeros to isolate positional embeddings
        x = torch.zeros((batch, seq_len, config.hidden_dim), device=device)

        output = pos_enc(x)

        # Extract positional embeddings (since x was zeros)
        embeddings = output[0]  # [seq_len, hidden_dim]

        # Different positions should have different embeddings
        pos_0 = embeddings[0]
        pos_1 = embeddings[1]
        pos_127 = embeddings[127]

        assert not torch.allclose(pos_0, pos_1)
        assert not torch.allclose(pos_0, pos_127)

    def test_variable_sequence_lengths(self, device: torch.device) -> None:
        """Test that encoding works for different sequence lengths up to max_seq_len."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="learned",
        )

        batch = 2

        pos_enc = LearnedPositionalEncoding(config=config).to(device)

        # Test various sequence lengths
        for seq_len in [16, 32, 128, 256, 512]:
            x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
            output = pos_enc(x)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_exceeds_max_seq_len_raises(self, device: torch.device) -> None:
        """Test that sequence length exceeding max_seq_len raises an error."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pos_encoding_type="learned",
        )

        batch, seq_len = 2, 256  # Exceeds max_seq_len=128

        pos_enc = LearnedPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with pytest.raises(AssertionError):
            pos_enc(x)


class TestGetPositionalEncodingModule:
    """Test suite for get_positional_encoding_module factory function."""

    def test_returns_sinusoidal_when_specified(self) -> None:
        """Test that factory returns SinusoidalPositionalEncoding when specified."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="sinusoidal",
        )

        module = get_positional_encoding_module(config)
        assert isinstance(module, SinusoidalPositionalEncoding)

    def test_returns_learned_when_specified(self) -> None:
        """Test that factory returns LearnedPositionalEncoding when specified."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="learned",
        )

        module = get_positional_encoding_module(config)
        assert isinstance(module, LearnedPositionalEncoding)

    def test_returns_learned_by_default(self) -> None:
        """Test that factory returns LearnedPositionalEncoding by default."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
        )  # pos_encoding_type defaults to "learned"

        module = get_positional_encoding_module(config)
        assert isinstance(module, LearnedPositionalEncoding)
