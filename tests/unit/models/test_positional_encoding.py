import pytest
import torch

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.positional_encoding import (
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    RotaryPositionalEncoding,
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

    def test_returns_rope_when_specified(self) -> None:
        """Test that factory returns RotaryPositionalEncoding when specified."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        module = get_positional_encoding_module(config)
        assert isinstance(module, RotaryPositionalEncoding)

    def test_returns_none_when_specified(self) -> None:
        """Test that factory returns NoPositionalEncoding when specified."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="none",
        )

        module = get_positional_encoding_module(config)
        assert isinstance(module, NoPositionalEncoding)


class TestNoPositionalEncoding:
    """Test suite for NoPositionalEncoding module."""

    def test_output_unchanged(self, device: torch.device) -> None:
        """Test that NoPE returns input unchanged."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="none",
        )

        batch, seq_len = 2, 128

        pos_enc = NoPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        output = pos_enc(x)

        assert output.shape == x.shape
        assert torch.allclose(output, x)

    def test_no_parameters(self, device: torch.device) -> None:
        """Test that NoPE has no learnable parameters."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="none",
        )

        pos_enc = NoPositionalEncoding(config=config).to(device)
        total_params = sum(p.numel() for p in pos_enc.parameters())
        assert total_params == 0


class TestRotaryPositionalEncoding:
    """Test suite for RotaryPositionalEncoding (RoPE) module."""

    def test_initialization(self, device: torch.device) -> None:
        """Test that RoPE initializes correctly with precomputed frequencies."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        rope = RotaryPositionalEncoding(config=config).to(device)

        # Check that freqs_complex has correct shape: [max_seq_len, head_dim/2]
        assert rope.freqs_complex.shape == (config.max_seq_len, config.head_dim // 2)
        assert rope.freqs_complex.dtype == torch.complex64

    def test_forward_is_noop(self, device: torch.device) -> None:
        """Test that forward pass is a no-op (rotation happens in attention)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, seq_len = 2, 64
        rope = RotaryPositionalEncoding(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        output = rope(x)

        # Forward should just pass through
        assert torch.allclose(output, x)

    def test_apply_rotary_pos_emb_shape(self, device: torch.device) -> None:
        """Test that apply_rotary_pos_emb preserves shapes."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, seq_len, head_dim = 2, 8, 64, 16

        rope = RotaryPositionalEncoding(config=config).to(device)
        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

        q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.any(torch.isnan(q_rot))
        assert not torch.any(torch.isnan(k_rot))
        assert not torch.any(torch.isinf(q_rot))
        assert not torch.any(torch.isinf(k_rot))

    def test_rotation_changes_vectors(self, device: torch.device) -> None:
        """Test that RoPE actually modifies the query and key vectors."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, seq_len, head_dim = 2, 8, 64, 16

        rope = RotaryPositionalEncoding(config=config).to(device)
        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

        q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)

        # Rotated vectors should be different from original
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)

    def test_relative_position_encoding(self, device: torch.device) -> None:
        """Test that RoPE encodes relative positions (key property).

        The inner product between rotated query at position m and rotated key
        at position n should depend only on (m-n), not on absolute positions.
        """
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, seq_len, head_dim = 1, 1, 10, 16

        rope = RotaryPositionalEncoding(config=config).to(device)

        # Create a constant vector at all positions
        const_vec = torch.randn(head_dim, device=device)
        q = const_vec.view(1, 1, 1, head_dim).expand(batch, n_heads, seq_len, head_dim).clone()
        k = const_vec.view(1, 1, 1, head_dim).expand(batch, n_heads, seq_len, head_dim).clone()

        q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)

        # Compute inner products for different position pairs with same relative distance
        # pos (0, 1) and pos (5, 6) both have relative distance 1
        inner_prod_01 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
        inner_prod_56 = (q_rot[0, 0, 5] * k_rot[0, 0, 6]).sum()

        # Should be approximately equal (relative position encoding property)
        assert torch.allclose(inner_prod_01, inner_prod_56, rtol=1e-4, atol=1e-5)

        # pos (0, 2) and pos (5, 7) both have relative distance 2
        inner_prod_02 = (q_rot[0, 0, 0] * k_rot[0, 0, 2]).sum()
        inner_prod_57 = (q_rot[0, 0, 5] * k_rot[0, 0, 7]).sum()

        assert torch.allclose(inner_prod_02, inner_prod_57, rtol=1e-4, atol=1e-5)

    def test_different_rope_theta(self, device: torch.device) -> None:
        """Test that different rope_theta values produce different rotations."""
        config1 = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
            rope_theta=10000.0,
        )

        config2 = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
            rope_theta=100000.0,  # Different theta
        )

        rope1 = RotaryPositionalEncoding(config=config1).to(device)
        rope2 = RotaryPositionalEncoding(config=config2).to(device)

        # Frequency buffers should be different
        assert not torch.allclose(rope1.freqs_complex, rope2.freqs_complex)

        # Rotations should be different
        batch, n_heads, seq_len, head_dim = 1, 1, 64, 16
        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

        q_rot1, k_rot1 = rope1.apply_rotary_pos_emb(q, k)
        q_rot2, k_rot2 = rope2.apply_rotary_pos_emb(q, k)

        assert not torch.allclose(q_rot1, q_rot2)
        assert not torch.allclose(k_rot1, k_rot2)

    def test_variable_sequence_lengths(self, device: torch.device) -> None:
        """Test that RoPE works for different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, head_dim = 2, 8, 16

        rope = RotaryPositionalEncoding(config=config).to(device)

        # Test various sequence lengths
        for seq_len in [16, 32, 64, 128, 256, 512]:
            q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

            q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)

            assert q_rot.shape == (batch, n_heads, seq_len, head_dim)
            assert k_rot.shape == (batch, n_heads, seq_len, head_dim)

    def test_exceeds_max_seq_len_raises(self, device: torch.device) -> None:
        """Test that sequence length exceeding max_seq_len raises an error."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=128,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, seq_len, head_dim = 2, 8, 256, 16  # seq_len > max_seq_len

        rope = RotaryPositionalEncoding(config=config).to(device)
        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

        with pytest.raises(AssertionError):
            rope.apply_rotary_pos_emb(q, k)

    def test_no_learnable_parameters(self, device: torch.device) -> None:
        """Test that RoPE has no learnable parameters (only buffers)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        rope = RotaryPositionalEncoding(config=config).to(device)
        total_params = sum(p.numel() for p in rope.parameters())
        assert total_params == 0

    def test_rotation_preserves_norm(self, device: torch.device) -> None:
        """Test that rotation approximately preserves vector norms."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="rope",
        )

        batch, n_heads, seq_len, head_dim = 2, 8, 64, 16

        rope = RotaryPositionalEncoding(config=config).to(device)
        q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

        q_rot, k_rot = rope.apply_rotary_pos_emb(q, k)

        # Compute norms
        q_norm = torch.norm(q, dim=-1)
        k_norm = torch.norm(k, dim=-1)
        q_rot_norm = torch.norm(q_rot, dim=-1)
        k_rot_norm = torch.norm(k_rot, dim=-1)

        # Rotation should approximately preserve norms
        assert torch.allclose(q_norm, q_rot_norm, rtol=1e-4, atol=1e-5)
        assert torch.allclose(k_norm, k_rot_norm, rtol=1e-4, atol=1e-5)
