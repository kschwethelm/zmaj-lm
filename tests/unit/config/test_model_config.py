"""Unit tests for TransformerConfig and TransformerBlockConfig"""

import pytest

from zmaj_lm.config import TransformerBlockConfig, TransformerConfig


class TestTransformerBlockConfig:
    """Tests for TransformerBlockConfig validation and defaults."""

    def test_hidden_dim_divisible_by_num_heads(self) -> None:
        """Test hidden_dim is divisible by num_heads."""
        with pytest.raises(ValueError, match="hidden_dim .* must be divisible by num_heads .*"):
            TransformerBlockConfig(hidden_dim=250, num_heads=4)

    def test_valid_num_heads(self) -> None:
        """Verify divisibility check does not reject valid configs."""
        config = TransformerBlockConfig(hidden_dim=256, num_heads=4)
        assert config.hidden_dim % config.num_heads == 0

    def test_mlp_dim_auto_set(self) -> None:
        """Test that mlp_dim is auto-set to 4 * hidden_dim if None."""
        config = TransformerBlockConfig(hidden_dim=256, mlp_dim=None)
        assert config.mlp_dim == 1024

    def test_mlp_dim_explicit(self) -> None:
        """Test that mlp_dim remains as set if explicitly provided."""
        config = TransformerBlockConfig(hidden_dim=256, mlp_dim=512)
        assert config.mlp_dim == 512

    def test_head_dim_computation(self) -> None:
        """Test that head_dim is computed correctly."""
        config = TransformerBlockConfig(hidden_dim=256, num_heads=4)
        assert config.head_dim == 64

    def test_single_head(self) -> None:
        """Test configuration with a single attention head."""
        config = TransformerBlockConfig(hidden_dim=128, num_heads=1)
        assert config.head_dim == 128

    def test_attention_dropout_rate_auto_set(self) -> None:
        """Test that attention_dropout_rate defaults to dropout_rate if None."""
        config = TransformerBlockConfig(dropout_rate=0.15, attention_dropout_rate=None)
        assert config.attention_dropout_rate == 0.15

    def test_attention_dropout_rate_explicit(self) -> None:
        """Test that attention_dropout_rate can be set explicitly."""
        config = TransformerBlockConfig(dropout_rate=0.1, attention_dropout_rate=0.2)
        assert config.attention_dropout_rate == 0.2
        assert config.dropout_rate == 0.1

    def test_residual_dropout_rate_auto_set(self) -> None:
        """Test that residual_dropout_rate defaults to dropout_rate if None."""
        config = TransformerBlockConfig(dropout_rate=0.15, residual_dropout_rate=None)
        assert config.residual_dropout_rate == 0.15

    def test_residual_dropout_rate_explicit(self) -> None:
        """Test that residual_dropout_rate can be set explicitly."""
        config = TransformerBlockConfig(dropout_rate=0.1, residual_dropout_rate=0.05)
        assert config.residual_dropout_rate == 0.05
        assert config.dropout_rate == 0.1

    def test_all_dropout_rates_independent(self) -> None:
        """Test that all three dropout rates can be set independently."""
        config = TransformerBlockConfig(
            dropout_rate=0.1, attention_dropout_rate=0.2, residual_dropout_rate=0.05
        )
        assert config.dropout_rate == 0.1
        assert config.attention_dropout_rate == 0.2
        assert config.residual_dropout_rate == 0.05

    def test_pos_encoding_type_default(self) -> None:
        """Test that pos_encoding_type defaults to 'learned'."""
        config = TransformerBlockConfig()
        assert config.pos_encoding_type == "learned"

    def test_pos_encoding_literal(self) -> None:
        """Test that pos_encoding_type accepts only valid literals."""
        options = ["learned", "sinusoidal", "rope", "none"]
        for option in options:
            config = TransformerBlockConfig(pos_encoding_type=option)
            assert config.pos_encoding_type == option

        with pytest.raises(ValueError):
            TransformerBlockConfig(pos_encoding_type="invalid_type")

    def test_window_size_none_default(self) -> None:
        """Test that window_size defaults to None (full attention)."""
        config = TransformerBlockConfig()
        assert config.window_size is None

    def test_window_size_valid(self) -> None:
        """Test that window_size accepts valid positive integers."""
        config = TransformerBlockConfig(window_size=512)
        assert config.window_size == 512

    def test_window_size_zero_raises_error(self) -> None:
        """Test that window_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            TransformerBlockConfig(window_size=0)

    def test_window_size_negative_raises_error(self) -> None:
        """Test that negative window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be positive"):
            TransformerBlockConfig(window_size=-10)


class TestTransformerConfig:
    """Tests for TransformerConfig with block_config."""

    def test_default_config(self) -> None:
        """Test that TransformerConfig creates a default block_config."""
        config = TransformerConfig()
        assert config.block_config is not None
        assert isinstance(config.block_config, TransformerBlockConfig)

    def test_nested_block_config(self) -> None:
        """Test creating TransformerConfig with explicit block_config."""
        block_config = TransformerBlockConfig(hidden_dim=512, num_heads=8)
        config = TransformerConfig(vocab_size=1000, block_config=block_config)
        assert config.vocab_size == 1000
        assert config.hidden_dim == 512
        assert config.head_dim == 64

    def test_serialization_roundtrip(self) -> None:
        """Ensure config can be serialized and deserialized."""
        block_config = TransformerBlockConfig(hidden_dim=512)
        config = TransformerConfig(vocab_size=1000, block_config=block_config)

        # Serialize to dict
        config_dict = config.model_dump()

        # Deserialize from dict
        restored = TransformerConfig.model_validate(config_dict)

        assert restored.vocab_size == 1000
        assert restored.hidden_dim == 512
        assert restored.block_config.mlp_dim == 2048  # Auto-computed

    def test_get_block_config_single(self) -> None:
        """Test get_block_config with single block_config."""
        block_config = TransformerBlockConfig(hidden_dim=256)
        config = TransformerConfig(num_layers=4, block_config=block_config)

        for i in range(4):
            bc = config.get_block_config(i)
            assert bc.hidden_dim == 256
            assert bc is block_config

    def test_heterogeneous_blocks(self) -> None:
        """Test TransformerConfig with list of different block configs."""
        blocks = [
            TransformerBlockConfig(hidden_dim=256, pos_encoding_type="rope"),
            TransformerBlockConfig(hidden_dim=256, pos_encoding_type="learned"),
        ]
        config = TransformerConfig(num_layers=2, block_config=blocks)

        assert config.get_block_config(0).pos_encoding_type == "rope"
        assert config.get_block_config(1).pos_encoding_type == "learned"

    def test_block_config_list_length_validation(self) -> None:
        """Test that block_config list length must match num_layers."""
        blocks = [TransformerBlockConfig(hidden_dim=256)]
        with pytest.raises(ValueError, match="block_config list length"):
            TransformerConfig(num_layers=2, block_config=blocks)
