"""Unit tests for TransformerConfig"""

import pytest

from zmaj_lm.config import TransformerConfig


class TestTransformerConfig:
    """Tests for TransformerConfig validation and defaults."""

    def test_hidden_dim_divisible_by_num_heads(self) -> None:
        """Test hidden_dim is divisible by num_heads."""
        with pytest.raises(ValueError, match="hidden_dim .* must be divisible by num_heads .*"):
            TransformerConfig(hidden_dim=250, num_heads=4)

    def test_valid_num_heads(self) -> None:
        """Verify divisibility check does not reject valid configs."""
        config = TransformerConfig(hidden_dim=256, num_heads=4)
        assert config.hidden_dim % config.num_heads == 0

    def test_mlp_dim_auto_set(self) -> None:
        """Test that mlp_dim is auto-set to 4 * hidden_dim if None."""
        config = TransformerConfig(hidden_dim=256, mlp_dim=None)
        assert config.mlp_dim == 1024

    def test_mlp_dim_explicit(self) -> None:
        """Test that mlp_dim remains as set if explicitly provided."""
        config = TransformerConfig(hidden_dim=256, mlp_dim=512)
        assert config.mlp_dim == 512

    def test_head_dim_computation(self) -> None:
        """Test that head_dim is computed correctly."""
        config = TransformerConfig(hidden_dim=256, num_heads=4)
        assert config.head_dim == 64

    def test_single_head(self) -> None:
        """Test configuration with a single attention head."""
        config = TransformerConfig(hidden_dim=128, num_heads=1)
        assert config.head_dim == 128

    def test_serialization_roundtrip(self) -> None:
        """Ensure config can be serialized and deserialized."""
        config = TransformerConfig(vocab_size=1000, hidden_dim=512)

        # Serialize to dict
        config_dict = config.model_dump()

        # Deserialize from dict
        restored = TransformerConfig.model_validate(config_dict)

        assert restored.vocab_size == 1000
        assert restored.hidden_dim == 512
        assert restored.mlp_dim == 2048  # Auto-computed

    def test_attention_dropout_rate_auto_set(self) -> None:
        """Test that attention_dropout_rate defaults to dropout_rate if None."""
        config = TransformerConfig(dropout_rate=0.15, attention_dropout_rate=None)
        assert config.attention_dropout_rate == 0.15

    def test_attention_dropout_rate_explicit(self) -> None:
        """Test that attention_dropout_rate can be set explicitly."""
        config = TransformerConfig(dropout_rate=0.1, attention_dropout_rate=0.2)
        assert config.attention_dropout_rate == 0.2
        assert config.dropout_rate == 0.1

    def test_residual_dropout_rate_auto_set(self) -> None:
        """Test that residual_dropout_rate defaults to dropout_rate if None."""
        config = TransformerConfig(dropout_rate=0.15, residual_dropout_rate=None)
        assert config.residual_dropout_rate == 0.15

    def test_residual_dropout_rate_explicit(self) -> None:
        """Test that residual_dropout_rate can be set explicitly."""
        config = TransformerConfig(dropout_rate=0.1, residual_dropout_rate=0.05)
        assert config.residual_dropout_rate == 0.05
        assert config.dropout_rate == 0.1

    def test_all_dropout_rates_independent(self) -> None:
        """Test that all three dropout rates can be set independently."""
        config = TransformerConfig(
            dropout_rate=0.1, attention_dropout_rate=0.2, residual_dropout_rate=0.05
        )
        assert config.dropout_rate == 0.1
        assert config.attention_dropout_rate == 0.2
        assert config.residual_dropout_rate == 0.05
