"""Unit tests for DatasetConfig"""

import pytest

from zmaj_lm.config.dataset_config import DatasetConfig


class TestDatasetConfig:
    """Tests for DatasetConfig validation and defaults."""

    def test_required_fields(self) -> None:
        """Test that required fields must be provided."""
        # Should fail without required fields
        with pytest.raises(ValueError):
            DatasetConfig()

        # Should succeed with required fields
        config = DatasetConfig(dataset_name="roneneldan/TinyStories", tokenizer_path="gpt2")
        assert config.dataset_name == "roneneldan/TinyStories"
        assert config.tokenizer_path == "gpt2"

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = DatasetConfig(dataset_name="roneneldan/TinyStories", tokenizer_path="gpt2")

        # Sequence and batch configuration
        assert config.seq_len == 1024
        assert config.batch_size == 32

        # Data splitting
        assert config.split_ratio == 0.95

        # Sequence packing configuration
        assert config.use_packing is True
        assert config.prevent_cross_doc_attention is False

        # Training configuration
        assert config.shuffle is True
        assert config.seed == 42

        # Performance options
        assert config.cache_dir is None
        assert config.num_proc == 8

    def test_serialization_roundtrip(self) -> None:
        """Ensure config can be serialized and deserialized."""
        config = DatasetConfig(
            dataset_name="wikitext",
            tokenizer_path="tokenizers/my_bpe",
            seq_len=512,
            batch_size=16,
            split_ratio=0.9,
            use_packing=False,
            prevent_cross_doc_attention=True,
            shuffle=False,
            seed=123,
            cache_dir="/tmp/datasets",
            num_proc=4,
        )

        # Serialize to dict
        config_dict = config.model_dump()

        # Deserialize from dict
        restored = DatasetConfig.model_validate(config_dict)

        assert restored.dataset_name == "wikitext"
        assert restored.tokenizer_path == "tokenizers/my_bpe"
        assert restored.seq_len == 512
        assert restored.batch_size == 16
        assert restored.split_ratio == 0.9
        assert restored.use_packing is False
        assert restored.prevent_cross_doc_attention is True
        assert restored.shuffle is False
        assert restored.seed == 123
        assert restored.cache_dir == "/tmp/datasets"
        assert restored.num_proc == 4
