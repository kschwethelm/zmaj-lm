"""Unit tests for TokenizerTrainingConfig"""

import pytest

from zmaj_lm.config.tokenizer_config import DatasetConfig, TokenizerTrainingConfig


class TestTokenizerTrainingConfig:
    """Tests for TokenizerTrainingConfig validation and computed fields."""

    def test_vocab_size_minimum_validation(self) -> None:
        """Test that vocab_size must be at least 256."""
        # Should fail with vocab_size < 256
        with pytest.raises(ValueError, match="vocab_size .* must be at least 256"):
            TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"), vocab_size=255)

        with pytest.raises(ValueError, match="vocab_size .* must be at least 256"):
            TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"), vocab_size=100)

        # Should succeed with vocab_size >= 256
        config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"), vocab_size=256)
        assert config.vocab_size == 256

    def test_valid_vocab_sizes(self) -> None:
        """Test that valid vocab sizes are accepted."""
        valid_sizes = [256, 512, 1000, 8000, 32_000, 50_000]
        for size in valid_sizes:
            config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"), vocab_size=size)
            assert config.vocab_size == size

    def test_special_tokens_computed_field(self) -> None:
        """Test that special_tokens computed field returns correct list."""
        config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"))

        # Default special tokens (includes im_start and im_end)
        assert config.special_tokens == [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>",
            "<|im_start|>",
            "<|im_end|>",
        ]
        assert len(config.special_tokens) == 6

        # Custom special tokens
        config_custom = TokenizerTrainingConfig(
            datasets=DatasetConfig(path="dummy"),
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            im_start_token="<start>",
            im_end_token="<end>",
        )
        assert config_custom.special_tokens == [
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "[UNK]",
            "<start>",
            "<end>",
        ]

    def test_num_special_tokens_computed_field(self) -> None:
        """Test that num_special_tokens returns the correct count."""
        config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"))

        # Should match length of special_tokens list (6 tokens including im_start/im_end)
        assert config.num_special_tokens == 6
        assert config.num_special_tokens == len(config.special_tokens)

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"))

        # Vocabulary defaults
        assert config.vocab_size == 32_000
        assert config.min_frequency == 2
        assert config.max_training_samples is None

        # Special token defaults
        assert config.pad_token == "<PAD>"
        assert config.bos_token == "<BOS>"
        assert config.eos_token == "<EOS>"
        assert config.unk_token == "<UNK>"
        assert config.im_start_token == "<|im_start|>"
        assert config.im_end_token == "<|im_end|>"

        # Output defaults
        assert config.save_path == "tokenizers/custom_bpe"

    def test_serialization_roundtrip(self) -> None:
        """Ensure config can be serialized and deserialized."""
        config = TokenizerTrainingConfig(
            datasets=DatasetConfig(path="wikitext", name="wikitext-2-raw-v1"),
            vocab_size=16_000,
            min_frequency=3,
            max_training_samples=10_000,
            save_path="tokenizers/my_tokenizer",
        )

        # Serialize to dict
        config_dict = config.model_dump()

        # Deserialize from dict
        restored = TokenizerTrainingConfig.model_validate(config_dict)

        assert isinstance(restored.datasets, DatasetConfig)
        assert restored.datasets.path == "wikitext"
        assert restored.datasets.name == "wikitext-2-raw-v1"
        assert restored.vocab_size == 16_000
        assert restored.min_frequency == 3
        assert restored.max_training_samples == 10_000
        assert restored.save_path == "tokenizers/my_tokenizer"

        # Computed fields should work after deserialization
        assert restored.num_special_tokens == 6
        assert len(restored.special_tokens) == 6

    def test_custom_special_tokens(self) -> None:
        """Test that custom special tokens can be configured."""
        config = TokenizerTrainingConfig(
            datasets=DatasetConfig(path="dummy"),
            pad_token="[PAD]",
            bos_token="[START]",
            eos_token="[END]",
            unk_token="[UNKNOWN]",
            im_start_token="<start>",
            im_end_token="<end>",
        )

        assert config.pad_token == "[PAD]"
        assert config.bos_token == "[START]"
        assert config.eos_token == "[END]"
        assert config.unk_token == "[UNKNOWN]"
        assert config.im_start_token == "<start>"
        assert config.im_end_token == "<end>"

        # special_tokens should reflect all custom tokens including im tokens
        assert config.special_tokens == [
            "[PAD]",
            "[START]",
            "[END]",
            "[UNKNOWN]",
            "<start>",
            "<end>",
        ]

    def test_im_tokens_included_in_special_tokens_list(self) -> None:
        """Test that im_start and im_end tokens are included in special_tokens list."""
        config = TokenizerTrainingConfig(datasets=DatasetConfig(path="dummy"))

        # im_start and im_end should be in special_tokens for proper tokenizer training
        assert config.im_start_token == "<|im_start|>"
        assert config.im_end_token == "<|im_end|>"
        assert config.im_start_token in config.special_tokens
        assert config.im_end_token in config.special_tokens
        assert config.special_tokens[-2:] == ["<|im_start|>", "<|im_end|>"]
