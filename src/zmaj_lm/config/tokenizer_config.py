from pydantic import BaseModel, computed_field, model_validator


class TokenizerTrainingConfig(BaseModel):
    """Configuration for training a BPE tokenizer from scratch using HuggingFace tokenizers.

    This config is used only for the tokenizer training process. Once trained,
    the tokenizer is saved and can be loaded by referencing its path.
    """

    # Training corpus
    training_corpus_path: str  # Path to training text file or directory
    max_training_samples: int | None = None  # Limit samples for faster training

    # Vocabulary
    vocab_size: int = 32_000
    min_frequency: int = 2  # Minimum frequency for BPE merges

    # Special tokens
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    im_start_token: str = "<|im_start|>"
    im_end_token: str = "<|im_end|>"

    # Output
    save_path: str = "tokenizers/custom_bpe"  # Where to save trained tokenizer

    @model_validator(mode="after")
    def validate_vocab_size(self) -> "TokenizerTrainingConfig":
        """Ensure vocab size is reasonable."""
        if self.vocab_size < 256:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be at least 256")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def special_tokens(self) -> list[str]:
        """List of all special tokens."""
        return [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.im_start_token,
            self.im_end_token,
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_special_tokens(self) -> int:
        """Number of special tokens."""
        return len(self.special_tokens)
