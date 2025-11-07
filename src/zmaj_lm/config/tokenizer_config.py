from pydantic import BaseModel, computed_field, model_validator


class DatasetConfig(BaseModel):
    """Configuration for loading a single dataset for tokenizer training.

    Examples:
        Simple usage:
        >>> DatasetConfig(path="wikitext")

        With dataset config:
        >>> DatasetConfig(path="code_search_net", name="python", text_column="whole_func_string")

        With specific split and weight:
        >>> DatasetConfig(path="wikipedia", name="20231101.en", split="train", text_column="text", weight=0.7)
    """

    path: str  # HuggingFace dataset path
    name: str | None = None  # Dataset configuration name (e.g., "python" for code_search_net)
    split: str = "train"  # Dataset split to use for training
    eval_split: str | None = None  # Dataset split to use for evaluation (defaults to split if None)
    text_column: str = "text"  # Column containing text data
    trust_remote_code: bool = False  # Whether to trust remote code in dataset loading
    weight: float = 1.0  # Sampling weight for dataset mixing (higher = more samples)


class TokenizerTrainingConfig(BaseModel):
    """Configuration for training a BPE tokenizer from scratch using HuggingFace tokenizers.

    This config is used only for the tokenizer training process. Once trained,
    the tokenizer is saved and can be loaded by referencing its path.
    """

    # Training corpus
    datasets: DatasetConfig | list[DatasetConfig]
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

    # Evaluation
    run_evaluation: bool = True  # Whether to run evaluation after training
    eval_samples: int = 100  # Number of samples from validation set to evaluate on
    eval_test_strings: list[str] | None = None  # Custom test strings for qualitative evaluation

    # Reproducibility
    seed: int = 42  # Random seed for dataset shuffling and sampling

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
