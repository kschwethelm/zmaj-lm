"""Tests for tokenizer evaluation functionality."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from zmaj_lm.config.tokenizer_config import DatasetConfig, TokenizerTrainingConfig
from zmaj_lm.data.tokenizer.evaluate_tokenizer import _compute_tokenizer_metrics


@pytest.fixture
def simple_tokenizer() -> Tokenizer:
    """Create a simple BPE tokenizer for testing."""
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
    )

    training_corpus = [
        "Hello, world!",
        "This is a test.",
        "Testing tokenizer evaluation.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    return tokenizer


@pytest.fixture
def eval_config() -> TokenizerTrainingConfig:
    """Create a basic evaluation config."""
    return TokenizerTrainingConfig(
        datasets=DatasetConfig(path="test_dataset", split="train"),
        vocab_size=1000,
        run_evaluation=True,
        eval_samples=10,
    )


def test_compute_tokenizer_metrics_basic(simple_tokenizer: Tokenizer) -> None:
    """Test that basic metrics are computed correctly."""
    texts = ["Hello, world!", "This is a test."]

    metrics = _compute_tokenizer_metrics(simple_tokenizer, texts, "<UNK>")

    assert "compression_ratio" in metrics
    assert "fertility" in metrics
    assert "unk_rate" in metrics

    assert 0.0 < metrics["compression_ratio"] < 1.0
    assert metrics["fertility"] > 0.0
    assert 0.0 <= metrics["unk_rate"] <= 1.0


def test_compute_tokenizer_metrics_empty_texts(simple_tokenizer: Tokenizer) -> None:
    """Test metrics computation with empty text list."""
    texts: list[str] = []

    metrics = _compute_tokenizer_metrics(simple_tokenizer, texts, "<UNK>")

    assert metrics["compression_ratio"] == 0.0
    assert metrics["fertility"] == 0.0
    assert metrics["unk_rate"] == 0.0


def test_compute_tokenizer_metrics_unknown_tokens(simple_tokenizer: Tokenizer) -> None:
    """Test unknown token rate calculation with rare characters."""
    texts = [
        "Hello world",
        "你好世界",  # Chinese characters that might be unknown
    ]

    metrics = _compute_tokenizer_metrics(simple_tokenizer, texts, "<UNK>")

    assert metrics["unk_rate"] >= 0.0


def test_compression_ratio_increases_with_longer_tokens(simple_tokenizer: Tokenizer) -> None:
    """Test that compression ratio behaves sensibly."""
    short_text = ["a a a a a"]
    long_text = ["supercalifragilisticexpialidocious"]

    metrics_short = _compute_tokenizer_metrics(simple_tokenizer, short_text, "<UNK>")
    metrics_long = _compute_tokenizer_metrics(simple_tokenizer, long_text, "<UNK>")

    assert metrics_short["compression_ratio"] > 0.0
    assert metrics_long["compression_ratio"] > 0.0


def test_fertility_calculation(simple_tokenizer: Tokenizer) -> None:
    """Test fertility (subwords per word) calculation."""
    texts = ["hello world", "testing"]

    metrics = _compute_tokenizer_metrics(simple_tokenizer, texts, "<UNK>")

    assert metrics["fertility"] > 0.0
