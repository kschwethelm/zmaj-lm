"""Tests for tokenizer training functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

from datasets import Dataset
from tokenizers import Tokenizer

from zmaj_lm.config.tokenizer_config import DatasetConfig, TokenizerTrainingConfig
from zmaj_lm.data.tokenizer.train_tokenizer import (
    _load_datasets,
    get_training_corpus,
    train_bpe_tokenizer,
)


def test_get_training_corpus_basic() -> None:
    """Test that get_training_corpus yields text samples correctly."""
    dataset = Dataset.from_dict({"text": ["Hello", "world", "test"]})

    corpus = list(get_training_corpus(dataset))

    assert corpus == ["Hello", "world", "test"]


def test_get_training_corpus_with_max_samples() -> None:
    """Test that max_samples limits the number of yielded samples."""
    dataset = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"]})

    corpus = list(get_training_corpus(dataset, max_samples=3))

    assert corpus == ["a", "b", "c"]
    assert len(corpus) == 3


def test_get_training_corpus_max_samples_exceeds_dataset() -> None:
    """Test that max_samples larger than dataset yields all samples."""
    dataset = Dataset.from_dict({"text": ["x", "y"]})

    corpus = list(get_training_corpus(dataset, max_samples=100))

    assert corpus == ["x", "y"]


@patch("zmaj_lm.data.tokenizer.train_tokenizer.load_dataset")
def test_load_datasets_single_dataset(mock_load_dataset: Mock) -> None:
    """Test loading a single dataset without mixing."""
    mock_dataset = Dataset.from_dict({"text": ["sample1", "sample2"]})
    mock_load_dataset.return_value = {"train": mock_dataset}

    config = TokenizerTrainingConfig(
        datasets=DatasetConfig(path="test_dataset", split="train"),
        vocab_size=1000,
    )

    result = _load_datasets(config)

    assert len(result) == 2
    assert result["text"][0] == "sample1"
    mock_load_dataset.assert_called_once()


@patch("zmaj_lm.data.tokenizer.train_tokenizer.load_dataset")
@patch("zmaj_lm.data.tokenizer.train_tokenizer.interleave_datasets")
def test_load_datasets_multiple_with_weights(
    mock_interleave: Mock, mock_load_dataset: Mock
) -> None:
    """Test loading multiple datasets with weight normalization."""
    dataset1 = Dataset.from_dict({"text": ["a", "b"]})
    dataset2 = Dataset.from_dict({"content": ["c", "d"]})

    mock_load_dataset.side_effect = [{"train": dataset1}, {"train": dataset2}]
    mock_interleave.return_value = Dataset.from_dict({"text": ["a", "c", "b", "d"]})

    config = TokenizerTrainingConfig(
        datasets=[
            DatasetConfig(path="dataset1", split="train", weight=3.0),
            DatasetConfig(path="dataset2", split="train", text_column="content", weight=1.0),
        ],
        vocab_size=1000,
    )

    _load_datasets(config)

    # Check interleave was called with normalized probabilities
    mock_interleave.assert_called_once()
    call_kwargs = mock_interleave.call_args[1]
    assert call_kwargs["probabilities"] == [0.75, 0.25]  # 3/(3+1), 1/(3+1)


@patch("zmaj_lm.data.tokenizer.train_tokenizer.load_dataset")
def test_load_datasets_column_renaming(mock_load_dataset: Mock) -> None:
    """Test that non-text columns are correctly renamed to 'text'."""
    mock_dataset = Dataset.from_dict({"content": ["sample1", "sample2"], "other": ["x", "y"]})
    mock_load_dataset.return_value = {"train": mock_dataset}

    config = TokenizerTrainingConfig(
        datasets=DatasetConfig(path="test", split="train", text_column="content"),
        vocab_size=1000,
    )

    result = _load_datasets(config)

    assert "text" in result.column_names
    assert "content" not in result.column_names
    assert "other" not in result.column_names


@patch("zmaj_lm.data.tokenizer.train_tokenizer._load_datasets")
def test_train_bpe_tokenizer_basic(mock_load_datasets: Mock, tmp_path: Path) -> None:
    """Test basic tokenizer training with minimal dataset."""
    mock_dataset = Dataset.from_dict(
        {"text": ["Hello world", "Testing tokenizer", "Machine learning is fun"]}
    )
    mock_load_datasets.return_value = mock_dataset

    config = TokenizerTrainingConfig(
        datasets=DatasetConfig(path="dummy", split="train"),
        vocab_size=500,
        save_path=str(tmp_path / "test_tokenizer.json"),
    )

    tokenizer = train_bpe_tokenizer(config)

    # Check tokenizer was created and saved
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.get_vocab_size() > config.num_special_tokens  # More than just special tokens
    assert (tmp_path / "test_tokenizer.json").exists()

    # Test that special tokens are in vocabulary
    vocab = tokenizer.get_vocab()
    assert config.bos_token in vocab
    assert config.eos_token in vocab
    assert config.unk_token in vocab
    assert config.pad_token in vocab
