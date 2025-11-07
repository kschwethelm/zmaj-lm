from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from loguru import logger
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from zmaj_lm.config.tokenizer_config import DatasetConfig, TokenizerTrainingConfig
from zmaj_lm.data.tokenizer.utils import build_load_args


def train_bpe_tokenizer(config: TokenizerTrainingConfig) -> Tokenizer:
    """Train a byte-level BPE tokenizer from scratch using the tokenizers library.

    This uses a GPT-2 style configuration:
    - Byte-level BPE for handling any Unicode character
    - No normalization (byte-level handles everything)
    - Template processing to add BOS/EOS tokens

    Args:
        config: Configuration for tokenizer training

    Returns:
        Trained tokenizer ready for use
    """
    tokenizer = Tokenizer(models.BPE(unk_token=config.unk_token))

    # Configure byte-level pre-tokenizer (GPT-2 style)
    # This splits on whitespace and punctuation while handling all Unicode via bytes
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # No normalization needed - byte-level handles everything
    tokenizer.normalizer = normalizers.Sequence([])

    # Configure decoder to convert bytes back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Configure post-processor to add BOS/EOS tokens
    # Note: Token IDs here are temporary placeholders and will be updated
    # to the actual IDs assigned during training
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[
            (config.bos_token, 1),
            (config.eos_token, 2),
        ],
    )

    # Load training data
    mixed_dataset = _load_datasets(config)
    training_corpus = get_training_corpus(mixed_dataset, max_samples=config.max_training_samples)

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        show_progress=True,
    )

    # Train the tokenizer
    logger.info(f"Training BPE tokenizer with vocab_size={config.vocab_size}...")
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    # Save the trained tokenizer
    logger.info(f"Saving tokenizer to {config.save_path}")
    save_path = Path(config.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))

    return tokenizer


def _load_datasets(config: TokenizerTrainingConfig) -> Dataset:
    """Load and mix datasets from config using weighted interleaving.

    Args:
        config: Tokenizer training configuration

    Returns:
        Mixed dataset with unified 'text' column
    """
    # Normalize datasets to list format
    dataset_configs = (
        [config.datasets] if isinstance(config.datasets, DatasetConfig) else config.datasets
    )

    logger.info(f"Loading {len(dataset_configs)} dataset(s)...")
    datasets: list[Dataset] = []
    weights: list[float] = []

    for dataset_config in dataset_configs:
        # Load dataset
        load_args = build_load_args(dataset_config)
        raw_dataset = load_dataset(**load_args)
        dataset = raw_dataset[dataset_config.split]

        # Normalize text column to 'text' if needed
        dataset = dataset.select_columns([dataset_config.text_column])
        if dataset_config.text_column != "text":
            dataset = dataset.rename_column(dataset_config.text_column, "text")

        logger.info(
            f"Loaded {dataset_config.path}"
            + (f"/{dataset_config.name}" if dataset_config.name else "")
            + f" ({dataset_config.split}): {len(dataset)} samples "
            + f"(weight: {dataset_config.weight})"
        )

        datasets.append(dataset)
        weights.append(dataset_config.weight)

    # Mix datasets if multiple, otherwise return the single dataset
    if len(datasets) == 1:
        logger.info("Using single dataset")
        return datasets[0]
    else:
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        logger.info(f"Mixing {len(datasets)} datasets with probabilities: {probabilities}")
        mixed_dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=config.seed,
            stopping_strategy="all_exhausted",
        )
        return mixed_dataset


def get_training_corpus(dataset: Dataset, max_samples: int | None = None) -> Iterator[str]:
    """Yield individual text samples from a dataset.

    Args:
        dataset: HuggingFace dataset with 'text' column
        max_samples: Optional limit on total number of samples to yield

    Yields:
        Individual text strings
    """
    for idx, sample in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            return

        yield sample["text"]
