"""Tokenizer evaluation functionality for assessing quality and performance."""

from datasets import load_dataset
from loguru import logger
from tokenizers import Tokenizer

from zmaj_lm.config.tokenizer_config import DatasetConfig, TokenizerTrainingConfig
from zmaj_lm.data.tokenizer.utils import build_load_args


def evaluate_tokenizer(tokenizer: Tokenizer, config: TokenizerTrainingConfig) -> dict[str, float]:
    """Evaluate tokenizer quality on validation samples and test strings.

    Computes key metrics:
    - Compression ratio: average tokens per character
    - Fertility: average subwords per whitespace-delimited word
    - Unknown token rate: proportion of UNK tokens in output

    Args:
        tokenizer: Trained tokenizer to evaluate
        config: Configuration containing evaluation parameters

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating tokenizer...")

    eval_texts = _get_eval_texts(config)
    metrics = _compute_tokenizer_metrics(tokenizer, eval_texts, config.unk_token)

    _log_evaluation_results(tokenizer, eval_texts, metrics)
    _show_example_tokenizations(tokenizer, config)
    _verify_roundtrip(tokenizer, eval_texts[:5])

    return metrics


def _get_eval_texts(config: TokenizerTrainingConfig) -> list[str]:
    """Get evaluation texts from config (validation set + custom test strings).

    Args:
        config: Configuration containing evaluation settings

    Returns:
        List of text strings for evaluation
    """
    eval_texts: list[str] = []

    if config.eval_test_strings:
        eval_texts.extend(config.eval_test_strings)

    if config.eval_samples > 0:
        eval_texts.extend(_load_validation_samples(config))

    return eval_texts


def _load_validation_samples(config: TokenizerTrainingConfig) -> list[str]:
    """Load validation samples from configured datasets.

    Args:
        config: Tokenizer training configuration

    Returns:
        List of validation text samples
    """
    validation_texts: list[str] = []
    dataset_configs = (
        [config.datasets] if isinstance(config.datasets, DatasetConfig) else config.datasets
    )

    for dataset_config in dataset_configs:
        try:
            load_args = build_load_args(dataset_config)
            raw_dataset = load_dataset(**load_args)

            # Use eval_split if specified, otherwise fall back to split
            split_name = (
                dataset_config.eval_split if dataset_config.eval_split else dataset_config.split
            )
            eval_dataset = raw_dataset[split_name]

            num_samples = min(config.eval_samples, len(eval_dataset))
            samples = eval_dataset.shuffle(seed=config.seed).select(range(num_samples))

            validation_texts.extend([sample[dataset_config.text_column] for sample in samples])
            logger.info(f"Loaded {num_samples} eval samples from {dataset_config.path}")

        except Exception as e:
            logger.warning(f"Could not load eval data from {dataset_config.path}: {e}")

    return validation_texts


def _compute_tokenizer_metrics(
    tokenizer: Tokenizer, texts: list[str], unk_token: str
) -> dict[str, float]:
    """Compute tokenizer quality metrics on a list of texts.

    Args:
        tokenizer: Tokenizer to evaluate
        texts: List of text strings
        unk_token: Unknown token string

    Returns:
        Dictionary containing compression_ratio, fertility, and unk_rate
    """
    total_chars = 0
    total_tokens = 0
    total_words = 0
    total_subwords = 0
    total_unks = 0

    unk_id = tokenizer.token_to_id(unk_token)

    for text in texts:
        encoding = tokenizer.encode(text)

        total_chars += len(text)
        total_tokens += len(encoding.ids)

        words = text.split()
        total_words += len(words)
        total_subwords += len(encoding.ids)

        if unk_id is not None:
            total_unks += sum(1 for token_id in encoding.ids if token_id == unk_id)

    compression_ratio = total_tokens / total_chars if total_chars > 0 else 0.0
    fertility = total_subwords / total_words if total_words > 0 else 0.0
    unk_rate = total_unks / total_tokens if total_tokens > 0 else 0.0

    return {
        "compression_ratio": compression_ratio,
        "fertility": fertility,
        "unk_rate": unk_rate,
    }


def _log_evaluation_results(
    tokenizer: Tokenizer, eval_texts: list[str], metrics: dict[str, float]
) -> None:
    """Log evaluation results summary.

    Args:
        tokenizer: Evaluated tokenizer
        eval_texts: Texts used for evaluation
        metrics: Computed metrics
    """
    logger.info("=" * 60)
    logger.info("Tokenizer Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"Evaluated on {len(eval_texts)} samples")
    logger.info(f"  Compression ratio: {metrics['compression_ratio']:.3f} tokens/char")
    logger.info(f"  Fertility: {metrics['fertility']:.3f} subwords/word")
    logger.info(f"  Unknown token rate: {metrics['unk_rate']:.4%}")
    logger.info("=" * 60)


def _show_example_tokenizations(tokenizer: Tokenizer, config: TokenizerTrainingConfig) -> None:
    """Show example tokenizations for different text types.

    Args:
        tokenizer: Tokenizer to demonstrate
        config: Configuration containing test strings
    """
    logger.info("Example tokenizations:")

    default_tests = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Neural networks learn from data. 🚀",
        "混合中文English文本",
    ]

    test_strings = config.eval_test_strings if config.eval_test_strings else default_tests

    for text in test_strings[:5]:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        logger.info(f"\nText: {text}")
        logger.info(f"Tokens ({len(tokens)}): {tokens}")


def _verify_roundtrip(tokenizer: Tokenizer, texts: list[str]) -> None:
    """Verify that encode->decode is a valid round trip.

    Args:
        tokenizer: Tokenizer to verify
        texts: Sample texts to test
    """
    logger.info("\nVerifying round-trip encoding/decoding...")

    all_passed = True
    for text in texts:
        encoding = tokenizer.encode(text)
        decoded_clean = tokenizer.decode(encoding.ids, skip_special_tokens=True)

        if decoded_clean.strip() != text.strip():
            logger.warning(f"Round-trip mismatch:\n  Original: {text}\n  Decoded: {decoded_clean}")
            all_passed = False

    if all_passed:
        logger.success("✓ Round-trip encoding/decoding verified")
    else:
        logger.warning("⚠ Some round-trip mismatches detected")
