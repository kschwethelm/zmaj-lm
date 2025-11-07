#!/usr/bin/env python3
"""Train a BPE tokenizer from a YAML configuration file.

Usage:
    uv run scripts/run_train_tokenizer.py configs/tokenizer_train/bpe_32k_eng_code.yaml
"""

import sys
from pathlib import Path

import yaml
from loguru import logger

from zmaj_lm.config.tokenizer_config import TokenizerTrainingConfig
from zmaj_lm.data.tokenizer.evaluate_tokenizer import evaluate_tokenizer
from zmaj_lm.data.tokenizer.train_tokenizer import train_bpe_tokenizer


def main() -> None:
    """Train tokenizer from config file."""
    if len(sys.argv) != 2:
        logger.error("Usage: uv run scripts/run_train_tokenizer.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = TokenizerTrainingConfig(**config_dict)
    logger.info(f"Training tokenizer with vocab_size={config.vocab_size}")

    tokenizer = train_bpe_tokenizer(config)

    logger.success(f"Tokenizer training complete! Saved to {config.save_path}")
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    if config.run_evaluation:
        evaluate_tokenizer(tokenizer, config)


if __name__ == "__main__":
    main()
