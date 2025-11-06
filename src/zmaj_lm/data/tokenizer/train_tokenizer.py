from collections.abc import Iterator

from tokenizers import Tokenizer

from zmaj_lm.config.tokenizer_config import TokenizerTrainingConfig


def train_bpe_tokenizer(config: TokenizerTrainingConfig) -> Tokenizer:
    pass


def get_training_corpus(path: str, max_samples: int | None = None) -> Iterator[str]:
    raise NotImplementedError
