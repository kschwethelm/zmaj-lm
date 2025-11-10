from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and preprocessing.

    This config defines parameters for loading HuggingFace datasets,
    tokenizing text, and creating dataloaders for language model training.
    """

    # Dataset source
    dataset_name: str  # HuggingFace dataset name (e.g., "roneneldan/TinyStories")
    dataset_config: str | None = None  # Optional dataset subset/config (e.g., "wikitext-2-raw-v1")
    tokenizer_path: str  # Path to pretrained tokenizer or HF tokenizer name

    # Sequence and batch configuration
    seq_len: int = 1024  # Fixed sequence length for chunking
    batch_size: int = 32  # Number of sequences per batch

    # Data splitting
    split_ratio: float = 0.95  # Fraction of data for training

    # Sequence packing configuration
    use_packing: bool = True  # If True, pack sequences without padding for efficiency
    prevent_cross_doc_attention: bool = False  # If True with packing, create block-diagonal masks

    # Training configuration
    shuffle: bool = True  # Whether to shuffle training data
    seed: int = 42  # Random seed for reproducibility

    # Performance options
    cache_dir: str | None = None  # Optional directory for caching datasets
    num_proc: int = 8  # Number of processes for parallel tokenization
