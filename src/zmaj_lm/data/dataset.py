import datasets as ds
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from zmaj_lm.config.dataset_config import DatasetConfig
from zmaj_lm.utils.masks import create_block_diagonal_mask, create_packing_mask


def lm_collate_fn(
    batch: list[dict[str, torch.Tensor]], use_packing_with_boundaries: bool
) -> dict[str, torch.Tensor]:
    """Custom collate function for language model batches.

    Creates input-target pairs by offsetting sequences by 1 token and
    properly handles both 1D and 2D attention masks.

    Args:
        batch: List of samples from LMDataset, each containing 'tokens' and 'attention_mask'
        use_packing_with_boundaries: Whether the batch uses block-diagonal attention masks

    Returns:
        Dictionary with keys:
            - input_ids: Shape (batch_size, seq_len - 1), dtype long
            - target_ids: Shape (batch_size, seq_len - 1), dtype long
            - attention_mask: Shape (batch_size, seq_len - 1) or (batch_size, seq_len - 1, seq_len - 1), dtype bool
    """
    # Stack all tokens and masks
    tokens = torch.stack([sample["tokens"] for sample in batch])  # (batch_size, seq_len)
    attention_masks = torch.stack(
        [sample["attention_mask"] for sample in batch]
    )  # (batch_size, seq_len) or (batch_size, seq_len, seq_len)

    # Create input-target pairs (offset by 1 for autoregressive LM)
    input_ids = tokens[:, :-1]  # All but last token
    target_ids = tokens[:, 1:]  # All but first token

    # Slice the pre-computed attention masks to match input sequence length
    if use_packing_with_boundaries:
        # Block-diagonal mask: (batch_size, seq_len, seq_len) -> (batch_size, seq_len-1, seq_len-1)
        attention_mask = attention_masks[:, :-1, :-1]
    else:
        # 1D mask: (batch_size, seq_len) -> (batch_size, seq_len-1)
        attention_mask = attention_masks[:, :-1]

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }


class LMDataset(Dataset):
    """PyTorch Dataset for tokenized language model training.

    Supports both sequence packing (for efficiency) and padding (for simplicity).
    When packing is enabled, can optionally create block-diagonal attention masks
    to prevent cross-document attention.

    Args:
        token_ids: List of tokenized sequences as tensors
        seq_len: Fixed sequence length for chunking
        use_packing: If True, concatenate and pack sequences. If False, pad individually.
        padding_token: Token ID to use for padding (required if use_packing=False)
        prevent_cross_doc_attention: If True with packing, create masks to prevent cross-document attention
        shuffle_docs: Whether to shuffle documents before packing/chunking
        seed: Random seed for document shuffling
    """

    def __init__(
        self,
        token_ids: list[torch.Tensor],
        seq_len: int,
        use_packing: bool = True,
        padding_token: int = 0,
        prevent_cross_doc_attention: bool = False,
        shuffle_docs: bool = True,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.use_packing = use_packing
        self.padding_token = padding_token
        self.prevent_cross_doc_attention = prevent_cross_doc_attention

        # Shuffle documents before packing/padding to avoid ordering biases
        # This ensures documents from similar topics/sources aren't packed together
        if shuffle_docs:
            generator = torch.Generator().manual_seed(seed)
            doc_indices = torch.randperm(len(token_ids), generator=generator).tolist()
            token_ids = [token_ids[i] for i in doc_indices]

        # Preprocess: chunk sequences and create masks
        if use_packing:
            if prevent_cross_doc_attention:
                self.chunks, self.attention_masks = self._pack_sequences_with_boundaries(token_ids)
            else:
                self.chunks, self.attention_masks = self._pack_sequences(token_ids)
        else:
            self.chunks, self.attention_masks = self._pad_sequences(token_ids, padding_token)

    def _pack_sequences(self, token_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate all sequences and chunk into fixed lengths.

        This maximizes GPU utilization by removing padding, but documents
        are concatenated so attention can cross document boundaries unless
        prevent_cross_doc_attention is enabled.

        Returns:
            Tuple of (chunks, attention_masks) where:
            - chunks: Token ID arrays of shape (num_chunks, seq_len)
            - attention_masks: All-True mask of shape (num_chunks, seq_len), dtype bool
        """
        all_tokens = torch.cat(token_ids)
        num_chunks = len(all_tokens) // self.seq_len
        # Reshape into chunks, dropping incomplete final chunk
        chunks = all_tokens[: num_chunks * self.seq_len].reshape(-1, self.seq_len)

        # Create all-ones attention mask using utility function
        attention_masks = create_packing_mask(num_chunks, self.seq_len, device=torch.device("cpu"))

        return chunks, attention_masks

    def _pack_sequences_with_boundaries(
        self, token_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate sequences and track document boundaries for block-diagonal attention.

        Returns:
            Tuple of (chunks, attention_masks) where:
            - chunks: Token ID arrays of shape (num_chunks, seq_len)
            - attention_masks: Block-diagonal masks of shape (num_chunks, seq_len, seq_len), dtype bool
        """
        # Concatenate all sequences
        all_tokens = torch.cat(token_ids)

        # Track which document each token belongs to
        doc_ids_list: list[int] = []
        for doc_idx, seq in enumerate(token_ids):
            doc_ids_list.extend([doc_idx] * len(seq))
        doc_ids_array = torch.tensor(doc_ids_list, dtype=torch.int32)

        # Chunk into fixed lengths
        num_chunks = len(all_tokens) // self.seq_len
        total_tokens = num_chunks * self.seq_len

        chunks = all_tokens[:total_tokens].reshape(-1, self.seq_len)
        doc_id_chunks = doc_ids_array[:total_tokens].reshape(-1, self.seq_len)

        # Create block-diagonal attention masks using utility function
        attention_masks = create_block_diagonal_mask(doc_id_chunks, dtype=torch.bool)

        return chunks, attention_masks

    def _pad_sequences(
        self, token_ids: list[torch.Tensor], padding_token: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad or truncate each sequence to fixed length.

        Returns:
            Tuple of (padded_sequences, attention_masks) where:
            - padded_sequences: Shape (num_sequences, seq_len)
            - attention_masks: Shape (num_sequences, seq_len), dtype bool (True for real tokens, False for padding)
        """
        padded_sequences_list: list[torch.Tensor] = []
        for seq in token_ids:
            if len(seq) < self.seq_len:
                # PyTorch pad: (left, right) for 1D
                padded_seq = torch.nn.functional.pad(
                    seq, (0, self.seq_len - len(seq)), value=padding_token
                )
            else:
                padded_seq = seq[: self.seq_len]
            padded_sequences_list.append(padded_seq)

        padded_sequences = torch.stack(padded_sequences_list)
        # Create attention mask: True for real tokens, False for padding
        attention_masks = (padded_sequences != padding_token).to(torch.bool)

        return padded_sequences, attention_masks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sequence and its attention mask.

        Args:
            idx: Index of the chunk to retrieve

        Returns:
            Dictionary with keys:
                - tokens: Shape (seq_len,), dtype long
                - attention_mask: Shape (seq_len,) or (seq_len, seq_len), dtype bool
        """
        return {
            "tokens": self.chunks[idx],
            "attention_mask": self.attention_masks[idx],
        }

    def __len__(self) -> int:
        """Return the number of chunks (samples) in the dataset."""
        return len(self.chunks)


def create_dataloaders(config: DatasetConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        config: DatasetConfig containing all dataset configuration parameters

    Returns:
        Tuple of (train_dataloader, val_dataloader)

    Note:
        With packing=True and prevent_cross_doc_attention=False, documents are
        concatenated end-to-end with EOS tokens as separators. This maximizes
        efficiency but allows cross-document attention.

        With prevent_cross_doc_attention=True, block-diagonal attention masks are
        created to enforce document boundaries at the cost of returning larger 2D masks.
    """
    # Load dataset
    dataset = ds.load_dataset(
        config.dataset_name,
        name=config.dataset_config,
        split="train",
        cache_dir=config.cache_dir,
    )

    # Load tokenizer and configure special tokens
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

    # Ensure tokenizer has required special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    padding_token = tokenizer.pad_token_id

    # Tokenize dataset with special tokens
    # add_special_tokens=True ensures BOS/EOS are added
    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            examples["text"],
            add_special_tokens=True,  # Add BOS/EOS tokens
            truncation=False,  # Don't truncate long documents
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        load_from_cache_file=True,  # Use cached tokenization if available
    )

    # Split train/val with the same seed for reproducibility
    split_dataset = tokenized_dataset.train_test_split(
        test_size=1 - config.split_ratio, seed=config.seed
    )
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    # Extract token IDs as list of PyTorch tensors
    train_token_ids = [
        torch.tensor(example["input_ids"], dtype=torch.long) for example in train_dataset
    ]
    val_token_ids = [
        torch.tensor(example["input_ids"], dtype=torch.long) for example in val_dataset
    ]

    # Create datasets
    train_lm_dataset = LMDataset(
        train_token_ids,
        config.seq_len,
        use_packing=config.use_packing,
        padding_token=padding_token,
        prevent_cross_doc_attention=config.prevent_cross_doc_attention,
        shuffle_docs=config.shuffle,
        seed=config.seed,
    )
    val_lm_dataset = LMDataset(
        val_token_ids,
        config.seq_len,
        use_packing=config.use_packing,  # Use same packing strategy for consistency
        padding_token=padding_token,
        prevent_cross_doc_attention=config.prevent_cross_doc_attention,
        shuffle_docs=False,  # Never shuffle validation documents
        seed=config.seed,
    )

    # Create collate function with proper configuration
    use_packing_with_boundaries = config.use_packing and config.prevent_cross_doc_attention

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return lm_collate_fn(batch, use_packing_with_boundaries)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_lm_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster transfer to GPU
        persistent_workers=config.num_workers > 0,  # Keep workers alive between epochs
    )
    val_dataloader = DataLoader(
        val_lm_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    return train_dataloader, val_dataloader
