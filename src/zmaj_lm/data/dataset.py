from collections.abc import Iterator

import datasets as ds
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from zmaj_lm.config.dataset_config import DatasetConfig
from zmaj_lm.utils.masks import create_block_diagonal_mask, create_packing_mask
from zmaj_lm.utils.prng import key_generator


class LMDataset:
    """Tokenized language model dataset.

    Args:
        token_ids: List of tokenized sequences as numpy arrays
        seq_len: Fixed sequence length for chunking
        batch_size: Number of sequences per batch
        use_packing: If True, concatenate and pack sequences. If False, pad individually.
        padding_token: Token ID to use for padding (required if use_packing=False)
        prevent_cross_doc_attention: If True with packing, create masks to prevent cross-document attention
        shuffle: Whether to shuffle documents before packing and chunks between epochs
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        token_ids: list[jax.Array],
        seq_len: int,
        batch_size: int,
        use_packing: bool = True,
        padding_token: int = 0,
        prevent_cross_doc_attention: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.use_packing = use_packing
        self.padding_token = padding_token
        self.prevent_cross_doc_attention = prevent_cross_doc_attention
        self.shuffle = shuffle
        self.key_gen = key_generator(jax.random.PRNGKey(seed))

        # Shuffle documents before packing/padding to avoid ordering biases
        # This ensures documents from similar topics/sources aren't packed together
        if shuffle:
            shuffle_key = next(self.key_gen)
            doc_indices = jax.random.permutation(shuffle_key, jnp.arange(len(token_ids)))
            token_ids = [token_ids[int(i)] for i in doc_indices]

        # Preprocess: chunk sequences and create masks
        self.doc_ids: jax.Array | None
        if use_packing:
            if prevent_cross_doc_attention:
                self.chunks, self.doc_ids, self.attention_masks = (
                    self._pack_sequences_with_boundaries(token_ids)
                )
            else:
                self.chunks, self.attention_masks = self._pack_sequences(token_ids)
                self.doc_ids = None
        else:
            self.chunks, self.attention_masks = self._pad_sequences(token_ids, padding_token)
            self.doc_ids = None

        self.num_batches = len(self.chunks) // batch_size

    def _pack_sequences(self, token_ids: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
        """Concatenate all sequences and chunk into fixed lengths.

        This maximizes GPU utilization by removing padding, but documents
        are concatenated so attention can cross document boundaries unless
        prevent_cross_doc_attention is enabled.

        Returns:
            Tuple of (chunks, attention_masks) where:
            - chunks: Token ID arrays of shape (num_chunks, seq_len)
            - attention_masks: All-True mask of shape (num_chunks, seq_len), dtype bool
        """
        all_tokens = jnp.concatenate(token_ids)
        num_chunks = len(all_tokens) // self.seq_len
        # Reshape into chunks, dropping incomplete final chunk
        chunks = all_tokens[: num_chunks * self.seq_len].reshape(-1, self.seq_len)

        # Create all-ones attention mask using utility function
        attention_masks = create_packing_mask(num_chunks, self.seq_len, dtype=jnp.bool_)

        return chunks, attention_masks

    def _pack_sequences_with_boundaries(
        self, token_ids: list[jax.Array]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Concatenate sequences and track document boundaries for block-diagonal attention.

        Returns:
            Tuple of (chunks, doc_ids, attention_masks) where:
            - chunks: Token ID arrays of shape (num_chunks, seq_len)
            - doc_ids: Document ID arrays of shape (num_chunks, seq_len) indicating which
                       document each token belongs to within the chunk
            - attention_masks: Block-diagonal masks of shape (num_chunks, seq_len, seq_len), dtype bool
        """
        # Concatenate all sequences
        all_tokens = jnp.concatenate(token_ids)

        # Track which document each token belongs to
        doc_ids_list: list[int] = []
        for doc_idx, seq in enumerate(token_ids):
            doc_ids_list.extend([doc_idx] * len(seq))
        doc_ids_array = jnp.array(doc_ids_list, dtype=jnp.int32)

        # Chunk into fixed lengths
        num_chunks = len(all_tokens) // self.seq_len
        total_tokens = num_chunks * self.seq_len

        chunks = all_tokens[:total_tokens].reshape(-1, self.seq_len)
        doc_id_chunks = doc_ids_array[:total_tokens].reshape(-1, self.seq_len)

        # Create block-diagonal attention masks using utility function
        attention_masks = create_block_diagonal_mask(doc_id_chunks, dtype=jnp.bool_)

        return chunks, doc_id_chunks, attention_masks

    def _pad_sequences(
        self, token_ids: list[jax.Array], padding_token: int
    ) -> tuple[jax.Array, jax.Array]:
        """Pad or truncate each sequence to fixed length.

        Returns:
            Tuple of (padded_sequences, attention_masks) where:
            - padded_sequences: Shape (num_sequences, seq_len)
            - attention_masks: Shape (num_sequences, seq_len), dtype bool (True for real tokens, False for padding)
        """
        padded_sequences_list: list[jax.Array] = []
        for seq in token_ids:
            if len(seq) < self.seq_len:
                padded_seq = jnp.pad(
                    seq, (0, self.seq_len - len(seq)), constant_values=padding_token
                )
            else:
                padded_seq = seq[: self.seq_len]
            padded_sequences_list.append(padded_seq)

        padded_sequences = jnp.stack(padded_sequences_list)
        # Create attention mask: True for real tokens, False for padding
        attention_masks = (padded_sequences != padding_token).astype(jnp.bool_)

        return padded_sequences, attention_masks

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over batches, yielding input-target pairs.

        Yields:
            Dictionary with keys:
                - input_ids: Shape (batch_size, seq_len - 1), dtype int
                - target_ids: Shape (batch_size, seq_len - 1), dtype int
                - attention_mask: Shape (batch_size, seq_len - 1) or (batch_size, seq_len - 1, seq_len - 1), dtype bool
                    - 1D: Simple padding/packing mask (True for real tokens, False for padding)
                    - 2D: Block-diagonal mask for document boundaries (when prevent_cross_doc_attention=True)
        """
        indices = jnp.arange(len(self.chunks))
        if self.shuffle:
            # Use JAX random for shuffling with fresh key for each epoch
            shuffle_key = next(self.key_gen)
            indices = jax.random.permutation(shuffle_key, indices)

        for i in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = self.chunks[batch_indices]  # Shape: (batch_size, seq_len)
            mask_batch = self.attention_masks[batch_indices]  # Pre-computed masks

            # Create input-target pairs (offset by 1 for autoregressive LM)
            input_ids = batch[:, :-1]  # All but last token
            target_ids = batch[:, 1:]  # All but first token

            # Slice the pre-computed attention masks to match input sequence length
            if self.use_packing and self.prevent_cross_doc_attention:
                # Block-diagonal mask: (batch_size, seq_len, seq_len) -> (batch_size, seq_len-1, seq_len-1)
                attention_mask = mask_batch[:, :-1, :-1]
            else:
                # 1D mask: (batch_size, seq_len) -> (batch_size, seq_len-1)
                attention_mask = mask_batch[:, :-1]

            yield {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "attention_mask": attention_mask,
            }

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches


def create_dataloaders(config: DatasetConfig) -> tuple[LMDataset, LMDataset]:
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

    # Extract token IDs as list of JAX arrays
    train_token_ids = [jnp.array(example["input_ids"]) for example in train_dataset]
    val_token_ids = [jnp.array(example["input_ids"]) for example in val_dataset]

    # Create dataloaders
    train_dataloader = LMDataset(
        train_token_ids,
        config.seq_len,
        config.batch_size,
        use_packing=config.use_packing,
        padding_token=padding_token,
        prevent_cross_doc_attention=config.prevent_cross_doc_attention,
        shuffle=config.shuffle,
        seed=config.seed,
    )
    val_dataloader = LMDataset(
        val_token_ids,
        config.seq_len,
        config.batch_size,
        use_packing=config.use_packing,  # Use same packing strategy for consistency
        padding_token=padding_token,
        prevent_cross_doc_attention=config.prevent_cross_doc_attention,
        shuffle=False,  # Never shuffle validation
        seed=config.seed,
    )

    return train_dataloader, val_dataloader
