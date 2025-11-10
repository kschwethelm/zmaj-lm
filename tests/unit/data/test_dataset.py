import jax.numpy as jnp

from zmaj_lm.data.dataset import LMDataset


class TestLMDatasetPacking:
    """Test sequence packing (concatenate all sequences, no padding)."""

    def test_pack_sequences_basic(self) -> None:
        """Sequences are concatenated and chunked into fixed lengths."""
        # Two docs: [1,2,3] and [4,5,6,7,8] -> concatenated: [1,2,3,4,5,6,7,8]
        # With seq_len=4, chunks into: [1,2,3,4] and [5,6,7,8]
        token_ids = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            prevent_cross_doc_attention=False,
            shuffle=False,
        )

        assert dataset.chunks.shape == (2, 4)
        assert dataset.attention_masks.shape == (2, 4)
        # All attention mask values are True for packing
        assert jnp.all(dataset.attention_masks)

    def test_pack_sequences_with_boundaries(self) -> None:
        """Packing with document boundaries creates block-diagonal masks."""
        # Two docs of length 3 each -> concatenated: [1,2,3,4,5,6]
        # Chunks into [1,2,3,4] and [5,6,...] (second incomplete, dropped)
        token_ids = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            prevent_cross_doc_attention=True,
            shuffle=False,
        )

        assert dataset.chunks.shape == (1, 4)  # Only 1 complete chunk
        assert dataset.doc_ids.shape == (1, 4)  # Document IDs tracked
        assert dataset.attention_masks.shape == (1, 4, 4)  # Block-diagonal mask

        # Check document IDs: first 3 tokens from doc 0, last token from doc 1
        assert jnp.array_equal(dataset.doc_ids[0], jnp.array([0, 0, 0, 1]))

        # Verify block-diagonal structure: tokens can only attend within their document
        mask = dataset.attention_masks[0]
        # Token 0 can attend to tokens 0,1,2 (doc 0) but not token 3 (doc 1)
        assert jnp.array_equal(mask[0], jnp.array([True, True, True, False]))
        # Token 3 can only attend to itself (doc 1)
        assert jnp.array_equal(mask[3], jnp.array([False, False, False, True]))


class TestLMDatasetPadding:
    """Test individual sequence padding (no packing)."""

    def test_pad_sequences_shorter_than_seq_len(self) -> None:
        """Short sequences are padded to seq_len."""
        token_ids = [jnp.array([1, 2]), jnp.array([3, 4, 5])]

        dataset = LMDataset(
            token_ids,
            seq_len=5,
            batch_size=1,
            use_packing=False,
            padding_token=0,
            shuffle=False,
        )

        assert dataset.chunks.shape == (2, 5)
        # First sequence: [1,2,0,0,0]
        assert jnp.array_equal(dataset.chunks[0], jnp.array([1, 2, 0, 0, 0]))
        # Second sequence: [3,4,5,0,0]
        assert jnp.array_equal(dataset.chunks[1], jnp.array([3, 4, 5, 0, 0]))

        # Attention mask: True for real tokens, False for padding
        assert jnp.array_equal(
            dataset.attention_masks[0], jnp.array([True, True, False, False, False])
        )
        assert jnp.array_equal(
            dataset.attention_masks[1], jnp.array([True, True, True, False, False])
        )

    def test_pad_sequences_truncate_longer_than_seq_len(self) -> None:
        """Long sequences are truncated to seq_len."""
        token_ids = [jnp.array([1, 2, 3, 4, 5, 6, 7])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=False,
            padding_token=0,
            shuffle=False,
        )

        # Truncated to first 4 tokens
        assert jnp.array_equal(dataset.chunks[0], jnp.array([1, 2, 3, 4]))
        # All tokens are real (no padding)
        assert jnp.all(dataset.attention_masks[0])


class TestLMDatasetIteration:
    """Test batch iteration and input-target pair creation."""

    def test_iteration_creates_input_target_pairs(self) -> None:
        """Batches contain input_ids, target_ids, and attention_mask."""
        # Simple packed dataset: [1,2,3,4,5,6,7,8]
        token_ids = [jnp.array([1, 2, 3, 4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=2,
            use_packing=True,
            shuffle=False,
        )

        batches = list(dataset)
        assert len(batches) == 1  # 2 chunks / batch_size=2 = 1 batch

        batch = batches[0]
        assert batch["input_ids"].shape == (2, 3)  # seq_len - 1
        assert batch["target_ids"].shape == (2, 3)
        assert batch["attention_mask"].shape == (2, 3)

        # First chunk [1,2,3,4]: input=[1,2,3], target=[2,3,4]
        assert jnp.array_equal(batch["input_ids"][0], jnp.array([1, 2, 3]))
        assert jnp.array_equal(batch["target_ids"][0], jnp.array([2, 3, 4]))

    def test_iteration_with_block_diagonal_mask(self) -> None:
        """Block-diagonal masks are correctly sliced for iteration."""
        token_ids = [jnp.array([1, 2, 3, 4])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            prevent_cross_doc_attention=True,
            shuffle=False,
        )

        batch = next(iter(dataset))
        # Mask should be 2D: (batch_size, seq_len-1, seq_len-1)
        assert batch["attention_mask"].shape == (1, 3, 3)

    def test_num_batches_calculation(self) -> None:
        """Number of batches is correctly calculated."""
        # 8 tokens -> 2 chunks with seq_len=4
        token_ids = [jnp.array(list(range(8)))]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=2,
            use_packing=True,
            shuffle=False,
        )

        assert len(dataset) == 1  # 2 chunks / batch_size=2 = 1 batch
        assert dataset.num_batches == 1

    def test_shuffle_affects_batch_order(self) -> None:
        """Shuffling changes batch order between epochs."""
        token_ids = [jnp.array(list(range(16)))]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            shuffle=True,
            seed=42,
        )

        # Collect first tokens from first 2 batches in epoch 1
        first_epoch_samples = jnp.array([next(iter(dataset))["input_ids"][0, 0] for _ in range(2)])
        # Collect first tokens from first 2 batches in epoch 2
        second_epoch_samples = jnp.array([next(iter(dataset))["input_ids"][0, 0] for _ in range(2)])

        # Different ordering between epochs (with high probability for seed=42)
        assert not jnp.array_equal(first_epoch_samples, second_epoch_samples)
