"""Tests for LMDataset - basic smoke tests for PyTorch migration."""

import torch
from torch.utils.data import DataLoader

from zmaj_lm.data.dataset import LMDataset, lm_collate_fn


class TestLMDatasetBasic:
    """Basic smoke tests for LMDataset functionality."""

    def test_dataset_creation_with_packing(self) -> None:
        """Test creating dataset with sequence packing."""
        # Create simple token sequences
        token_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            use_packing=True,
            prevent_cross_doc_attention=False,
            shuffle_docs=False,
        )

        # Should create chunks of length 4
        assert dataset.chunks.shape[1] == 4
        # Should return PyTorch tensors
        assert isinstance(dataset.chunks, torch.Tensor)
        assert isinstance(dataset.attention_masks, torch.Tensor)

    def test_dataset_creation_with_padding(self) -> None:
        """Test creating dataset with padding."""
        token_ids = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]

        dataset = LMDataset(
            token_ids,
            seq_len=5,
            use_packing=False,
            padding_token=0,
            shuffle_docs=False,
        )

        # Should pad to seq_len
        assert dataset.chunks.shape == (2, 5)
        # Should return PyTorch tensors
        assert isinstance(dataset.chunks, torch.Tensor)
        assert isinstance(dataset.attention_masks, torch.Tensor)

    def test_getitem_returns_correct_format(self) -> None:
        """Test that __getitem__ returns dict with correct keys."""
        token_ids = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            use_packing=True,
            shuffle_docs=False,
        )

        # Get a single sample
        sample = dataset[0]

        # Should return dict with expected keys
        assert isinstance(sample, dict)
        assert "tokens" in sample
        assert "attention_mask" in sample

        # All values should be PyTorch tensors
        assert isinstance(sample["tokens"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)

        # Sample should have seq_len
        assert sample["tokens"].shape[0] == 4

    def test_dataloader_with_collate_fn(self) -> None:
        """Test that DataLoader with custom collate function produces correct batches."""
        token_ids = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            use_packing=True,
            shuffle_docs=False,
        )

        def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return lm_collate_fn(batch, use_packing_with_boundaries=False)

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        batch = next(iter(dataloader))

        # Should return dict with expected keys
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "target_ids" in batch
        assert "attention_mask" in batch

        # All values should be PyTorch tensors
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["target_ids"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)

        # Batch should have correct shape (batch_size, seq_len - 1)
        assert batch["input_ids"].shape == (1, 3)
        assert batch["target_ids"].shape == (1, 3)

    def test_block_diagonal_mask_with_packing(self) -> None:
        """Test that document boundaries create proper masks."""
        token_ids = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            use_packing=True,
            prevent_cross_doc_attention=True,
            shuffle_docs=False,
        )

        # Should create 2D attention masks (seq_len, seq_len) for block-diagonal
        assert dataset.attention_masks.ndim == 3  # (num_chunks, seq_len, seq_len)
        assert dataset.attention_masks.shape[1] == 4
        assert dataset.attention_masks.shape[2] == 4
