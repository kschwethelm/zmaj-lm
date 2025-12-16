"""Tests for LMDataset - basic smoke tests for PyTorch migration."""

import torch

from zmaj_lm.data.dataset import LMDataset


class TestLMDatasetBasic:
    """Basic smoke tests for LMDataset functionality."""

    def test_dataset_creation_with_packing(self) -> None:
        """Test creating dataset with sequence packing."""
        # Create simple token sequences
        token_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            prevent_cross_doc_attention=False,
            shuffle=False,
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
            batch_size=1,
            use_packing=False,
            padding_token=0,
            shuffle=False,
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
            batch_size=1,
            use_packing=True,
            shuffle=False,
        )

        batch = next(iter(dataset))

        # Should return dict with expected keys
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "target_ids" in batch
        assert "attention_mask" in batch

        # All values should be PyTorch tensors
        assert isinstance(batch["input_ids"], torch.Tensor)
        assert isinstance(batch["target_ids"], torch.Tensor)
        assert isinstance(batch["attention_mask"], torch.Tensor)

    def test_block_diagonal_mask_with_packing(self) -> None:
        """Test that document boundaries create proper masks."""
        token_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        dataset = LMDataset(
            token_ids,
            seq_len=4,
            batch_size=1,
            use_packing=True,
            prevent_cross_doc_attention=True,
            shuffle=False,
        )

        # Should track document IDs
        assert hasattr(dataset, "doc_ids")
        # Should create 2D or 3D attention masks
        assert dataset.attention_masks.ndim in [2, 3]
