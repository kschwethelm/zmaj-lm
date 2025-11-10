import pytest

from zmaj_lm.config.dataset_config import DatasetConfig
from zmaj_lm.data.dataset import create_dataloaders


@pytest.mark.slow
def test_create_dataloaders_with_real_data() -> None:
    """Integration test for create_dataloaders with HuggingFace dataset.

    This test requires network access to download a small dataset and tokenizer.
    It verifies the complete pipeline from dataset loading to dataloader creation.
    """
    config = DatasetConfig(
        dataset_name="mrdbourke/learn_hf_food_not_food_image_captions",  # Small dataset for testing
        tokenizer_path="EleutherAI/gpt-neo-125M",  # Small, commonly available tokenizer
        seq_len=64,  # Small sequence length for fast testing
        batch_size=2,
        split_ratio=0.8,
        use_packing=True,
        prevent_cross_doc_attention=False,
        shuffle=True,
        seed=42,
        num_proc=1,  # Single process for deterministic testing
    )

    train_loader, val_loader = create_dataloaders(config)

    # Verify dataloaders are created
    assert train_loader is not None
    assert val_loader is not None

    # Check that we have batches
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Verify batch structure
    train_batch = next(iter(train_loader))
    assert "input_ids" in train_batch
    assert "target_ids" in train_batch
    assert "attention_mask" in train_batch

    # Verify batch shapes
    assert train_batch["input_ids"].shape == (config.batch_size, config.seq_len - 1)
    assert train_batch["target_ids"].shape == (config.batch_size, config.seq_len - 1)
    # For packing without cross-doc attention, mask is 1D
    assert train_batch["attention_mask"].shape == (config.batch_size, config.seq_len - 1)

    # Verify validation loader doesn't shuffle
    val_batch1 = next(iter(val_loader))
    val_batch2 = next(iter(val_loader))
    # First batches should be the same across epochs when not shuffling
    assert (val_batch1["input_ids"] == val_batch2["input_ids"]).all()


@pytest.mark.slow
def test_create_dataloaders_with_document_boundaries() -> None:
    """Test create_dataloaders with prevent_cross_doc_attention enabled."""
    config = DatasetConfig(
        dataset_name="mrdbourke/learn_hf_food_not_food_image_captions",
        tokenizer_path="EleutherAI/gpt-neo-125M",
        seq_len=64,
        batch_size=2,
        split_ratio=0.8,
        use_packing=True,
        prevent_cross_doc_attention=True,  # Enable block-diagonal masks
        shuffle=False,
        seed=42,
        num_proc=1,
    )

    train_loader, val_loader = create_dataloaders(config)

    # Verify batch structure
    train_batch = next(iter(train_loader))

    # With prevent_cross_doc_attention, mask should be 2D (block-diagonal)
    assert train_batch["attention_mask"].ndim == 3  # (batch, seq_len-1, seq_len-1)
    assert train_batch["attention_mask"].shape == (
        config.batch_size,
        config.seq_len - 1,
        config.seq_len - 1,
    )


@pytest.mark.slow
def test_create_dataloaders_with_padding() -> None:
    """Test create_dataloaders with padding instead of packing."""
    config = DatasetConfig(
        dataset_name="mrdbourke/learn_hf_food_not_food_image_captions",
        tokenizer_path="EleutherAI/gpt-neo-125M",
        seq_len=64,
        batch_size=2,
        split_ratio=0.8,
        use_packing=False,  # Use padding
        shuffle=False,
        seed=42,
        num_proc=1,
    )

    train_loader, val_loader = create_dataloaders(config)

    # Verify batch structure
    train_batch = next(iter(train_loader))

    # Attention mask should indicate real vs padded tokens
    assert train_batch["attention_mask"].shape == (config.batch_size, config.seq_len - 1)
    # Some tokens should be masked (False) due to padding
    # This assumes at least some sequences are shorter than seq_len
