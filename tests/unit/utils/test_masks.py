"""Unit tests for attention mask utilities."""

import pytest
import torch

from zmaj_lm.utils.masks import (
    combine_masks,
    create_block_diagonal_mask,
    create_causal_mask,
    create_decoder_mask,
    create_packing_mask,
    create_padding_mask,
    mask_to_bias,
)


class TestCreateCausalMask:
    """Tests for create_causal_mask function."""

    def test_shape(self, device: torch.device) -> None:
        """Test that causal mask has correct shape."""
        seq_len = 8
        mask = create_causal_mask(seq_len, device=device)
        assert mask.shape == (1, seq_len, seq_len)

    def test_triangular_property(self, device: torch.device) -> None:
        """Test that causal mask is lower triangular."""
        mask = create_causal_mask(5, device=device)
        # Position i can attend to position j only if j <= i
        for i in range(5):
            for j in range(5):
                if j <= i:
                    assert mask[0, i, j], f"Position {i} should attend to {j}"
                else:
                    assert not mask[0, i, j], f"Position {i} should NOT attend to {j}"

    def test_dtype(self, device: torch.device) -> None:
        """Test that output dtype matches requested dtype."""
        mask_bool = create_causal_mask(4, dtype=torch.bool, device=device)
        assert mask_bool.dtype == torch.bool

        mask_int = create_causal_mask(4, dtype=torch.int32, device=device)
        assert mask_int.dtype == torch.int32

    def test_single_token(self, device: torch.device) -> None:
        """Test edge case with seq_len=1."""
        mask = create_causal_mask(1, device=device)
        assert mask.shape == (1, 1, 1)
        assert mask[0, 0, 0]

    def test_diagonal_all_true(self, device: torch.device) -> None:
        """Test that diagonal is all True (tokens can attend to themselves)."""
        mask = create_causal_mask(10, device=device)
        diagonal = torch.diag(mask[0])
        assert torch.all(diagonal)


class TestCreatePaddingMask:
    """Tests for create_padding_mask function."""

    def test_shape(self, device: torch.device) -> None:
        """Test that padding mask has correct shape."""
        batch_size = 4
        max_len = 10
        lengths = torch.tensor([5, 8, 3, 10], device=device)
        mask = create_padding_mask(lengths, max_len)
        assert mask.shape == (batch_size, max_len)

    def test_masking_correctness(self, device: torch.device) -> None:
        """Test that padding mask correctly identifies valid vs padding tokens."""
        lengths = torch.tensor([3, 5, 2], device=device)
        max_len = 5
        mask = create_padding_mask(lengths, max_len)

        # First sequence: length 3
        assert torch.equal(mask[0], torch.tensor([True, True, True, False, False], device=device))
        # Second sequence: length 5 (no padding)
        assert torch.equal(mask[1], torch.tensor([True, True, True, True, True], device=device))
        # Third sequence: length 2
        assert torch.equal(mask[2], torch.tensor([True, True, False, False, False], device=device))

    def test_all_same_length(self, device: torch.device) -> None:
        """Test when all sequences have same length (no actual padding)."""
        lengths = torch.tensor([7, 7, 7], device=device)
        max_len = 7
        mask = create_padding_mask(lengths, max_len)
        assert torch.all(mask)

    def test_single_sequence(self, device: torch.device) -> None:
        """Test with batch_size=1."""
        lengths = torch.tensor([4], device=device)
        max_len = 6
        mask = create_padding_mask(lengths, max_len)
        assert mask.shape == (1, 6)
        assert torch.equal(
            mask[0], torch.tensor([True, True, True, True, False, False], device=device)
        )

    def test_zero_length_sequence(self, device: torch.device) -> None:
        """Test edge case with length=0 (entirely padding)."""
        lengths = torch.tensor([0, 3], device=device)
        max_len = 4
        mask = create_padding_mask(lengths, max_len)
        # First sequence is all padding
        assert torch.equal(mask[0], torch.tensor([False, False, False, False], device=device))
        # Second sequence has 3 valid tokens
        assert torch.equal(mask[1], torch.tensor([True, True, True, False], device=device))


class TestCombineMasks:
    """Tests for combine_masks function."""

    def test_combine_two_masks(self, device: torch.device) -> None:
        """Test combining two masks with logical AND."""
        mask1 = torch.tensor([[True, True, False]], device=device)
        mask2 = torch.tensor([[True, False, False]], device=device)
        result = combine_masks(mask1, mask2)
        expected = torch.tensor([[True, False, False]], device=device)
        assert torch.equal(result, expected)

    def test_combine_multiple_masks(self, device: torch.device) -> None:
        """Test combining more than two masks."""
        mask1 = torch.tensor([[True, True, True, True]], device=device)
        mask2 = torch.tensor([[True, True, False, True]], device=device)
        mask3 = torch.tensor([[True, False, False, True]], device=device)
        result = combine_masks(mask1, mask2, mask3)
        expected = torch.tensor([[True, False, False, True]], device=device)
        assert torch.equal(result, expected)

    def test_broadcasting(self, device: torch.device) -> None:
        """Test that masks broadcast correctly."""
        # Shape (1, 3, 3)
        mask1 = torch.tensor(
            [[[True, False, False], [True, True, False], [True, True, True]]], device=device
        )
        # Shape (1, 1, 3)
        mask2 = torch.tensor([[[True, True, False]]], device=device)

        result = combine_masks(mask1, mask2)
        # Broadcasting should work: (1, 3, 3) and (1, 1, 3) -> (1, 3, 3)
        assert result.shape == (1, 3, 3)
        # Each row of mask1 should be AND'd with mask2
        assert torch.equal(result[0, 0], torch.tensor([True, False, False], device=device))
        assert torch.equal(result[0, 1], torch.tensor([True, True, False], device=device))
        assert torch.equal(result[0, 2], torch.tensor([True, True, False], device=device))

    def test_single_mask(self, device: torch.device) -> None:
        """Test that single mask is returned unchanged."""
        mask = torch.tensor([[True, False, True]], device=device)
        result = combine_masks(mask)
        assert torch.equal(result, mask)

    def test_no_masks_raises_error(self) -> None:
        """Test that providing no masks raises ValueError."""
        with pytest.raises(ValueError, match="At least one mask"):
            combine_masks()


class TestMaskToBias:
    """Tests for mask_to_bias function."""

    def test_values(self, device: torch.device) -> None:
        """Test that True->0.0 and False->mask_value conversion works."""
        mask = torch.tensor([[True, False, True]], device=device)
        bias = mask_to_bias(mask, mask_value=-1e10)
        assert bias[0, 0] == 0.0
        assert bias[0, 1] == -1e10
        assert bias[0, 2] == 0.0

    def test_custom_mask_value(self, device: torch.device) -> None:
        """Test using custom mask_value."""
        mask = torch.tensor([[True, False]], device=device)
        bias = mask_to_bias(mask, mask_value=-1e4)
        assert bias[0, 1] == -1e4

    def test_dtype(self, device: torch.device) -> None:
        """Test that output dtype matches requested dtype."""
        mask = torch.tensor([[True, False]], device=device)
        bias_f32 = mask_to_bias(mask, dtype=torch.float32)
        assert bias_f32.dtype == torch.float32

        bias_f16 = mask_to_bias(mask, dtype=torch.float16)
        assert bias_f16.dtype == torch.float16

    def test_shape_preserved(self, device: torch.device) -> None:
        """Test that output shape matches input shape."""
        mask = torch.ones((2, 5, 5), dtype=torch.bool, device=device)
        bias = mask_to_bias(mask)
        assert bias.shape == mask.shape

    def test_all_true(self, device: torch.device) -> None:
        """Test mask with all True values."""
        mask = torch.ones((3, 4), dtype=torch.bool, device=device)
        bias = mask_to_bias(mask)
        assert torch.all(bias == 0.0)

    def test_all_false(self, device: torch.device) -> None:
        """Test mask with all False values."""
        mask = torch.zeros((3, 4), dtype=torch.bool, device=device)
        bias = mask_to_bias(mask, mask_value=-1e10)
        assert torch.all(bias == -1e10)


class TestCreatePackingMask:
    """Tests for create_packing_mask function."""

    def test_shape(self, device: torch.device) -> None:
        """Test that packing mask has correct shape."""
        batch_size = 4
        seq_len = 10
        mask = create_packing_mask(batch_size, seq_len, device=device)
        assert mask.shape == (batch_size, seq_len)

    def test_all_ones(self, device: torch.device) -> None:
        """Test that packing mask is all ones (no padding)."""
        batch_size = 3
        seq_len = 8
        mask = create_packing_mask(batch_size, seq_len, device=device)
        assert torch.all(mask)

    def test_dtype(self, device: torch.device) -> None:
        """Test that output dtype matches requested dtype."""
        mask_bool = create_packing_mask(2, 5, dtype=torch.bool, device=device)
        assert mask_bool.dtype == torch.bool

        mask_int = create_packing_mask(2, 5, dtype=torch.int32, device=device)
        assert mask_int.dtype == torch.int32

    def test_single_sequence(self, device: torch.device) -> None:
        """Test with batch_size=1."""
        mask = create_packing_mask(1, 10, device=device)
        assert mask.shape == (1, 10)
        assert torch.all(mask)

    def test_single_token(self, device: torch.device) -> None:
        """Test edge case with seq_len=1."""
        mask = create_packing_mask(3, 1, device=device)
        assert mask.shape == (3, 1)
        assert torch.all(mask)


class TestCreateBlockDiagonalMask:
    """Tests for create_block_diagonal_mask function."""

    def test_shape(self, device: torch.device) -> None:
        """Test that block diagonal mask has correct shape."""
        batch_size = 2
        seq_len = 6
        doc_ids = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1]], device=device)
        mask = create_block_diagonal_mask(doc_ids)
        assert mask.shape == (batch_size, seq_len, seq_len)

    def test_same_document_attention(self, device: torch.device) -> None:
        """Test that tokens from same document can attend to each other."""
        # Single batch with doc_ids: [0, 0, 1, 1]
        doc_ids = torch.tensor([[0, 0, 1, 1]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # Tokens 0 and 1 are from doc 0, should attend to each other
        assert mask[0, 0, 0]
        assert mask[0, 0, 1]
        assert mask[0, 1, 0]
        assert mask[0, 1, 1]

        # Tokens 2 and 3 are from doc 1, should attend to each other
        assert mask[0, 2, 2]
        assert mask[0, 2, 3]
        assert mask[0, 3, 2]
        assert mask[0, 3, 3]

    def test_different_document_no_attention(self, device: torch.device) -> None:
        """Test that tokens from different documents cannot attend to each other."""
        # Single batch with doc_ids: [0, 0, 1, 1]
        doc_ids = torch.tensor([[0, 0, 1, 1]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # Tokens from doc 0 should NOT attend to tokens from doc 1
        assert not mask[0, 0, 2]
        assert not mask[0, 0, 3]
        assert not mask[0, 1, 2]
        assert not mask[0, 1, 3]

        # Tokens from doc 1 should NOT attend to tokens from doc 0
        assert not mask[0, 2, 0]
        assert not mask[0, 2, 1]
        assert not mask[0, 3, 0]
        assert not mask[0, 3, 1]

    def test_single_document(self, device: torch.device) -> None:
        """Test when all tokens belong to the same document."""
        # All tokens from document 0
        doc_ids = torch.tensor([[0, 0, 0, 0]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # All tokens should be able to attend to all other tokens
        assert torch.all(mask)

    def test_multiple_documents(self, device: torch.device) -> None:
        """Test with multiple documents in sequence."""
        # Documents: [0, 1, 1, 2, 2, 2]
        doc_ids = torch.tensor([[0, 1, 1, 2, 2, 2]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # Token 0 (doc 0) is isolated
        assert mask[0, 0, 0]
        for j in range(1, 6):
            assert not mask[0, 0, j]

        # Tokens 1,2 (doc 1) attend to each other
        assert mask[0, 1, 1] and mask[0, 1, 2]
        assert mask[0, 2, 1] and mask[0, 2, 2]

        # Tokens 3,4,5 (doc 2) attend to each other
        assert mask[0, 3, 3] and mask[0, 3, 4] and mask[0, 3, 5]
        assert mask[0, 4, 3] and mask[0, 4, 4] and mask[0, 4, 5]
        assert mask[0, 5, 3] and mask[0, 5, 4] and mask[0, 5, 5]

    def test_batch_processing(self, device: torch.device) -> None:
        """Test with multiple sequences in batch."""
        # Batch of 2 with different document structures
        doc_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # First sequence: [0, 0, 1, 1]
        assert mask[0, 0, 1] and not mask[0, 0, 2]
        assert mask[1, 0, 0] and not mask[1, 0, 1]

    def test_dtype(self, device: torch.device) -> None:
        """Test that output dtype matches requested dtype."""
        doc_ids = torch.tensor([[0, 0, 1, 1]], device=device)

        mask_bool = create_block_diagonal_mask(doc_ids, dtype=torch.bool)
        assert mask_bool.dtype == torch.bool

        mask_int = create_block_diagonal_mask(doc_ids, dtype=torch.int32)
        assert mask_int.dtype == torch.int32

    def test_symmetric(self, device: torch.device) -> None:
        """Test that mask is symmetric (if i can attend to j, then j can attend to i)."""
        doc_ids = torch.tensor([[0, 0, 1, 1, 2]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # Mask should be symmetric
        for i in range(5):
            for j in range(5):
                assert mask[0, i, j] == mask[0, j, i]

    def test_diagonal_all_true(self, device: torch.device) -> None:
        """Test that diagonal is all True (tokens can attend to themselves)."""
        doc_ids = torch.tensor([[0, 1, 2, 1, 0]], device=device)
        mask = create_block_diagonal_mask(doc_ids)

        # Every token should attend to itself
        for i in range(5):
            assert mask[0, i, i]


class TestCreateDecoderMask:
    """Tests for create_decoder_mask function."""

    def test_shape_without_padding(self, device: torch.device) -> None:
        """Test shape when no padding mask is provided."""
        seq_len = 6
        mask = create_decoder_mask(seq_len, attention_mask=None, device=device)
        assert mask.shape == (1, seq_len, seq_len)

    def test_shape_with_padding(self, device: torch.device) -> None:
        """Test shape when padding mask is provided."""
        seq_len = 5
        batch_size = 3
        lengths = torch.tensor([4, 5, 2], device=device)
        attention_mask = create_padding_mask(lengths, seq_len)
        mask = create_decoder_mask(seq_len, device=device, attention_mask=attention_mask)
        assert mask.shape == (batch_size, seq_len, seq_len)

    def test_causal_property_preserved(self, device: torch.device) -> None:
        """Test that causal property is maintained in combined mask."""
        lengths = torch.tensor([5, 5], device=device)  # No actual padding
        attention_mask = create_padding_mask(lengths, 5)
        mask = create_decoder_mask(5, device=device, attention_mask=attention_mask)

        # Should still be causal (lower triangular)
        for b in range(2):
            for i in range(5):
                for j in range(5):
                    if j > i:
                        assert not mask[b, i, j], f"Future masking violated at [{b},{i},{j}]"

    def test_padding_property_preserved(self, device: torch.device) -> None:
        """Test that padding positions are masked out."""
        seq_len = 4
        lengths = torch.tensor([2, 4], device=device)
        attention_mask = create_padding_mask(lengths, seq_len)
        mask = create_decoder_mask(seq_len, device=device, attention_mask=attention_mask)

        # First sequence: length 2, so positions 2,3 are padding
        # No position should be able to attend to padding positions
        for i in range(4):
            assert not mask[0, i, 2], f"Position {i} should not attend to padding at 2"
            assert not mask[0, i, 3], f"Position {i} should not attend to padding at 3"

        # Second sequence: length 4, no padding, only causal masking applies
        assert mask[1, 3, 2]  # Position 3 can attend to 2
        assert not mask[1, 2, 3]  # Position 2 cannot attend to 3 (future)

    def test_combined_constraints(self, device: torch.device) -> None:
        """Test that both causal and padding constraints are enforced."""
        seq_len = 3
        lengths = torch.tensor([2], device=device)  # Length 2, so position 2 is padding
        attention_mask = create_padding_mask(lengths, seq_len)
        mask = create_decoder_mask(seq_len, device=device, attention_mask=attention_mask)

        # Position 0: can attend to 0 only (position 0 is valid, 1,2 are future/padding)
        assert mask[0, 0, 0]
        assert not mask[0, 0, 1]  # Future
        assert not mask[0, 0, 2]  # Future + padding

        # Position 1: can attend to 0,1 (both valid, 2 is future+padding)
        assert mask[0, 1, 0]
        assert mask[0, 1, 1]
        assert not mask[0, 1, 2]  # Future + padding

        # Position 2: is padding, but let's check anyway
        # Can only attend to positions 0,1 (not 2, even though causal allows it, because 2 is padding)
        assert mask[0, 2, 0]
        assert mask[0, 2, 1]
        assert not mask[0, 2, 2]  # Padding position

    def test_no_padding_equals_causal(self, device: torch.device) -> None:
        """Test that with attention_mask=None, output equals pure causal mask."""
        seq_len = 4
        decoder_mask = create_decoder_mask(seq_len, attention_mask=None, device=device)
        causal_mask = create_causal_mask(seq_len, device=device)
        assert torch.equal(decoder_mask, causal_mask)

    def test_dtype(self, device: torch.device) -> None:
        """Test that output dtype matches requested dtype."""
        lengths = torch.tensor([3, 4], device=device)
        attention_mask = create_padding_mask(lengths, 4)
        mask_bool = create_decoder_mask(
            4, device=device, attention_mask=attention_mask, dtype=torch.bool
        )
        assert mask_bool.dtype == torch.bool

        mask_int = create_decoder_mask(
            4, device=device, attention_mask=attention_mask, dtype=torch.int32
        )
        assert mask_int.dtype == torch.int32
