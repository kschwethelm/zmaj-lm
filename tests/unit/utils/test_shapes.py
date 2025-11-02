"""Unit tests for shape utilities."""

import jax.numpy as jnp
import pytest

from zmaj_lm.utils.shapes import (
    assert_shape,
    merge_heads,
    merge_heads_transposed,
    shape_str,
    split_heads,
    split_heads_transposed,
)


class TestSplitHeads:
    """Tests for split_heads function."""

    def test_shape(self) -> None:
        """Test that split_heads produces correct output shape."""
        batch, seq_len, d_model = 4, 128, 768
        n_heads = 12
        x = jnp.ones((batch, seq_len, d_model))

        result = split_heads(x, n_heads)

        assert result.shape == (batch, seq_len, n_heads, d_model // n_heads)

    def test_values_preserved(self) -> None:
        """Test that split_heads preserves values, only reshapes."""
        batch, seq_len, d_model = 2, 4, 8
        n_heads = 2
        # Create array with unique values
        x = jnp.arange(batch * seq_len * d_model).reshape(batch, seq_len, d_model)

        result = split_heads(x, n_heads)

        # Values should be preserved, just reshaped
        assert jnp.array_equal(result.reshape(batch, seq_len, d_model), x)

    def test_single_head(self) -> None:
        """Test edge case with n_heads=1."""
        batch, seq_len, d_model = 2, 10, 64
        x = jnp.ones((batch, seq_len, d_model))

        result = split_heads(x, n_heads=1)

        assert result.shape == (batch, seq_len, 1, d_model)

    def test_indivisible_raises_assertion(self) -> None:
        """Test that indivisible d_model raises assertion error."""
        batch, seq_len, d_model = 2, 10, 65  # 65 not divisible by 12
        x = jnp.ones((batch, seq_len, d_model))

        with pytest.raises(AssertionError, match="must be divisible"):
            split_heads(x, n_heads=12)

    def test_typical_transformer_dimensions(self) -> None:
        """Test with typical transformer dimensions."""
        # GPT-2 small: d_model=768, n_heads=12, d_head=64
        x = jnp.ones((4, 128, 768))
        result = split_heads(x, n_heads=12)
        assert result.shape == (4, 128, 12, 64)

        # GPT-2 medium: d_model=1024, n_heads=16, d_head=64
        x = jnp.ones((4, 128, 1024))
        result = split_heads(x, n_heads=16)
        assert result.shape == (4, 128, 16, 64)


class TestSplitHeadsTransposed:
    """Tests for split_heads_transposed function."""

    def test_shape(self) -> None:
        """Test that split_heads_transposed produces correct output shape."""
        batch, seq_len, d_model = 4, 128, 768
        n_heads = 12
        x = jnp.ones((batch, seq_len, d_model))

        result = split_heads_transposed(x, n_heads)

        # Should be [batch, n_heads, seq_len, d_head]
        assert result.shape == (batch, n_heads, seq_len, d_model // n_heads)

    def test_transpose_order(self) -> None:
        """Test that transpose reorders dimensions correctly."""
        batch, seq_len, d_model = 2, 3, 4
        n_heads = 2
        # Create array with unique values to track transposition
        x = jnp.arange(batch * seq_len * d_model).reshape(batch, seq_len, d_model)

        result = split_heads_transposed(x, n_heads)

        # Verify shape
        assert result.shape == (batch, n_heads, seq_len, d_model // n_heads)

        # Manually compute expected result
        expected = split_heads(x, n_heads).transpose(0, 2, 1, 3)
        assert jnp.array_equal(result, expected)

    def test_values_preserved(self) -> None:
        """Test that values are preserved after split and transpose."""
        batch, seq_len, d_model = 2, 4, 8
        n_heads = 2
        x = jnp.arange(batch * seq_len * d_model).reshape(batch, seq_len, d_model)

        result = split_heads_transposed(x, n_heads)

        # Transpose back and reshape should recover original
        recovered = result.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        assert jnp.array_equal(recovered, x)


class TestMergeHeads:
    """Tests for merge_heads function."""

    def test_shape(self) -> None:
        """Test that merge_heads produces correct output shape."""
        batch, seq_len, n_heads, d_head = 4, 128, 12, 64
        x = jnp.ones((batch, seq_len, n_heads, d_head))

        result = merge_heads(x)

        assert result.shape == (batch, seq_len, n_heads * d_head)

    def test_round_trip_with_split(self) -> None:
        """Test that split_heads -> merge_heads is identity."""
        batch, seq_len, d_model = 4, 128, 768
        n_heads = 12
        x = (
            jnp.arange(batch * seq_len * d_model)
            .reshape(batch, seq_len, d_model)
            .astype(jnp.float32)
        )

        # Round trip
        split = split_heads(x, n_heads)
        recovered = merge_heads(split)

        assert jnp.array_equal(recovered, x)
        assert recovered.shape == x.shape

    def test_single_head(self) -> None:
        """Test edge case with single head."""
        batch, seq_len, n_heads, d_head = 2, 10, 1, 64
        x = jnp.ones((batch, seq_len, n_heads, d_head))

        result = merge_heads(x)

        assert result.shape == (batch, seq_len, d_head)


class TestMergeHeadsTransposed:
    """Tests for merge_heads_transposed function."""

    def test_shape(self) -> None:
        """Test that merge_heads_transposed produces correct output shape."""
        batch, n_heads, seq_len, d_head = 4, 12, 128, 64
        x = jnp.ones((batch, n_heads, seq_len, d_head))

        result = merge_heads_transposed(x)

        assert result.shape == (batch, seq_len, n_heads * d_head)

    def test_round_trip_with_split_transposed(self) -> None:
        """Test that split_heads_transposed -> merge_heads_transposed is identity."""
        batch, seq_len, d_model = 4, 128, 768
        n_heads = 12
        x = (
            jnp.arange(batch * seq_len * d_model)
            .reshape(batch, seq_len, d_model)
            .astype(jnp.float32)
        )

        # Round trip with transposed operations
        split = split_heads_transposed(x, n_heads)
        assert split.shape == (batch, n_heads, seq_len, d_model // n_heads)

        recovered = merge_heads_transposed(split)
        assert jnp.array_equal(recovered, x)
        assert recovered.shape == x.shape

    def test_transpose_correctness(self) -> None:
        """Test that merge_heads_transposed correctly untransposes."""
        batch, n_heads, seq_len, d_head = 2, 3, 4, 5
        # Create array with unique values
        x = jnp.arange(batch * n_heads * seq_len * d_head).reshape(batch, n_heads, seq_len, d_head)

        result = merge_heads_transposed(x)

        # Manually compute expected: transpose then reshape
        expected = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * d_head)
        assert jnp.array_equal(result, expected)


class TestAssertShape:
    """Tests for assert_shape function."""

    def test_correct_shape_no_error(self) -> None:
        """Test that correct shape raises no error."""
        x = jnp.ones((4, 128, 768))
        # Should not raise
        assert_shape(x, (4, 128, 768), "test_tensor")

    def test_wildcard_dimensions(self) -> None:
        """Test that None wildcards allow any dimension."""
        x = jnp.ones((4, 128, 768))
        # Should not raise with wildcards
        assert_shape(x, (None, 128, 768), "test_tensor")
        assert_shape(x, (4, None, 768), "test_tensor")
        assert_shape(x, (None, None, None), "test_tensor")

    def test_wrong_dimension_raises_error(self) -> None:
        """Test that wrong dimension raises ValueError with clear message."""
        x = jnp.ones((4, 128, 768))

        with pytest.raises(ValueError, match="incorrect shape at dimension"):
            assert_shape(x, (4, 128, 512), "test_tensor")

    def test_wrong_number_of_dimensions_raises_error(self) -> None:
        """Test that wrong number of dimensions raises ValueError."""
        x = jnp.ones((4, 128, 768))

        with pytest.raises(ValueError, match="incorrect number of dimensions"):
            assert_shape(x, (4, 128), "test_tensor")

    def test_error_message_includes_tensor_name(self) -> None:
        """Test that error message includes the tensor name."""
        x = jnp.ones((4, 128))

        with pytest.raises(ValueError, match="my_tensor"):
            assert_shape(x, (4, 256), "my_tensor")


class TestShapeStr:
    """Tests for shape_str function."""

    def test_with_name(self) -> None:
        """Test shape_str formatting with name."""
        x = jnp.ones((4, 128, 768), dtype=jnp.float32)
        result = shape_str(x, "query")

        assert "query:" in result
        assert "(4, 128, 768)" in result
        assert "float32" in result

    def test_without_name(self) -> None:
        """Test shape_str formatting without name."""
        x = jnp.ones((4, 128, 768), dtype=jnp.float32)
        result = shape_str(x)

        assert "(4, 128, 768)" in result
        assert "float32" in result

    def test_different_dtypes(self) -> None:
        """Test that shape_str includes dtype information."""
        x_f32 = jnp.ones((4, 128), dtype=jnp.float32)
        x_f16 = jnp.ones((4, 128), dtype=jnp.float16)
        x_int = jnp.ones((4, 128), dtype=jnp.int32)

        assert "float32" in shape_str(x_f32)
        assert "float16" in shape_str(x_f16)
        assert "int32" in shape_str(x_int)


class TestIntegrationScenarios:
    """Integration tests simulating typical attention mechanism usage."""

    def test_attention_flow_non_transposed(self) -> None:
        """Test typical flow: project -> split -> attend -> merge."""
        batch, seq_len, d_model, n_heads = 2, 4, 8, 2
        d_head = d_model // n_heads

        # Simulate Q, K, V projections
        q = jnp.ones((batch, seq_len, d_model))

        # Split heads
        q_heads = split_heads(q, n_heads)
        assert q_heads.shape == (batch, seq_len, n_heads, d_head)

        # Merge back
        q_merged = merge_heads(q_heads)
        assert q_merged.shape == (batch, seq_len, d_model)
        assert jnp.array_equal(q_merged, q)

    def test_attention_flow_transposed(self) -> None:
        """Test typical flow with transposed layout for attention."""
        batch, seq_len, d_model, n_heads = 2, 4, 8, 2
        d_head = d_model // n_heads

        # Simulate Q projection
        q = jnp.ones((batch, seq_len, d_model))

        # Split and transpose for attention
        q_heads = split_heads_transposed(q, n_heads)
        assert q_heads.shape == (batch, n_heads, seq_len, d_head)

        # After attention, merge back
        q_merged = merge_heads_transposed(q_heads)
        assert q_merged.shape == (batch, seq_len, d_model)
        assert jnp.array_equal(q_merged, q)

    def test_shape_validation_in_pipeline(self) -> None:
        """Test using assert_shape to validate intermediate shapes."""
        batch, seq_len, d_model, n_heads = 4, 128, 768, 12
        d_head = d_model // n_heads

        q = jnp.ones((batch, seq_len, d_model))

        # Validate input
        assert_shape(q, (None, None, d_model), "q")

        # Split and validate
        q_heads = split_heads_transposed(q, n_heads)
        assert_shape(q_heads, (None, n_heads, None, d_head), "q_heads")

        # Merge and validate
        q_merged = merge_heads_transposed(q_heads)
        assert_shape(q_merged, (None, None, d_model), "q_merged")
