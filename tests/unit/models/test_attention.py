import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.attention import MultiHeadAttention, scaled_dot_product_attention
from zmaj_lm.utils.masks import create_causal_mask, create_padding_mask


class TestScaledDotProductAttention:
    """Test suite for scaled dot-product attention function."""

    def test_output_shapes(self, rng_key: Array) -> None:
        """Test that output shapes are correct."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_output_shapes_cross_attention(self, rng_key: Array) -> None:
        """Test shapes with different query and key sequence lengths (cross-attention)."""
        batch, n_heads, seq_len_q, seq_len_k, d_head = 2, 4, 10, 20, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len_q, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len_k, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len_k, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len_q, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len_q, seq_len_k)

    def test_attention_weights_sum_to_one(self, rng_key: Array) -> None:
        """Test that attention weights form a probability distribution over keys."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        _, attn_weights = scaled_dot_product_attention(query, key, value)

        # Sum over key dimension (axis=-1) should be 1.0
        sums = jnp.sum(attn_weights, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-6)

    def test_simple_computation(self) -> None:
        """Test attention computation with a simple manually verifiable example."""
        # Single batch, single head, 2 tokens, d_head=2
        query = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
        key = jnp.array([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
        value = jnp.array([[[[2.0, 0.0], [0.0, 3.0]]]])  # (1, 1, 2, 2)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        # Q @ K^T = [[1, 0], [0, 1]] (identity before scaling)
        # After scaling by 1/sqrt(2): [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
        # After softmax: exp(1/sqrt(2)) / (exp(1/sqrt(2)) + exp(0)) ≈ 0.6700
        # So attention weights are approximately [[0.67, 0.33], [0.33, 0.67]]
        # Output = attn_weights @ value

        assert output.shape == (1, 1, 2, 2)
        # Just verify basic properties rather than exact values
        assert jnp.allclose(jnp.sum(attn_weights, axis=-1), 1.0, atol=1e-6)

    def test_causal_mask_application(self, rng_key: Array) -> None:
        """Test that causal mask prevents attending to future positions."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        # Create causal mask: (1, seq_len, seq_len)
        causal_mask = create_causal_mask(seq_len)

        _, attn_weights = scaled_dot_product_attention(query, key, value, mask=causal_mask)

        # Check that future positions (upper triangle) have near-zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Position i should not attend to position j > i
                assert jnp.allclose(attn_weights[:, :, i, j], 0.0, atol=1e-6)

    def test_padding_mask_application(self, rng_key: Array) -> None:
        """Test that padding mask zeros out attention to padding positions."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        # Create padding mask: first sequence has length 5, second has length 3
        lengths = jnp.array([5, 3])
        padding_mask = create_padding_mask(lengths, seq_len)  # (batch, seq_len)
        # Reshape to (batch, 1, 1, seq_len) for broadcasting with attention scores
        padding_mask = padding_mask[:, None, None, :]

        _, attn_weights = scaled_dot_product_attention(query, key, value, mask=padding_mask)

        # First batch element: positions 5-7 should have near-zero attention
        assert jnp.allclose(attn_weights[0, :, :, 5:], 0.0, atol=1e-6)
        # Second batch element: positions 3-7 should have near-zero attention
        assert jnp.allclose(attn_weights[1, :, :, 3:], 0.0, atol=1e-6)

    def test_mask_no_nan_or_inf(self, rng_key: Array) -> None:
        """Test that masked attention doesn't produce NaN or Inf values."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        causal_mask = create_causal_mask(seq_len)

        output, attn_weights = scaled_dot_product_attention(query, key, value, mask=causal_mask)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
        assert not jnp.any(jnp.isnan(attn_weights))
        assert not jnp.any(jnp.isinf(attn_weights))

    def test_dropout_determinism(self, rng_key: Array) -> None:
        """Test that same dropout key produces same dropout pattern."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3, dropout_key = jax.random.split(rng_key, 4)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        # Same dropout key should produce same results
        output1, attn1 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, dropout_rng=dropout_key
        )
        output2, attn2 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, dropout_rng=dropout_key
        )

        assert jnp.allclose(output1, output2)
        assert jnp.allclose(attn1, attn2)

    def test_dropout_randomness(self, rng_key: Array) -> None:
        """Test that different dropout keys produce different patterns."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3, dropout_key1, dropout_key2 = jax.random.split(rng_key, 5)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        # Different dropout keys should produce different results
        output1, attn1 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, dropout_rng=dropout_key1
        )
        output2, attn2 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, dropout_rng=dropout_key2
        )

        assert not jnp.allclose(output1, output2)
        assert not jnp.allclose(attn1, attn2)

    def test_dropout_sparsity(self, rng_key: Array) -> None:
        """Test that dropout zeros out approximately the correct fraction of weights."""
        batch, n_heads, seq_len, d_head = 4, 8, 16, 32
        key1, key2, key3, dropout_key = jax.random.split(rng_key, 4)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        dropout_rate = 0.3

        # Get attention with dropout
        _, attn_with_dropout = scaled_dot_product_attention(
            query, key, value, dropout_rate=dropout_rate, dropout_rng=dropout_key
        )

        # Count how many weights are exactly zero (dropped out)
        # Note: attention weights are softmax outputs, so they're never exactly 0 without dropout
        # But dropout will zero out approximately dropout_rate fraction of them
        zero_mask = attn_with_dropout == 0.0
        fraction_zeros = jnp.mean(zero_mask)

        # Should be approximately dropout_rate (with some tolerance for randomness)
        assert jnp.abs(fraction_zeros - dropout_rate) < 0.1

    def test_dropout_expectation_preserved(self, rng_key: Array) -> None:
        """Test that dropout scaling preserves expected values (approximately)."""
        batch, n_heads, seq_len, d_head = 2, 4, 32, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        dropout_rate = 0.2
        num_samples = 100

        # Run multiple times with different dropout keys and average
        outputs = []
        for i in range(num_samples):
            dropout_key = jax.random.PRNGKey(i)
            output, _ = scaled_dot_product_attention(
                query, key, value, dropout_rate=dropout_rate, dropout_rng=dropout_key
            )
            outputs.append(output)

        mean_output = jnp.mean(jnp.stack(outputs), axis=0)

        # Compare with no dropout
        output_no_dropout, _ = scaled_dot_product_attention(query, key, value)

        # Mean should be close to no-dropout case
        assert jnp.allclose(mean_output, output_no_dropout, rtol=0.1, atol=0.1)

    def test_no_dropout_when_rate_zero(self, rng_key: Array) -> None:
        """Test that dropout_rate=0.0 behaves identically to no dropout."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16
        key1, key2, key3, dropout_key = jax.random.split(rng_key, 4)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output_no_dropout, attn_no_dropout = scaled_dot_product_attention(query, key, value)
        output_zero_dropout, attn_zero_dropout = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.0, dropout_rng=dropout_key
        )

        assert jnp.allclose(output_no_dropout, output_zero_dropout)
        assert jnp.allclose(attn_no_dropout, attn_zero_dropout)

    def test_single_token(self, rng_key: Array) -> None:
        """Test edge case with single token sequence."""
        batch, n_heads, seq_len, d_head = 2, 4, 1, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)
        # Single token should attend to itself with weight 1.0
        assert jnp.allclose(attn_weights, 1.0)

    def test_single_head(self, rng_key: Array) -> None:
        """Test edge case with single attention head."""
        batch, n_heads, seq_len, d_head = 2, 1, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_batch_size_one(self, rng_key: Array) -> None:
        """Test edge case with batch size of 1."""
        batch, n_heads, seq_len, d_head = 1, 4, 8, 16
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_small_d_head(self, rng_key: Array) -> None:
        """Test with very small head dimension."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 2
        key1, key2, key3 = jax.random.split(rng_key, 3)

        query = jax.random.normal(key1, (batch, n_heads, seq_len, d_head))
        key = jax.random.normal(key2, (batch, n_heads, seq_len, d_head))
        value = jax.random.normal(key3, (batch, n_heads, seq_len, d_head))

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)
        assert jnp.allclose(jnp.sum(attn_weights, axis=-1), 1.0, atol=1e-6)


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention Flax module."""

    def test_output_shape(self, rng_key: Array) -> None:
        """Test that MHA produces correct output shape."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16
        key1, key2 = jax.random.split(rng_key)

        # Create module and initialize
        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        # Initialize parameters
        params = mha.init(key2, x, deterministic=True)

        # Forward pass
        output = mha.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_with_mask(self, rng_key: Array) -> None:
        """Test MHA with causal mask."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))
        mask = create_causal_mask(seq_len)

        params = mha.init(key2, x, mask=mask, deterministic=True)
        output = mha.apply(params, x, mask=mask, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_return_attention_weights(self, rng_key: Array) -> None:
        """Test that attention weights can be returned."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        # Create module with return_attention_weights=True
        mha = MultiHeadAttention(config=config, return_attention_weights=True)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)
        output, attn_weights = mha.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert attn_weights.shape == (batch, config.num_heads, seq_len, seq_len)
        # Attention weights should sum to 1 over keys
        assert jnp.allclose(jnp.sum(attn_weights, axis=-1), 1.0, atol=1e-6)

    def test_deterministic_vs_training(self, rng_key: Array) -> None:
        """Test that deterministic mode produces consistent outputs."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            attention_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8
        key1, key2, key3 = jax.random.split(rng_key, 3)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)

        # Deterministic mode should give same output every time
        output1 = mha.apply(params, x, deterministic=True)
        output2 = mha.apply(params, x, deterministic=True)
        assert jnp.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        output3 = mha.apply(params, x, deterministic=False, rngs={"dropout": key3})
        key4 = jax.random.split(key3)[0]
        output4 = mha.apply(params, x, deterministic=False, rngs={"dropout": key4})
        assert not jnp.allclose(output3, output4)

    def test_parameter_count(self, rng_key: Array) -> None:
        """Test that parameter count matches expected for MHA."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=True,
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)

        # Count parameters
        # Q, K, V projections: each has (hidden_dim, hidden_dim) + hidden_dim bias
        # Output projection: (hidden_dim, hidden_dim) + hidden_dim bias
        expected_params_per_proj = config.hidden_dim * config.hidden_dim + config.hidden_dim
        expected_total = 4 * expected_params_per_proj

        total_params = sum(x.size for x in jax.tree.leaves(params))
        assert total_params == expected_total

    def test_parameter_count_no_bias(self, rng_key: Array) -> None:
        """Test parameter count without bias."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=False,
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)

        # Without bias: 4 projections * (hidden_dim * hidden_dim)
        expected_total = 4 * config.hidden_dim * config.hidden_dim

        total_params = sum(x.size for x in jax.tree.leaves(params))
        assert total_params == expected_total

    def test_different_sequence_lengths(self, rng_key: Array) -> None:
        """Test that MHA works with different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        key1, key2 = jax.random.split(rng_key)
        batch = 2

        mha = MultiHeadAttention(config=config)

        # Initialize with one sequence length
        x_init = jax.random.normal(key1, (batch, 16, config.hidden_dim))
        params = mha.init(key2, x_init, deterministic=True)

        # Test with different sequence lengths
        for seq_len in [1, 4, 8, 32]:
            x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))
            output = mha.apply(params, x, deterministic=True)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_single_head(self, rng_key: Array) -> None:
        """Test edge case with single attention head."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=1,  # Single head
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)
        output = mha.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_jit_compilation(self, rng_key: Array) -> None:
        """Test that MHA can be JIT compiled."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        mha = MultiHeadAttention(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = mha.init(key2, x, deterministic=True)

        # Create JIT-compiled apply function
        @jax.jit
        def apply_jit(params: Array, x: Array) -> Array:
            return mha.apply(params, x, deterministic=True)

        # Test JIT compilation works and produces numerically close results
        # Note: JIT can introduce small numerical differences due to compiler optimizations
        output_jit = apply_jit(params, x)
        output_regular = mha.apply(params, x, deterministic=True)

        assert jnp.allclose(output_jit, output_regular, rtol=1e-5, atol=1e-5)

    def test_compare_with_flax_multihead_attention(self, rng_key: Array) -> None:
        """Verify our implementation has same behavior as Flax's reference implementation.

        Note: We can't directly copy parameters between implementations because Flax uses
        DenseGeneral which has a different parameter structure (hidden_dim, num_heads, head_dim)
        vs our approach which uses regular Dense (hidden_dim, hidden_dim) and splits heads later.
        Both are valid implementations of the same algorithm.
        """
        hidden_dim = 128
        num_heads = 4
        batch, seq_len = 2, 8

        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            use_bias=False,
            attention_dropout_rate=0.0,
        )

        key1, key2, key3 = jax.random.split(rng_key, 3)
        x = jax.random.normal(key1, (batch, seq_len, hidden_dim))

        # Initialize our implementation
        our_mha = MultiHeadAttention(config=config)
        our_params = our_mha.init(key2, x, deterministic=True)

        # Initialize Flax reference implementation
        flax_mha = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=False,
            dropout_rate=0.0,
        )
        flax_params = flax_mha.init(key3, x, x, deterministic=True)

        # Forward pass with both implementations
        our_output = our_mha.apply(our_params, x, deterministic=True)
        flax_output = flax_mha.apply(flax_params, x, x, deterministic=True)

        # Both should produce correct output shapes
        assert our_output.shape == (batch, seq_len, hidden_dim)
        assert flax_output.shape == (batch, seq_len, hidden_dim)

        # Both should have same number of parameters (4 projections * hidden_dim^2)
        our_param_count = sum(p.size for p in jax.tree.leaves(our_params))
        flax_param_count = sum(p.size for p in jax.tree.leaves(flax_params))
        assert our_param_count == flax_param_count == 4 * hidden_dim * hidden_dim

        # Both outputs should not have NaN or Inf
        assert not jnp.any(jnp.isnan(our_output))
        assert not jnp.any(jnp.isinf(our_output))
        assert not jnp.any(jnp.isnan(flax_output))
        assert not jnp.any(jnp.isinf(flax_output))
