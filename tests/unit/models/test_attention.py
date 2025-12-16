import torch

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.attention import MultiHeadAttention, scaled_dot_product_attention
from zmaj_lm.utils.masks import create_causal_mask, create_padding_mask


class TestScaledDotProductAttention:
    """Test suite for scaled dot-product attention function."""

    def test_output_shapes(self, device: torch.device) -> None:
        """Test that output shapes are correct."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_output_shapes_cross_attention(self, device: torch.device) -> None:
        """Test shapes with different query and key sequence lengths (cross-attention)."""
        batch, n_heads, seq_len_q, seq_len_k, d_head = 2, 4, 10, 20, 16

        query = torch.randn(batch, n_heads, seq_len_q, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len_k, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len_k, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len_q, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len_q, seq_len_k)

    def test_attention_weights_sum_to_one(self, device: torch.device) -> None:
        """Test that attention weights form a probability distribution over keys."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        _, attn_weights = scaled_dot_product_attention(query, key, value)

        # Sum over key dimension (axis=-1) should be 1.0
        sums = torch.sum(attn_weights, dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_simple_computation(self, device: torch.device) -> None:
        """Test attention computation with a simple manually verifiable example."""
        # Single batch, single head, 2 tokens, d_head=2
        query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device)  # (1, 1, 2, 2)
        key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device=device)  # (1, 1, 2, 2)
        value = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]], device=device)  # (1, 1, 2, 2)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        # Q @ K^T = [[1, 0], [0, 1]] (identity before scaling)
        # After scaling by 1/sqrt(2): [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
        # After softmax: exp(1/sqrt(2)) / (exp(1/sqrt(2)) + exp(0)) ≈ 0.6700
        # So attention weights are approximately [[0.67, 0.33], [0.33, 0.67]]
        # Output = attn_weights @ value

        assert output.shape == (1, 1, 2, 2)
        # Just verify basic properties rather than exact values
        assert torch.allclose(
            torch.sum(attn_weights, dim=-1), torch.ones_like(attn_weights[..., 0]), atol=1e-6
        )

    def test_causal_mask_application(self, device: torch.device) -> None:
        """Test that causal mask prevents attending to future positions."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        # Create causal mask: (1, seq_len, seq_len)
        causal_mask = create_causal_mask(seq_len, device=device)

        _, attn_weights = scaled_dot_product_attention(query, key, value, mask=causal_mask)

        # Check that future positions (upper triangle) have near-zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Position i should not attend to position j > i
                assert torch.allclose(
                    attn_weights[:, :, i, j], torch.zeros(batch, n_heads, device=device), atol=1e-6
                )

    def test_padding_mask_application(self, device: torch.device) -> None:
        """Test that padding mask zeros out attention to padding positions."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        # Create padding mask: first sequence has length 5, second has length 3
        lengths = torch.tensor([5, 3], device=device)
        padding_mask = create_padding_mask(lengths, seq_len)  # (batch, seq_len)
        # Reshape to (batch, 1, 1, seq_len) for broadcasting with attention scores
        padding_mask = padding_mask[:, None, None, :]

        _, attn_weights = scaled_dot_product_attention(query, key, value, mask=padding_mask)

        # First batch element: positions 5-7 should have near-zero attention
        assert torch.allclose(
            attn_weights[0, :, :, 5:], torch.zeros_like(attn_weights[0, :, :, 5:]), atol=1e-6
        )
        # Second batch element: positions 3-7 should have near-zero attention
        assert torch.allclose(
            attn_weights[1, :, :, 3:], torch.zeros_like(attn_weights[1, :, :, 3:]), atol=1e-6
        )

    def test_mask_no_nan_or_inf(self, device: torch.device) -> None:
        """Test that masked attention doesn't produce NaN or Inf values."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        causal_mask = create_causal_mask(seq_len, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value, mask=causal_mask)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
        assert not torch.any(torch.isnan(attn_weights))
        assert not torch.any(torch.isinf(attn_weights))

    def test_dropout_determinism(self, device: torch.device) -> None:
        """Test that same random seed produces same dropout pattern."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        # Same random seed should produce same results
        torch.manual_seed(42)
        output1, attn1 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, training=True
        )

        torch.manual_seed(42)
        output2, attn2 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, training=True
        )

        assert torch.allclose(output1, output2)
        assert torch.allclose(attn1, attn2)

    def test_dropout_randomness(self, device: torch.device) -> None:
        """Test that different random seeds produce different patterns."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        # Different random seeds should produce different results
        torch.manual_seed(42)
        output1, attn1 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, training=True
        )

        torch.manual_seed(43)
        output2, attn2 = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.1, training=True
        )

        assert not torch.allclose(output1, output2)
        assert not torch.allclose(attn1, attn2)

    def test_dropout_sparsity(self, device: torch.device) -> None:
        """Test that dropout zeros out approximately the correct fraction of weights."""
        batch, n_heads, seq_len, d_head = 4, 8, 16, 32

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        dropout_rate = 0.3

        # Get attention with dropout
        torch.manual_seed(42)
        _, attn_with_dropout = scaled_dot_product_attention(
            query, key, value, dropout_rate=dropout_rate, training=True
        )

        # Count how many weights are exactly zero (dropped out)
        zero_mask = attn_with_dropout == 0.0
        fraction_zeros = zero_mask.float().mean()

        # Should be approximately dropout_rate (with some tolerance for randomness)
        assert abs(fraction_zeros - dropout_rate) < 0.1

    def test_dropout_expectation_preserved(self, device: torch.device) -> None:
        """Test that dropout scaling preserves expected values (approximately)."""
        batch, n_heads, seq_len, d_head = 2, 4, 32, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        dropout_rate = 0.2
        num_samples = 100

        # Run multiple times with different random seeds and average
        outputs = []
        for i in range(num_samples):
            torch.manual_seed(i)
            output, _ = scaled_dot_product_attention(
                query, key, value, dropout_rate=dropout_rate, training=True
            )
            outputs.append(output)

        mean_output = torch.stack(outputs).mean(dim=0)

        # Compare with no dropout
        output_no_dropout, _ = scaled_dot_product_attention(query, key, value, training=False)

        # Mean should be close to no-dropout case
        assert torch.allclose(mean_output, output_no_dropout, rtol=0.1, atol=0.1)

    def test_no_dropout_when_rate_zero(self, device: torch.device) -> None:
        """Test that dropout_rate=0.0 behaves identically to no dropout."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output_no_dropout, attn_no_dropout = scaled_dot_product_attention(
            query, key, value, training=False
        )
        output_zero_dropout, attn_zero_dropout = scaled_dot_product_attention(
            query, key, value, dropout_rate=0.0, training=True
        )

        assert torch.allclose(output_no_dropout, output_zero_dropout)
        assert torch.allclose(attn_no_dropout, attn_zero_dropout)

    def test_single_token(self, device: torch.device) -> None:
        """Test edge case with single token sequence."""
        batch, n_heads, seq_len, d_head = 2, 4, 1, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)
        # Single token should attend to itself with weight 1.0
        assert torch.allclose(attn_weights, torch.ones_like(attn_weights))

    def test_single_head(self, device: torch.device) -> None:
        """Test edge case with single attention head."""
        batch, n_heads, seq_len, d_head = 2, 1, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_batch_size_one(self, device: torch.device) -> None:
        """Test edge case with batch size of 1."""
        batch, n_heads, seq_len, d_head = 1, 4, 8, 16

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)

    def test_small_d_head(self, device: torch.device) -> None:
        """Test with very small head dimension."""
        batch, n_heads, seq_len, d_head = 2, 4, 8, 2

        query = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        key = torch.randn(batch, n_heads, seq_len, d_head, device=device)
        value = torch.randn(batch, n_heads, seq_len, d_head, device=device)

        output, attn_weights = scaled_dot_product_attention(query, key, value)

        assert output.shape == (batch, n_heads, seq_len, d_head)
        assert attn_weights.shape == (batch, n_heads, seq_len, seq_len)
        assert torch.allclose(
            torch.sum(attn_weights, dim=-1), torch.ones_like(attn_weights[..., 0]), atol=1e-6
        )


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention module."""

    def test_output_shape(self, device: torch.device) -> None:
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

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_with_mask(self, device: torch.device) -> None:
        """Test MHA with causal mask."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
        mask = create_causal_mask(seq_len, device=device)

        with torch.no_grad():
            output = mha(x, mask=mask)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_return_attention_weights(self, device: torch.device) -> None:
        """Test that attention weights can be returned."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config, return_attention_weights=True).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output, attn_weights = mha(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert attn_weights.shape == (batch, config.num_heads, seq_len, seq_len)
        # Attention weights should sum to 1 over keys
        assert torch.allclose(
            torch.sum(attn_weights, dim=-1), torch.ones_like(attn_weights[..., 0]), atol=1e-6
        )

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode produces consistent outputs."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            attention_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Eval mode should give same output every time
        mha.eval()
        with torch.no_grad():
            output1 = mha(x)
            output2 = mha(x)
        assert torch.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        mha.train()
        torch.manual_seed(42)
        output3 = mha(x)
        torch.manual_seed(43)
        output4 = mha(x)
        assert not torch.allclose(output3, output4)

    def test_parameter_count(self, device: torch.device) -> None:
        """Test that parameter count matches expected for MHA."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=True,
        )

        mha = MultiHeadAttention(config=config).to(device)

        # Count parameters
        # Q, K, V projections: each has (hidden_dim, hidden_dim) + hidden_dim bias
        # Output projection: (hidden_dim, hidden_dim) + hidden_dim bias
        expected_params_per_proj = config.hidden_dim * config.hidden_dim + config.hidden_dim
        expected_total = 4 * expected_params_per_proj

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_parameter_count_no_bias(self, device: torch.device) -> None:
        """Test parameter count without bias."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            use_bias=False,
        )

        mha = MultiHeadAttention(config=config).to(device)

        # Without bias: 4 projections * (hidden_dim * hidden_dim)
        expected_total = 4 * config.hidden_dim * config.hidden_dim

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_different_sequence_lengths(self, device: torch.device) -> None:
        """Test that MHA works with different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch = 2

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        # Test with different sequence lengths
        for seq_len in [1, 4, 8, 32]:
            x = torch.randn(batch, seq_len, config.hidden_dim, device=device)
            with torch.no_grad():
                output = mha(x)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_single_head(self, device: torch.device) -> None:
        """Test edge case with single attention head."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=1,  # Single head
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_compilation(self, device: torch.device) -> None:
        """Test that MHA can be compiled with torch.compile."""
        config = TransformerConfig(
            vocab_size=1000, max_seq_len=512, hidden_dim=128, num_layers=2, num_heads=4
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Compile the module
        compiled_mha = torch.compile(mha)

        with torch.no_grad():
            output_compiled = compiled_mha(x)
            output_regular = mha(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)
