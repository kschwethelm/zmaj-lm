import torch

from zmaj_lm.config.model_config import TransformerBlockConfig
from zmaj_lm.models.attention import MultiHeadAttention
from zmaj_lm.utils.masks import create_causal_mask


class TestMultiHeadAttention:
    """Test suite for MultiHeadAttention module."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that MHA produces correct output shape."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            mlp_dim=1024,
        )

        batch, seq_len = 2, 16

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)

    def test_with_mask(self, device: torch.device) -> None:
        """Test MHA with causal mask."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)
        mask = create_causal_mask(seq_len, device=device)

        with torch.no_grad():
            output = mha(x, mask=mask)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_deterministic_vs_training(self, device: torch.device) -> None:
        """Test that eval mode produces consistent outputs."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            attention_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=block_config).to(device)
        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

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
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            use_bias=True,
        )

        mha = MultiHeadAttention(config=block_config).to(device)

        # Q projection: (hidden_dim, hidden_dim) + hidden_dim bias
        q_params = block_config.hidden_dim * block_config.hidden_dim + block_config.hidden_dim
        # K, V projections: with MHA (num_kv_heads == num_heads), same as Q
        # each has (hidden_dim, num_kv_heads * head_dim) + num_kv_heads * head_dim bias
        kv_dim = block_config.num_kv_heads * block_config.head_dim  # equals hidden_dim for MHA
        kv_params_each = block_config.hidden_dim * kv_dim + kv_dim
        # Output projection: (hidden_dim, hidden_dim) + hidden_dim bias
        out_params = block_config.hidden_dim * block_config.hidden_dim + block_config.hidden_dim

        expected_total = q_params + 2 * kv_params_each + out_params

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_parameter_count_no_bias(self, device: torch.device) -> None:
        """Test parameter count without bias."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            use_bias=False,
        )

        mha = MultiHeadAttention(config=block_config).to(device)

        # Q projection: hidden_dim * hidden_dim
        q_params = block_config.hidden_dim * block_config.hidden_dim
        # K, V projections: with MHA, each has hidden_dim * (num_kv_heads * head_dim)
        kv_dim = block_config.num_kv_heads * block_config.head_dim  # equals hidden_dim for MHA
        kv_params_each = block_config.hidden_dim * kv_dim
        # Output projection: hidden_dim * hidden_dim
        out_params = block_config.hidden_dim * block_config.hidden_dim

        expected_total = q_params + 2 * kv_params_each + out_params

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_different_sequence_lengths(self, device: torch.device) -> None:
        """Test that MHA works with different sequence lengths."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
        )

        batch = 2

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        # Test with different sequence lengths
        for seq_len in [1, 4, 8, 32]:
            x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)
            with torch.no_grad():
                output = mha(x)
            assert output.shape == (batch, seq_len, block_config.hidden_dim)

    def test_single_head(self, device: torch.device) -> None:
        """Test edge case with single attention head."""
        block_config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=1,  # Single head
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)

    def test_compilation(self, device: torch.device) -> None:
        """Test that MHA can be compiled with torch.compile."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        # Compile the module
        compiled_mha = torch.compile(mha)

        with torch.no_grad():
            output_compiled = compiled_mha(x)
            output_regular = mha(x)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)


class TestGroupedQueryAttention:
    """Test suite for Grouped Query Attention (GQA) functionality."""

    def test_gqa_output_shape(self, device: torch.device) -> None:
        """Test that GQA produces correct output shape."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,  # GQA with 4 query heads per KV head
        )

        batch, seq_len = 2, 16

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)

    def test_mqa_output_shape(self, device: torch.device) -> None:
        """Test Multi-Query Attention (MQA) with single KV head."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=1,  # MQA: all query heads share single KV head
        )

        batch, seq_len = 2, 16

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

        with torch.no_grad():
            output = mha(x)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)

    def test_gqa_parameter_count(self, device: torch.device) -> None:
        """Test that GQA has correct reduced parameter count."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
            use_bias=True,
        )

        mha = MultiHeadAttention(config=block_config).to(device)

        # Q projection: (hidden_dim, hidden_dim) + hidden_dim bias
        q_params = block_config.hidden_dim * block_config.hidden_dim + block_config.hidden_dim
        # K, V projections: each has (hidden_dim, num_kv_heads * head_dim) + num_kv_heads * head_dim bias
        kv_dim = block_config.num_kv_heads * block_config.head_dim
        kv_params_each = block_config.hidden_dim * kv_dim + kv_dim
        # Output projection: (hidden_dim, hidden_dim) + hidden_dim bias
        out_params = block_config.hidden_dim * block_config.hidden_dim + block_config.hidden_dim

        expected_total = q_params + 2 * kv_params_each + out_params

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_mqa_parameter_count(self, device: torch.device) -> None:
        """Test that MQA has minimal parameter count."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=1,  # MQA
            use_bias=False,
        )

        mha = MultiHeadAttention(config=block_config).to(device)

        # Q projection: hidden_dim * hidden_dim
        q_params = block_config.hidden_dim * block_config.hidden_dim
        # K, V projections: each has hidden_dim * head_dim (only 1 head)
        kv_params_each = block_config.hidden_dim * block_config.head_dim
        # Output projection: hidden_dim * hidden_dim
        out_params = block_config.hidden_dim * block_config.hidden_dim

        expected_total = q_params + 2 * kv_params_each + out_params

        total_params = sum(p.numel() for p in mha.parameters())
        assert total_params == expected_total

    def test_gqa_different_ratios(self, device: torch.device) -> None:
        """Test GQA with various query-to-KV head ratios."""
        batch, seq_len = 2, 8

        # Test different valid GQA configurations
        configs = [
            (8, 4),  # 2 query heads per KV head
            (8, 2),  # 4 query heads per KV head
            (16, 4),  # 4 query heads per KV head
            (12, 3),  # 4 query heads per KV head
        ]

        for num_heads, num_kv_heads in configs:
            block_config = TransformerBlockConfig(
                hidden_dim=num_heads * 32,  # Ensure divisibility
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

            mha = MultiHeadAttention(config=block_config).to(device)
            mha.eval()

            x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)

            with torch.no_grad():
                output = mha(x)

            assert output.shape == (batch, seq_len, block_config.hidden_dim)
            assert not torch.any(torch.isnan(output))

    def test_gqa_vs_mha_different_outputs(self, device: torch.device) -> None:
        """Test that GQA and MHA produce different outputs with same initialization."""
        batch, seq_len = 2, 8
        hidden_dim = 128
        num_heads = 8

        # Create input
        x = torch.randn(batch, seq_len, hidden_dim, device=device)

        # MHA configuration
        block_config_mha = TransformerBlockConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # MHA
        )

        # GQA configuration
        block_config_gqa = TransformerBlockConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=2,  # GQA
        )

        mha = MultiHeadAttention(config=block_config_mha).to(device)
        gqa = MultiHeadAttention(config=block_config_gqa).to(device)

        mha.eval()
        gqa.eval()

        with torch.no_grad():
            output_mha = mha(x)
            output_gqa = gqa(x)

        # Outputs should have same shape but different values
        assert output_mha.shape == output_gqa.shape
        assert not torch.allclose(output_mha, output_gqa)

    def test_config_default_num_kv_heads(self) -> None:
        """Test that num_kv_heads defaults to num_heads for backward compatibility."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            # num_kv_heads not specified
        )

        assert block_config.num_kv_heads == block_config.num_heads
        assert block_config.num_kv_groups == 1

    def test_config_num_kv_groups(self) -> None:
        """Test that num_kv_groups is computed correctly."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            num_kv_heads=2,
        )

        assert block_config.num_kv_groups == 4  # 8 / 2

    def test_gqa_with_causal_mask(self, device: torch.device) -> None:
        """Test that GQA works correctly with causal masking."""
        block_config = TransformerBlockConfig(
            hidden_dim=128,
            num_heads=4,
            num_kv_heads=2,
        )

        batch, seq_len = 2, 8

        mha = MultiHeadAttention(config=block_config).to(device)
        mha.eval()

        x = torch.randn(batch, seq_len, block_config.hidden_dim, device=device)
        mask = create_causal_mask(seq_len, device=device)

        with torch.no_grad():
            output = mha(x, mask=mask)

        assert output.shape == (batch, seq_len, block_config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
