import torch

from zmaj_lm.config.model_config import TransformerBlockConfig, TransformerConfig
from zmaj_lm.models.gpt import GPTModel


class TestGPTModel:
    """Test suite for GPTModel (complete transformer)."""

    def test_output_shape(self, device: torch.device) -> None:
        """Test that GPT model produces correct output shape (logits over vocabulary)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 64

        model = GPTModel(config=config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))

    def test_causal_masking(self, device: torch.device) -> None:
        """Test that model applies causal masking (future tokens don't affect past)."""
        config = TransformerConfig(
            vocab_size=100,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.0,  # No dropout for deterministic test
        )

        batch, seq_len = 1, 8

        model = GPTModel(config=config).to(device)
        model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Get predictions for full sequence
        with torch.no_grad():
            logits_full = model(input_ids)

        # Get predictions for truncated sequence (first 5 tokens)
        input_ids_truncated = input_ids[:, :5]
        with torch.no_grad():
            logits_truncated = model(input_ids_truncated)

        # Due to causal masking, predictions for first 5 positions should be identical
        # (they don't see the last 3 tokens anyway)
        assert torch.allclose(
            logits_full[:, :5, :], logits_truncated[:, :5, :], rtol=1e-5, atol=1e-5
        )

    def test_parameter_count(self, device: torch.device) -> None:
        """Test total parameter count matches expected for the architecture."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            num_layers=2,
            block_config=TransformerBlockConfig(
                hidden_dim=128,
                num_heads=4,
                mlp_dim=512,
                pos_encoding_type="learned",
                use_bias=True,
                activation="gelu",
            ),
        )

        model = GPTModel(config=config).to(device)

        # Calculate expected parameters
        # Token embeddings: vocab_size × hidden_dim
        token_embed_params = config.vocab_size * config.hidden_dim

        # Positional embeddings (learned): max_seq_len × hidden_dim
        pos_embed_params = config.max_seq_len * config.hidden_dim

        # Per transformer block:
        # - Multi-head attention: 4 projections (Q, K, V, O) each hidden_dim × hidden_dim
        # - FFN: 2 layers (hidden_dim × mlp_dim + mlp_dim × hidden_dim)
        # - LayerNorm: 2 per block (scale + bias for each)
        per_block_params = (
            # Attention: 4 × (hidden_dim × hidden_dim + hidden_dim bias)
            4 * (config.hidden_dim * config.hidden_dim + config.hidden_dim)
            # FFN: (hidden_dim × mlp_dim + mlp_dim) + (mlp_dim × hidden_dim + hidden_dim)
            + (config.hidden_dim * config.mlp_dim + config.mlp_dim)
            + (config.mlp_dim * config.hidden_dim + config.hidden_dim)
            # LayerNorm ×2: 2 × (hidden_dim scale + hidden_dim bias)
            + 2 * (config.hidden_dim + config.hidden_dim)
        )

        # Total for all blocks
        transformer_params = config.num_layers * per_block_params

        # Final LayerNorm: scale + bias
        final_norm_params = config.hidden_dim + config.hidden_dim

        # Total (no separate output projection due to weight tying)
        expected_total = (
            token_embed_params + pos_embed_params + transformer_params + final_norm_params
        )

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == expected_total

    def test_deterministic_vs_training_mode(self, device: torch.device) -> None:
        """Test that deterministic mode is consistent and training mode varies due to dropout."""
        config = TransformerConfig(
            vocab_size=500,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 32

        model = GPTModel(config=config).to(device)
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Eval mode should give same output
        model.eval()
        with torch.no_grad():
            logits1 = model(input_ids)
            logits2 = model(input_ids)
        assert torch.allclose(logits1, logits2)

        # Training mode with dropout should give different outputs
        model.train()
        torch.manual_seed(42)
        logits3 = model(input_ids)
        torch.manual_seed(43)
        logits4 = model(input_ids)
        assert not torch.allclose(logits3, logits4)

    def test_variable_sequence_lengths(self, device: torch.device) -> None:
        """Test that model handles different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch = 2

        model = GPTModel(config=config).to(device)
        model.eval()

        # Test various sequence lengths
        for seq_len in [8, 16, 32, 64, 128, 256]:
            input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_different_configs(self, device: torch.device) -> None:
        """Test that model works with different configuration options."""
        configs = [
            # Small model with sinusoidal positions
            TransformerConfig(
                vocab_size=500,
                max_seq_len=128,
                hidden_dim=64,
                num_layers=2,
                num_heads=2,
                pos_encoding_type="sinusoidal",
            ),
            # Larger model with learned positions
            TransformerConfig(
                vocab_size=2000,
                max_seq_len=256,
                hidden_dim=256,
                num_layers=4,
                num_heads=8,
                pos_encoding_type="learned",
            ),
        ]

        batch, seq_len = 2, 32

        for config in configs:
            model = GPTModel(config=config).to(device)
            model.eval()

            input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

            with torch.no_grad():
                logits = model(input_ids)

            assert logits.shape == (batch, seq_len, config.vocab_size)
            assert not torch.any(torch.isnan(logits))

    def test_gradient_flow(self, device: torch.device) -> None:
        """Test that gradients flow through the entire model."""
        config = TransformerConfig(
            vocab_size=100,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.0,  # Disable dropout for gradient test
        )

        batch, seq_len = 2, 16

        model = GPTModel(config=config).to(device)
        model.train()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Forward pass
        logits = model(input_ids)

        # Simple loss function (mean squared error)
        loss = (logits**2).mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
            assert not torch.any(torch.isnan(param.grad)), f"NaN gradient for {name}"

        assert not torch.isnan(loss)

    def test_compilation(self, device: torch.device) -> None:
        """Test that model can be compiled with torch.compile."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch, seq_len = 2, 32

        model = GPTModel(config=config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Compile the model
        compiled_model = torch.compile(model)

        with torch.no_grad():
            logits_compiled = compiled_model(input_ids)
            logits_regular = model(input_ids)

        assert torch.allclose(logits_compiled, logits_regular, rtol=1e-5, atol=1e-5)

    def test_batch_size_one(self, device: torch.device) -> None:
        """Test that model works with batch size of 1."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch, seq_len = 1, 32

        model = GPTModel(config=config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not torch.any(torch.isnan(logits))

    def test_positional_encoding_integration(self, device: torch.device) -> None:
        """Test that different positional encodings are properly integrated."""
        batch, seq_len = 2, 32

        for pos_type in ["learned", "sinusoidal"]:
            config = TransformerConfig(
                vocab_size=500,
                max_seq_len=128,
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                pos_encoding_type=pos_type,
                dropout_rate=0.0,
            )

            model = GPTModel(config=config).to(device)
            model.eval()

            # Same tokens at different positions should give different predictions
            input_ids = torch.full((batch, seq_len), fill_value=5, device=device)

            with torch.no_grad():
                logits = model(input_ids)

            # First and last position should have different predictions
            # (even though input tokens are the same)
            first_pos_logits = logits[0, 0, :]
            last_pos_logits = logits[0, -1, :]

            assert not torch.allclose(first_pos_logits, last_pos_logits, rtol=0.1)

    def test_rmsnorm(self, device: torch.device) -> None:
        """Test that GPT model works with RMSNorm."""
        from zmaj_lm.config.model_config import TransformerBlockConfig

        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            num_layers=2,
            block_config=TransformerBlockConfig(
                hidden_dim=128,
                num_heads=4,
                mlp_dim=512,
                norm_type="rmsnorm",
            ),
        )

        batch, seq_len = 2, 32

        model = GPTModel(config=config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))

        # Verify that RMSNorm is used in transformer blocks and final norm
        assert isinstance(model.transformer_blocks[0].layernorm_1, torch.nn.RMSNorm)
        assert isinstance(model.transformer_blocks[0].layernorm_2, torch.nn.RMSNorm)
        assert isinstance(model.final_layernorm, torch.nn.RMSNorm)
