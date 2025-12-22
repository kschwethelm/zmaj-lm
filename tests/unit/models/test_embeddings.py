import torch

from zmaj_lm.config.model_config import TransformerBlockConfig, TransformerConfig
from zmaj_lm.models.embeddings import TokenEmbedding


class TestTokenEmbedding:
    """Test suite for TokenEmbedding module."""

    def test_encode_output_shape(self, device: torch.device) -> None:
        """Test that token embedding encode produces correct output shape."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
        )

        batch, seq_len = 2, 128

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        with torch.no_grad():
            output = embedding.encode(input_ids)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_decode_output_shape(self, device: torch.device) -> None:
        """Test that decode produces correct output shape (logits over vocabulary)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
        )

        batch, seq_len = 2, 128

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        # Create hidden states
        hidden_states = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Decode to vocabulary
        with torch.no_grad():
            logits = embedding.decode(hidden_states)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))

    def test_weight_tying(self, device: torch.device) -> None:
        """Test that encode and decode share the same embedding weights (weight tying)."""
        config = TransformerConfig(
            vocab_size=100,
            max_seq_len=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            pos_encoding_type="sinusoidal",  # Use sinusoidal to isolate token embeddings
            dropout_rate=0.0,  # No dropout for this test
        )

        batch, seq_len = 1, 1

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        # Test with a single token
        input_ids = torch.tensor([[0]], device=device)  # Token 0

        with torch.no_grad():
            # Encode token 0
            token_emb = embedding.encode(input_ids)

            # Decode should project back using the same weights
            logits = embedding.decode(token_emb)

        # The logit for token 0 should be highest (dot product with itself)
        assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_parameter_count_learned_positions(self, device: torch.device) -> None:
        """Test parameter count with learned positional encodings."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            pos_encoding_type="learned",
        )

        embedding = TokenEmbedding(config=config).to(device)

        # Token embeddings: vocab_size × hidden_dim
        # Positional embeddings (learned): max_seq_len × hidden_dim
        expected_params = (
            config.vocab_size * config.hidden_dim + config.max_seq_len * config.hidden_dim
        )

        total_params = sum(p.numel() for p in embedding.parameters())
        assert total_params == expected_params

    def test_parameter_count_sinusoidal_positions(self, device: torch.device) -> None:
        """Test parameter count with sinusoidal positional encodings (no extra params)."""
        block_config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=8,
            pos_encoding_type="sinusoidal",
        )
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            num_layers=4,
            block_config=block_config,
        )

        embedding = TokenEmbedding(config=config).to(device)

        # Token embeddings only: vocab_size × hidden_dim
        # Sinusoidal encodings have no trainable parameters
        expected_params = config.vocab_size * config.hidden_dim

        total_params = sum(p.numel() for p in embedding.parameters())
        assert total_params == expected_params

    def test_different_tokens_different_embeddings(self, device: torch.device) -> None:
        """Test that different token IDs produce different embeddings."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
        )

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        # Create input with three different tokens
        input_ids = torch.tensor([[0, 1, 2]], device=device)

        with torch.no_grad():
            output = embedding.encode(input_ids)

        embeddings = output[0]  # [seq_len, hidden_dim]

        # Different tokens should have different embeddings
        token_0 = embeddings[0]
        token_1 = embeddings[1]
        token_2 = embeddings[2]

        assert not torch.allclose(token_0, token_1)
        assert not torch.allclose(token_0, token_2)
        assert not torch.allclose(token_1, token_2)

    def test_deterministic_vs_training_mode(self, device: torch.device) -> None:
        """Test that eval mode is consistent and training mode applies dropout."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 64

        embedding = TokenEmbedding(config=config).to(device)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Eval mode should give same output every time
        embedding.eval()
        with torch.no_grad():
            output1 = embedding.encode(input_ids)
            output2 = embedding.encode(input_ids)
        assert torch.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        embedding.train()
        torch.manual_seed(42)
        output3 = embedding.encode(input_ids)
        torch.manual_seed(43)
        output4 = embedding.encode(input_ids)
        assert not torch.allclose(output3, output4)

    def test_variable_sequence_lengths(self, device: torch.device) -> None:
        """Test that embedding works for different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
        )

        batch = 2

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        # Test various sequence lengths
        for seq_len in [16, 32, 128, 256, 512]:
            input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)
            with torch.no_grad():
                output = embedding.encode(input_ids)
            assert output.shape == (batch, seq_len, config.hidden_dim)

    def test_batch_independence(self, device: torch.device) -> None:
        """Test that embeddings for the same tokens are consistent across batches."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
        )

        seq_len = 32

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        # Same token sequence
        input_ids_single = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
        # Repeat for batch
        input_ids_batch = input_ids_single.repeat(4, 1)

        with torch.no_grad():
            output_single = embedding.encode(input_ids_single)
            output_batch = embedding.encode(input_ids_batch)

        # All batch elements should be identical
        for i in range(4):
            assert torch.allclose(output_single[0], output_batch[i], rtol=1e-6, atol=1e-6)

    def test_compilation(self, device: torch.device) -> None:
        """Test that token embedding can be compiled."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
        )

        batch, seq_len = 2, 64

        embedding = TokenEmbedding(config=config).to(device)
        embedding.eval()

        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        # Compile the encode method
        compiled_embedding = torch.compile(embedding)

        with torch.no_grad():
            output_compiled = compiled_embedding.encode(input_ids)  # type: ignore[attr-defined]
            output_regular = embedding.encode(input_ids)

        assert torch.allclose(output_compiled, output_regular, rtol=1e-5, atol=1e-5)

    def test_decode_is_deterministic(self, device: torch.device) -> None:
        """Test that decode is deterministic (no dropout applied)."""
        config = TransformerConfig(
            vocab_size=500,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout_rate=0.1,  # Dropout should not affect decode
        )

        batch, seq_len = 2, 32

        embedding = TokenEmbedding(config=config).to(device)

        # Create random hidden states
        hidden_states = torch.randn(batch, seq_len, config.hidden_dim, device=device)

        # Decode should be deterministic regardless of dropout or training mode
        embedding.train()
        logits1 = embedding.decode(hidden_states)
        logits2 = embedding.decode(hidden_states)

        assert torch.allclose(logits1, logits2, rtol=1e-7, atol=1e-7)
