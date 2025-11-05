from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.gpt import GPTModel


class TestGPTModel:
    """Test suite for GPTModel (complete transformer)."""

    def test_output_shape(self, rng_key: Array) -> None:
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
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)
        logits = model.apply(params, input_ids, deterministic=True)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits))
        assert not jnp.any(jnp.isinf(logits))

    def test_causal_masking(self, rng_key: Array) -> None:
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
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)

        # Get predictions for full sequence
        logits_full = model.apply(params, input_ids, deterministic=True)

        # Get predictions for truncated sequence (first 5 tokens)
        input_ids_truncated = input_ids[:, :5]
        logits_truncated = model.apply(params, input_ids_truncated, deterministic=True)

        # Due to causal masking, predictions for first 5 positions should be identical
        # (they don't see the last 3 tokens anyway)
        assert jnp.allclose(logits_full[:, :5, :], logits_truncated[:, :5, :], rtol=1e-5, atol=1e-5)

    def test_parameter_count(self, rng_key: Array) -> None:
        """Test total parameter count matches expected for the architecture."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
            pos_encoding_type="learned",
            use_bias=True,
        )

        batch, seq_len = 2, 32
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)

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

        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == expected_total

    def test_deterministic_vs_training_mode(self, rng_key: Array) -> None:
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
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)

        # Deterministic mode should give same output
        logits1 = model.apply(params, input_ids, deterministic=True)
        logits2 = model.apply(params, input_ids, deterministic=True)
        assert jnp.allclose(logits1, logits2)

        # Training mode with dropout should give different outputs
        logits3 = model.apply(params, input_ids, deterministic=False, rngs={"dropout": key3})
        logits4 = model.apply(params, input_ids, deterministic=False, rngs={"dropout": key4})
        assert not jnp.allclose(logits3, logits4)

    def test_variable_sequence_lengths(self, rng_key: Array) -> None:
        """Test that model handles different sequence lengths."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch = 2
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)

        # Initialize with one sequence length
        input_ids_init = jax.random.randint(key1, (batch, 64), 0, config.vocab_size)
        params = model.init(key2, input_ids_init, deterministic=True)

        # Test various sequence lengths
        for seq_len in [8, 16, 32, 64, 128, 256]:
            input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)
            logits = model.apply(params, input_ids, deterministic=True)
            assert logits.shape == (batch, seq_len, config.vocab_size)

    def test_different_configs(self, rng_key: Array) -> None:
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
            key1, key2 = jax.random.split(rng_key)
            model = GPTModel(config=config)
            input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

            params = model.init(key2, input_ids, deterministic=True)
            logits = model.apply(params, input_ids, deterministic=True)

            assert logits.shape == (batch, seq_len, config.vocab_size)
            assert not jnp.any(jnp.isnan(logits))

    def test_gradient_flow(self, rng_key: Array) -> None:
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
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)

        # Define a simple loss function
        def loss_fn(params: Any) -> Array:
            logits = model.apply(params, input_ids, deterministic=True)
            # Mean squared error with dummy target
            return jnp.mean(logits**2)

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Check that gradients exist and are non-zero for all parameters
        def check_grads(grad_tree: Any) -> None:
            for leaf in jax.tree.leaves(grad_tree):
                assert not jnp.all(leaf == 0), "Some gradients are zero"
                assert not jnp.any(jnp.isnan(leaf)), "Some gradients are NaN"

        check_grads(grads)
        assert not jnp.isnan(loss)

    def test_jit_compilation(self, rng_key: Array) -> None:
        """Test that model can be JIT compiled."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch, seq_len = 2, 32
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)

        # Create JIT-compiled forward pass
        @jax.jit
        def forward_jit(params: Any, input_ids: Array) -> Array:
            return model.apply(params, input_ids, deterministic=True)

        logits_jit = forward_jit(params, input_ids)
        logits_regular = model.apply(params, input_ids, deterministic=True)

        assert jnp.allclose(logits_jit, logits_regular, rtol=1e-5, atol=1e-5)

    def test_batch_size_one(self, rng_key: Array) -> None:
        """Test that model works with batch size of 1."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=256,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        batch, seq_len = 1, 32
        key1, key2 = jax.random.split(rng_key)

        model = GPTModel(config=config)
        input_ids = jax.random.randint(key1, (batch, seq_len), 0, config.vocab_size)

        params = model.init(key2, input_ids, deterministic=True)
        logits = model.apply(params, input_ids, deterministic=True)

        assert logits.shape == (batch, seq_len, config.vocab_size)
        assert not jnp.any(jnp.isnan(logits))

    def test_positional_encoding_integration(self, rng_key: Array) -> None:
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

            key1, key2 = jax.random.split(rng_key)
            model = GPTModel(config=config)

            # Same tokens at different positions should give different predictions
            input_ids = jnp.full((batch, seq_len), fill_value=5)  # All token ID 5

            params = model.init(key2, input_ids, deterministic=True)
            logits = model.apply(params, input_ids, deterministic=True)

            # First and last position should have different predictions
            # (even though input tokens are the same)
            first_pos_logits = logits[0, 0, :]
            last_pos_logits = logits[0, -1, :]

            assert not jnp.allclose(first_pos_logits, last_pos_logits, rtol=0.1)
