import jax
import jax.numpy as jnp
from jax import Array

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.transformer_block import TransformerBlock
from zmaj_lm.utils.masks import create_causal_mask


class TestTransformerBlock:
    """Test suite for TransformerBlock module."""

    def test_output_shape(self, rng_key: Array) -> None:
        """Test that TransformerBlock preserves input shape."""
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

        block = TransformerBlock(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = block.init(key2, x, deterministic=True)
        output = block.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_with_causal_mask(self, rng_key: Array) -> None:
        """Test TransformerBlock with causal mask."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        block = TransformerBlock(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))
        mask = create_causal_mask(seq_len)

        params = block.init(key2, x, mask=mask, deterministic=True)
        output = block.apply(params, x, mask=mask, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_deterministic_vs_training(self, rng_key: Array) -> None:
        """Test that deterministic mode is consistent and training mode varies with dropout."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            residual_dropout_rate=0.1,
        )

        batch, seq_len = 2, 8
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        block = TransformerBlock(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = block.init(key2, x, deterministic=True)

        # Deterministic mode should give same output every time
        output1 = block.apply(params, x, deterministic=True)
        output2 = block.apply(params, x, deterministic=True)
        assert jnp.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        output3 = block.apply(params, x, deterministic=False, rngs={"dropout": key3})
        output4 = block.apply(params, x, deterministic=False, rngs={"dropout": key4})
        assert not jnp.allclose(output3, output4)

    def test_gradient_flow(self, rng_key: Array) -> None:
        """Test that gradients flow through residual connections."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,  # Disable dropout for gradient test
        )

        batch, seq_len = 2, 4
        key1, key2 = jax.random.split(rng_key)

        block = TransformerBlock(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = block.init(key2, x, deterministic=True)

        # Define a simple loss function
        def loss_fn(params: Array, x: Array) -> Array:
            output = block.apply(params, x, deterministic=True)
            return jnp.sum(output**2)

        # Compute gradients
        grads = jax.grad(loss_fn)(params, x)

        # Check that all parameters have non-zero gradients
        grad_leaves = jax.tree.leaves(grads)
        for grad in grad_leaves:
            # Gradients should exist and not be all zeros
            assert not jnp.all(grad == 0.0)
            assert not jnp.any(jnp.isnan(grad))
            assert not jnp.any(jnp.isinf(grad))

    def test_jit_compilation(self, rng_key: Array) -> None:
        """Test that TransformerBlock can be JIT compiled."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        block = TransformerBlock(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = block.init(key2, x, deterministic=True)

        # Create JIT-compiled apply function
        @jax.jit
        def apply_jit(params: Array, x: Array) -> Array:
            return block.apply(params, x, deterministic=True)

        output_jit = apply_jit(params, x)
        output_regular = block.apply(params, x, deterministic=True)

        assert jnp.allclose(output_jit, output_regular, rtol=1e-5, atol=1e-5)
