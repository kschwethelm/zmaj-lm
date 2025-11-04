import jax
import jax.numpy as jnp
from jax import Array

from zmaj_lm.config.model_config import TransformerConfig
from zmaj_lm.models.feedforward import FeedForward


class TestFeedForward:
    """Test suite for FeedForward (MLP) module."""

    def test_output_shape(self, rng_key: Array) -> None:
        """Test that FFN preserves input shape (batch, seq_len, hidden_dim)."""
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

        ffn = FeedForward(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = ffn.init(key2, x, deterministic=True)
        output = ffn.apply(params, x, deterministic=True)

        assert output.shape == (batch, seq_len, config.hidden_dim)
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_parameter_count(self, rng_key: Array) -> None:
        """Test that parameter count matches expected for FFN."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            mlp_dim=1024,
            use_bias=True,
        )

        batch, seq_len = 2, 8
        key1, key2 = jax.random.split(rng_key)

        ffn = FeedForward(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = ffn.init(key2, x, deterministic=True)

        # First layer: (hidden_dim -> mlp_dim) + mlp_dim bias
        # Second layer: (mlp_dim -> hidden_dim) + hidden_dim bias
        expected_total = (
            config.hidden_dim * config.mlp_dim
            + config.mlp_dim
            + config.mlp_dim * config.hidden_dim
            + config.hidden_dim
        )

        total_params = sum(x.size for x in jax.tree.leaves(params))
        assert total_params == expected_total

    def test_deterministic_vs_training(self, rng_key: Array) -> None:
        """Test that deterministic mode produces consistent outputs and training mode applies dropout."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_dim=512,
            dropout_rate=0.1,
        )

        batch, seq_len = 2, 8
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        ffn = FeedForward(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = ffn.init(key2, x, deterministic=True)

        # Deterministic mode should give same output every time
        output1 = ffn.apply(params, x, deterministic=True)
        output2 = ffn.apply(params, x, deterministic=True)
        assert jnp.allclose(output1, output2)

        # Training mode with dropout should give different outputs
        output3 = ffn.apply(params, x, deterministic=False, rngs={"dropout": key3})
        output4 = ffn.apply(params, x, deterministic=False, rngs={"dropout": key4})
        assert not jnp.allclose(output3, output4)

    def test_activation_nonlinearity(self, rng_key: Array) -> None:
        """Test that FFN applies nonlinear activation (GELU)."""
        config = TransformerConfig(
            vocab_size=1000,
            max_seq_len=512,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=256,
            dropout_rate=0.0,
        )

        key1, key2 = jax.random.split(rng_key)

        ffn = FeedForward(config=config)
        # Use a simple input pattern
        x = jnp.ones((1, 4, config.hidden_dim))

        params = ffn.init(key2, x, deterministic=True)
        output = ffn.apply(params, x, deterministic=True)

        # If activation is working, output should not be a simple linear transformation
        # We verify this by checking that FFN(2*x) != 2*FFN(x) (non-linear behavior)
        x_scaled = 2.0 * x
        output_scaled_input = ffn.apply(params, x_scaled, deterministic=True)
        output_scaled = 2.0 * output

        # Due to GELU nonlinearity, these should NOT be equal
        assert not jnp.allclose(output_scaled_input, output_scaled, rtol=0.01)

    def test_jit_compilation(self, rng_key: Array) -> None:
        """Test that FFN can be JIT compiled."""
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

        ffn = FeedForward(config=config)
        x = jax.random.normal(key1, (batch, seq_len, config.hidden_dim))

        params = ffn.init(key2, x, deterministic=True)

        # Create JIT-compiled apply function
        @jax.jit
        def apply_jit(params: Array, x: Array) -> Array:
            return ffn.apply(params, x, deterministic=True)

        output_jit = apply_jit(params, x)
        output_regular = ffn.apply(params, x, deterministic=True)

        assert jnp.allclose(output_jit, output_regular, rtol=1e-5, atol=1e-5)
