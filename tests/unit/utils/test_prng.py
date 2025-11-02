"""Unit tests for the PRNG utilities."""

import jax
import jax.random as jr

from zmaj_lm.utils.prng import key_generator, split_key


class TestKeyGenerator:
    """Tests for PRNG utility function: key_generator."""

    def test_key_generator_yields_keys(self) -> None:
        """Test that key_generator yields independent PRNG keys."""
        key = jr.PRNGKey(72)
        gen = key_generator(key)
        keys = [next(gen) for _ in range(5)]
        assert len(keys) == 5
        assert all(isinstance(k, jax.Array) for k in keys)
        assert all(not jax.numpy.array_equal(k, key) for k in keys)

    def test_key_generator_deterministic(self) -> None:
        """Test that generator with same seed produces same sequence."""
        key = jr.PRNGKey(99)
        gen1 = key_generator(key)
        gen2 = key_generator(key)

        keys1 = [next(gen1) for _ in range(10)]
        keys2 = [next(gen2) for _ in range(10)]

        for k1, k2 in zip(keys1, keys2):
            assert jax.numpy.array_equal(k1, k2)


class TestSplitKey:
    """Tests for PRNG utility function: split_key."""

    def test_split_key_default(self) -> None:
        """Test splitting key with default num=2."""
        key = jr.PRNGKey(72)
        keys = split_key(key)
        assert isinstance(keys, list)
        assert len(keys) == 2
        assert all(isinstance(k, jax.Array) for k in keys)

    def test_split_key_custom_num(self) -> None:
        """Test splitting key with custom num."""
        key = jr.PRNGKey(72)
        num_subkeys = 5
        keys = split_key(key, num=num_subkeys)
        assert isinstance(keys, list)
        assert len(keys) == num_subkeys
        assert all(isinstance(k, jax.Array) for k in keys)

    def test_split_key_deterministic(self) -> None:
        """Test that splitting the same key produces identical results."""
        key = jr.PRNGKey(42)
        keys1 = split_key(key, num=3)
        keys2 = split_key(key, num=3)

        # Same key should produce identical splits
        for k1, k2 in zip(keys1, keys2):
            assert jax.numpy.array_equal(k1, k2)

    def test_split_key_independence(self) -> None:
        """Test that subkeys generate independent random values."""
        key = jr.PRNGKey(42)
        keys = split_key(key, num=100)

        # Generate random normals from each key
        values = [jr.normal(k, shape=(1000,)) for k in keys]
        means = [v.mean().item() for v in values]

        # Means should vary (not all identical)
        assert len(set(means)) > 1  # At least some variation
        # Could also check: standard deviation of means ≈ 1/sqrt(1000)
