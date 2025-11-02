from collections.abc import Iterator

import jax
import jax.random as jr


def split_key(key: jax.Array, num: int = 2) -> list[jax.Array]:
    """Split PRNG into `num` independent subkeys.

    Args:
        key: JAX PRNG key.
        num: Number of subkeys to generate.

    Returns:
        List of `num` independent PRNG keys.
    """
    return list(jr.split(key, num=num))


def key_generator(key: jax.Array) -> Iterator[jax.Array]:
    """Yield infinite sequence of independent PRNG keys.

    Args:
        key: Initial PRNG key.

    Yields:
        Independent PRNG keys, one per iteration
    """
    while True:
        key, subkey = jr.split(key)
        yield subkey
