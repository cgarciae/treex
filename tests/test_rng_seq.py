import jax
import jax.numpy as jnp
import numpy as np

import treex as tx


class TestRNGSeq:
    def test_next(self):
        next_key = tx.KeySeq().init(42)

        internal_key = next_key.key
        key_next = next_key()
        next_internal_key = next_key.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(key_next, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])

    def test_jit(self):
        next_key = tx.KeySeq().init(42)
        internal_key = next_key.key

        @jax.jit
        def f(next_key):
            return next_key, next_key()

        next_key, key_next = f(next_key)
        next_internal_key = next_key.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(key_next, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])
