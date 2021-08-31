import jax
import jax.numpy as jnp
import numpy as np

import treex as tx


class TestRNGSeq:
    def test_next(self):
        rng = tx.RNGSeq().init(42)

        internal_key = rng.key
        next_key = rng.next()
        next_internal_key = rng.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(next_key, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])

    def test_jit(self):
        rng = tx.RNGSeq().init(42)
        internal_key = rng.key

        @jax.jit
        def f(rng):
            return rng, rng.next()

        rng, next_key = f(rng)
        next_internal_key = rng.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(next_key, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])
