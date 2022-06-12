import jax
import jax.numpy as jnp
import numpy as np

import treex as tx


class TestRNGSeq:
    def test_next(self):
        seq = tx.KeySeq(42)

        internal_key = seq.key
        key_next = seq.next()
        next_internal_key = seq.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(key_next, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])

    def test_next_function(self):
        seq = tx.KeySeq(42)

        internal_key = seq.key
        key_next = next(seq)
        next_internal_key = seq.key

        assert isinstance(internal_key, jnp.ndarray)
        assert isinstance(next_internal_key, jnp.ndarray)
        assert np.allclose(key_next, jax.random.split(internal_key)[0])
        assert np.allclose(next_internal_key, jax.random.split(internal_key)[1])
