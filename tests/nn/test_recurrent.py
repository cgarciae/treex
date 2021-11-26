import hypothesis as hp
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex.nn import recurrent


class TestGRU:
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        hidden_dim=st.integers(min_value=1, max_value=32),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_init_carry(self, batch_size, hidden_dim):
        next_key = tx.KeySeq().init(42)
        carry = recurrent.GRU().module.initialize_carry(
            next_key, (batch_size,), hidden_dim
        )
        assert carry.shape == (batch_size, hidden_dim)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        hidden_dim=st.integers(min_value=1, max_value=32),
        features=st.integers(min_value=1, max_value=32),
        time_dim=st.integers(min_value=1, max_value=32),
        time_major=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_forward(self, batch_size, hidden_dim, features, time_dim, time_major):
        key = tx.Key(8)

        gru = recurrent.GRU(time_major=time_major)
        gru = gru.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, 1, features))))

        carry = gru.module.initialize_carry(key, (batch_size,), hidden_dim)
        dims = (
            (time_dim, batch_size, features)
            if time_major
            else (batch_size, time_dim, features)
        )
        y = gru(carry, jnp.ones(dims))

        assert y[0].shape == (batch_size, hidden_dim)

        if time_major:
            assert y[1].shape == (time_dim, batch_size, hidden_dim)
        else:
            assert y[1].shape == (batch_size, time_dim, hidden_dim)
