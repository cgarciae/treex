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
        carry = recurrent.GRUCell.initialize_carry(next_key, (batch_size,), hidden_dim)
        assert carry.shape == (batch_size, hidden_dim)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        hidden_dim=st.integers(min_value=1, max_value=32),
        features=st.integers(min_value=1, max_value=32),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_initialize_carry(self, batch_size, hidden_dim, features):
        key = tx.Key(8)
        gru = recurrent.GRUCell()
        gru = gru.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, features))))
        carry = gru.init_carry((batch_size,))
        assert carry.shape == (batch_size, hidden_dim)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        hidden_dim=st.integers(min_value=1, max_value=32),
        features=st.integers(min_value=1, max_value=32),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_forward(self, batch_size, hidden_dim, features):
        key = tx.Key(8)
        gru = recurrent.GRUCell()
        gru = gru.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, features))))
        carry = gru.init_carry((batch_size,))
        y = gru(carry, jnp.ones((batch_size, features)))
        assert y[0].shape == y[1].shape == carry.shape
