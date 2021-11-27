import hypothesis as hp
import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
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
        timesteps=st.integers(min_value=1, max_value=32),
        time_major=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_forward(self, batch_size, hidden_dim, features, timesteps, time_major):
        key = tx.Key(8)

        gru = recurrent.GRU(return_state=True, return_sequences=True, time_major=time_major)
        gru = gru.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, 1, features))))

        carry = gru.module.initialize_carry(key, (batch_size,), hidden_dim)

        dims = (batch_size, timesteps, features)
        if time_major:
            dims = (timesteps, batch_size, features)

        final_state, sequences = gru(carry, jnp.ones(dims))

        assert final_state.shape == (batch_size, hidden_dim)

        if time_major:
            assert sequences.shape == (timesteps, batch_size, hidden_dim)
        else:
            assert sequences.shape == (batch_size, timesteps, hidden_dim)

    @hp.given(return_state=st.booleans(), return_sequences=st.booleans())
    @hp.settings(deadline=None, max_examples=20)
    def test_return_state_and_sequences(self, return_state, return_sequences):
        key = tx.Key(8)
        hidden_dim = 5
        features = 10
        batch_size = 32
        time = 10

        gru = recurrent.GRU(return_state=return_state, return_sequences=return_sequences)
        gru = gru.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, 1, features))))

        output = gru(jnp.zeros((batch_size, hidden_dim)), jnp.ones((batch_size, time, features)))

        sequence_shape = (batch_size, time, hidden_dim)
        state_shape = (batch_size, hidden_dim)
        if return_sequences and not return_state:
            assert output.shape == sequence_shape
        elif return_state and return_sequences:
            assert output[0].shape == state_shape and output[1].shape == sequence_shape
        else:
            assert output.shape == state_shape

    def test_reverse(self):
        key = tx.Key(8)
        hidden_dim = 5
        features = 10
        batch_size = 32
        time = 10

        gru_fwd = recurrent.GRU(go_backwards=False)
        gru_fwd = gru_fwd.init(key, (jnp.ones((1, hidden_dim)), jnp.ones((1, 1, features))))

        gru_bwd = recurrent.GRU(go_backwards=True)
        gru_bwd.params = gru_fwd.params
        init_carry, inputs  = (jnp.zeros((batch_size, hidden_dim)), jnp.ones((batch_size, time, features)))

        assert np.allclose(gru_fwd(init_carry, inputs[:, ::-1, :]), gru_fwd(init_carry, inputs))
