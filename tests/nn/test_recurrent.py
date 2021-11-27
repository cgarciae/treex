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
        carry = recurrent.GRU(hidden_dim).module.initialize_carry(
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

        gru = recurrent.GRU(hidden_dim,
            return_state=True, return_sequences=True, time_major=time_major
        )
        gru = gru.init(key, (jnp.ones((1, 1, features)), jnp.ones((1, hidden_dim))))

        carry = gru.module.initialize_carry(key, (batch_size,), hidden_dim)

        dims = (batch_size, timesteps, features)
        if time_major:
            dims = (timesteps, batch_size, features)

        sequences, final_state = gru(jnp.ones(dims), carry)

        assert final_state.shape == (batch_size, hidden_dim)

        if time_major:
            assert sequences.shape == (timesteps, batch_size, hidden_dim)
        else:
            assert sequences.shape == (batch_size, timesteps, hidden_dim)

    def test_jit(self):
        x = np.random.uniform(size=(10, 20, 2))
        module = recurrent.GRU(3).init(42, (x, jnp.zeros((10, 3))))

        @jax.jit
        def f(module, x):
            return module, module(x, jnp.zeros((10, 3)))

        module2, y = f(module, x)
        assert y.shape == (10, 3)
        print(jax.tree_leaves(module))
        print(jax.tree_leaves(module2))
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.parameters()),
                jax.tree_leaves(module2.parameters()),
            )
        )

    @hp.given(return_state=st.booleans(), return_sequences=st.booleans())
    @hp.settings(deadline=None, max_examples=20)
    def test_return_state_and_sequences(self, return_state, return_sequences):
        key = tx.Key(8)
        hidden_dim = 5
        features = 10
        batch_size = 32
        time = 10

        gru = recurrent.GRU(hidden_dim,
            return_state=return_state, return_sequences=return_sequences
        )
        gru = gru.init(key, (jnp.ones((1, 1, features)), jnp.zeros((1, hidden_dim))))

        output = gru(
            jnp.ones((batch_size, time, features)), jnp.zeros((batch_size, hidden_dim))
        )

        sequence_shape = (batch_size, time, hidden_dim)
        state_shape = (batch_size, hidden_dim)
        if return_sequences and not return_state:
            assert output.shape == sequence_shape
        elif return_state and return_sequences:
            assert output[0].shape == sequence_shape and output[1].shape == state_shape
        else:
            assert output.shape == state_shape

    def test_backward_mode(self):
        key = tx.Key(8)
        hidden_dim = 5
        features = 10
        batch_size = 32
        time = 10

        gru_fwd = recurrent.GRU(hidden_dim, go_backwards=False)
        gru_fwd = gru_fwd.init(
            key, (jnp.ones((1, 1, features)), jnp.zeros((1, hidden_dim)))
        )

        gru_bwd = recurrent.GRU(hidden_dim, go_backwards=True)
        gru_bwd.params = gru_fwd.params
        inputs, init_carry = (
            jnp.ones((batch_size, time, features)),
            jnp.zeros((batch_size, hidden_dim)),
        )

        assert np.allclose(
            gru_fwd(inputs[:, ::-1, :], init_carry), gru_fwd(inputs, init_carry)
        )

    def test_optional_initial_state(self):
        key = tx.Key(8)
        hidden_dim = 5
        features = 10
        batch_size = 32
        time = 10

        gru = recurrent.GRU(hidden_dim, go_backwards=False)
        gru = gru.init(
            key, (jnp.ones((1, 1, features)), jnp.zeros((1, hidden_dim)))
        )

        inputs = np.random.rand(batch_size, time, features)
        assert np.allclose(gru(inputs), gru(inputs, np.zeros((batch_size, hidden_dim))))
        assert np.allclose(gru(inputs), gru(inputs, gru.initialize_state(batch_size)))
