import unittest

import hypothesis as hp
import jax
import numpy as np
from flax import linen
from hypothesis import strategies as st

import treex as tx


class DropoutTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.integers(min_value=1, max_value=32),
        training=st.booleans(),
        rate=st.floats(min_value=0.0, max_value=1.0),
        broadcast_dims=st.lists(
            st.integers(min_value=0, max_value=2), min_size=0, max_size=2
        ),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_dropout_equivalence(
        self,
        batch_size,
        length,
        channels,
        training,
        rate,
        broadcast_dims,
    ):
        deterministic = not training
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = jax.random.PRNGKey(42)

        flax_module = linen.Dropout(
            rate=rate,
            broadcast_dims=broadcast_dims,
            deterministic=deterministic,
        )
        treex_module = tx.Dropout(
            rate=rate,
            broadcast_dims=broadcast_dims,
        ).train(training)

        flax_key, _ = jax.random.split(key)  # emulate init split
        variables = flax_module.init({"dropout": flax_key}, x)
        treex_module = treex_module.init(key)

        # split key same way tx.Dropout does internally
        rng, _ = jax.random.split(flax_key, 2)

        y_flax = flax_module.apply(variables, x, rng=rng)
        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(42)

        y = module(x)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(42)

        flat = jax.tree_leaves(module)

        assert len(flat) == 1

    def test_slice(self):
        module = tx.Dropout(0.5).init(42)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 0

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 1

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(42)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 2)
        assert not all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.State)),
                jax.tree_leaves(module2.filter(tx.State)),
            )
        )

    def test_eval(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(42)

        y1 = module(x)
        y2 = module(x)

        assert not np.allclose(y1, y2)

        module = module.eval()

        y1 = module(x)
        y2 = module(x)

        assert np.allclose(y1, y2)
