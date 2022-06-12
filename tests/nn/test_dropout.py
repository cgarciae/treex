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
        rate=st.floats(min_value=0.0, max_value=1.0),
        broadcast_dims=st.lists(
            st.integers(min_value=0, max_value=2), min_size=0, max_size=2
        ),
        training=st.booleans(),
        frozen=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_dropout_equivalence(
        self,
        batch_size,
        length,
        channels,
        rate,
        broadcast_dims,
        training,
        frozen,
    ):
        deterministic = not training or frozen
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = tx.Key(42)

        flax_module = linen.Dropout(
            rate=rate,
            broadcast_dims=broadcast_dims,
            deterministic=deterministic,
        )
        treex_module = (
            tx.Dropout(
                rate=rate,
                broadcast_dims=broadcast_dims,
            )
            .train(training)
            .freeze(frozen)
        )

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init({"dropout": flax_key}, x)
        treex_module = treex_module.init(key=key)(x)

        # split key same way tx.Dropout does internally
        rng, _ = tx.iter_split(flax_key, 2)

        y_flax = flax_module.apply(variables, x, rngs={"dropout": rng})
        y_treex, _ = treex_module.apply(key=flax_key)(x)

        assert np.allclose(y_flax, y_treex)

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(key=42)(x)

        y, module = module.apply(key=420)(x)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(key=42)(x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 0

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(key=42)(x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 0

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(key=42)(x)

        @jax.jit
        def f(module, x):
            return module.apply(key=42)(x)

        y, module2 = f(module, x)

        assert y.shape == (10, 2)

    def test_eval(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Dropout(0.5).init(key=42)(x)

        y1, module = module.apply(key=42)(x)
        y2, module = module.apply(key=69)(x)

        assert not np.allclose(y1, y2)

        module = module.eval()

        y1, module = module.apply(key=42)(x)
        y2, module = module.apply(key=69)(x)

        assert np.allclose(y1, y2)
