import unittest

import hypothesis as hp
import jax
import numpy as np
import treex as tx
from flax import linen
from hypothesis import strategies as st


class BatchNormTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        channels=st.integers(min_value=1, max_value=32),
        training=st.booleans(),
        axis=st.sampled_from([-1, 1]),  # flax has an error with axis = -2 and 0
    )
    @hp.settings(deadline=None)
    def test_equivalence(self, batch_size, channels, training, axis):
        use_running_average = not training
        shape = (batch_size, channels)

        x = np.random.uniform(size=shape)

        key = jax.random.PRNGKey(42)

        flax_module = linen.BatchNorm(
            use_running_average=use_running_average,
            axis=axis,
        )
        treex_module = tx.nn.BatchNorm(
            features_in=shape[axis],
            axis=axis,
        ).train(training)

        treex_module = treex_module.init(key)
        variables = flax_module.init(key, x)

        assert set(variables["params"]) == set(treex_module.params)
        assert all(
            np.allclose(variables["params"][name], treex_module.params[name])
            for name in variables["params"]
        )
        assert all(
            np.allclose(variables["batch_stats"][name], treex_module.batch_stats[name])
            for name in variables["batch_stats"]
        )

        y_flax, updates = flax_module.apply(variables, x, mutable=["batch_stats"])
        variables = variables.copy(updates)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)
        assert all(
            np.allclose(variables["params"][name], treex_module.params[name])
            for name in variables["params"]
        )
        assert all(
            np.allclose(variables["batch_stats"][name], treex_module.batch_stats[name])
            for name in variables["batch_stats"]
        )

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        y = module(x)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        flat = jax.tree_leaves(module)

        assert len(flat) == 4

    def test_slice(self):
        module = tx.nn.BatchNorm(2).init(42)

        flat = jax.tree_leaves(module.slice(tx.Parameter))

        assert len(flat) == 2

        flat = jax.tree_leaves(module.slice(tx.State))

        assert len(flat) == 2

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 2)
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.slice(tx.Parameter)),
                jax.tree_leaves(module2.slice(tx.Parameter)),
            )
        )
        assert not all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.slice(tx.State)),
                jax.tree_leaves(module2.slice(tx.State)),
            )
        )

    def test_eval(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        @jax.jit
        def f(module, x):
            module = module.train(False)

            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 2)
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.slice(tx.Parameter)),
                jax.tree_leaves(module2.slice(tx.Parameter)),
            )
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.slice(tx.State)),
                jax.tree_leaves(module2.slice(tx.State)),
            )
        )
