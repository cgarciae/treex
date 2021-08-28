import unittest

import hypothesis as hp
import jax
import numpy as np
from flax import linen
from hypothesis import strategies as st

import treex as tx

BIAS_INITS = (
    tx.initializers.zeros,
    tx.initializers.ones,
    tx.initializers.normal(),
    tx.initializers.uniform(),
)
KERNEL_INITS = BIAS_INITS + (
    tx.initializers.xavier_uniform(),
    tx.initializers.xavier_normal(),
    tx.initializers.lecun_uniform(),
    tx.initializers.lecun_normal(),
    tx.initializers.kaiming_uniform(),
    tx.initializers.kaiming_normal(),
)


class LinearTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        features_in=st.integers(min_value=1, max_value=32),
        training=st.booleans(),
        features=st.integers(min_value=1, max_value=32),  #
        use_bias=st.booleans(),
        kernel_init=st.sampled_from(KERNEL_INITS),
        bias_init=st.sampled_from(BIAS_INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence(
        self,
        batch_size,
        length,
        features_in,
        training,
        features,
        use_bias,
        kernel_init,
        bias_init,
    ):
        shape = (batch_size, length, length, features_in)

        x = np.random.uniform(size=shape)

        key = jax.random.PRNGKey(42)

        flax_module = linen.Dense(
            features=features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        treex_module = tx.Linear(
            features_in=features_in,
            features_out=features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        ).train(training)

        flax_key, _ = jax.random.split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key)

        assert set(variables["params"]) == set(treex_module.params)
        assert all(
            np.allclose(variables["params"][name], treex_module.params[name])
            for name in variables["params"]
        )

        y_flax, updates = flax_module.apply(variables, x, mutable=["batch_stats"])
        variables = variables.copy(updates)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        assert all(
            np.allclose(variables["params"][name], treex_module.params[name])
            for name in variables["params"]
        )

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(2, 3).init(42)

        y = module(x)

        assert y.shape == (10, 3)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(2, 3).init(42)

        flat = jax.tree_leaves(module)

        assert len(flat) == 2

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(2, 3).init(42)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(2, 3).init(42)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 3)
        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(module), jax.tree_leaves(module2))
        )
