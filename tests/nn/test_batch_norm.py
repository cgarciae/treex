import unittest

import hypothesis as hp
import jax
import numpy as np
import treex as tx
from flax import linen
from hypothesis import strategies as st

INITS = (
    tx.initializers.zeros,
    tx.initializers.ones,
    tx.initializers.normal(),
    tx.initializers.uniform(),
)


class BatchNormTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.integers(min_value=1, max_value=32),
        training=st.booleans(),
        axis=st.sampled_from([-1]),  # flax has an error with other axis
        momentum=st.floats(min_value=0.01, max_value=1.0),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
    )
    @hp.settings(deadline=None)
    def test_equivalence(
        self,
        batch_size,
        length,
        channels,
        training,
        axis,
        momentum,
        epsilon,
        use_bias,
        use_scale,
        bias_init,
        scale_init,
    ):
        use_running_average = not training
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = jax.random.PRNGKey(42)

        flax_module = linen.BatchNorm(
            use_running_average=use_running_average,
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        )
        treex_module = tx.nn.BatchNorm(
            features_in=shape[axis],
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        ).train(training)

        flax_key, _ = jax.random.split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key)

        if "params" in variables:
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

        if "params" in variables:
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

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.BatchStat))

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
                jax.tree_leaves(module.filter(tx.Parameter)),
                jax.tree_leaves(module2.filter(tx.Parameter)),
            )
        )
        assert not all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.BatchStat)),
                jax.tree_leaves(module2.filter(tx.BatchStat)),
            )
        )

    def test_eval(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        @jax.jit
        def f(module, x):
            module = module.eval()

            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 2)
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.Parameter)),
                jax.tree_leaves(module2.filter(tx.Parameter)),
            )
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.BatchStat)),
                jax.tree_leaves(module2.filter(tx.BatchStat)),
            )
        )
