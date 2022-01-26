import unittest

import hypothesis as hp
import jax
import numpy as np
from flax import linen
from hypothesis import strategies as st

import treex as tx

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
        axis=st.sampled_from([-1]),  # flax has an error with other axis
        momentum=st.floats(min_value=0.01, max_value=1.0),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
        training=st.booleans(),
        frozen=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence(
        self,
        batch_size,
        length,
        channels,
        axis,
        momentum,
        epsilon,
        use_bias,
        use_scale,
        bias_init,
        scale_init,
        training,
        frozen,
    ):
        use_running_average = not training or frozen
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = tx.Key(42)

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
        treex_module = (
            tx.BatchNorm(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon,
                use_bias=use_bias,
                use_scale=use_scale,
                bias_init=bias_init,
                scale_init=scale_init,
            )
            .train(training)
            .freeze(frozen)
        )

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key, x)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

        assert np.allclose(variables["batch_stats"]["mean"], treex_module.mean)
        assert np.allclose(variables["batch_stats"]["var"], treex_module.var)

        y_flax, updates = flax_module.apply(variables, x, mutable=["batch_stats"])
        variables = variables.copy(updates)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

        assert np.allclose(variables["batch_stats"]["mean"], treex_module.mean)
        assert np.allclose(variables["batch_stats"]["var"], treex_module.var)

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.BatchNorm().init(42, x)

        y = module(x)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.BatchNorm().init(42, x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 5

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.BatchNorm().init(42, x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))
        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.BatchStat))
        assert len(flat) == 2

        flat = jax.tree_leaves(
            module.filter(lambda field: not issubclass(field.kind, tx.TreePart))
        )
        assert len(flat) == 1

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.BatchNorm().init(42, x)

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
        module = tx.BatchNorm().init(42, x)

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


class LayerNormTest(unittest.TestCase):
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.integers(min_value=1, max_value=32),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence(
        self,
        batch_size,
        length,
        channels,
        epsilon,
        use_bias,
        use_scale,
        bias_init,
        scale_init,
    ):
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = tx.Key(42)

        flax_module = linen.LayerNorm(
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        )
        treex_module = tx.LayerNorm(
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        )

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key, x)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

        y_flax = flax_module.apply(variables, x)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.LayerNorm().init(42, x)

        y = module(x)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.LayerNorm().init(42, x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 2

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.LayerNorm().init(42, x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))
        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.BatchStat))
        assert len(flat) == 0

        flat = jax.tree_leaves(
            module.filter(lambda field: not issubclass(field.kind, tx.TreePart))
        )
        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.LayerNorm().init(42, x)

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

    def test_eval(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.LayerNorm().init(42, x)

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


class GroupNormTest(unittest.TestCase):
    def _test_equivalence(
        self,
        batch_size,
        length,
        channels,
        num_groups,
        group_size,
        epsilon,
        use_bias,
        use_scale,
        bias_init,
        scale_init,
    ):
        shape = (batch_size, length, channels)

        x = np.random.uniform(size=shape)

        key = tx.Key(42)

        flax_module = linen.GroupNorm(
            num_groups=num_groups,
            group_size=group_size,
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        )
        treex_module = tx.GroupNorm(
            num_groups=num_groups,
            group_size=group_size,
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            bias_init=bias_init,
            scale_init=scale_init,
        )

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key, x)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

        y_flax = flax_module.apply(variables, x)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        if use_scale:
            assert np.allclose(variables["params"]["scale"], treex_module.scale)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.integers(min_value=1, max_value=32),
        num_groups=st.none(),
        group_size=st.just(1),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence_channels(self, **kwargs):
        self._test_equivalence(**kwargs)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.just(32),
        num_groups=st.sampled_from([2 ** i for i in range(5)]),
        group_size=st.none(),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence_num_groups(self, **kwargs):
        self._test_equivalence(**kwargs)

    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        channels=st.just(32),
        num_groups=st.none(),
        group_size=st.sampled_from([2 ** i for i in range(5)]),
        epsilon=st.floats(min_value=0.000001, max_value=0.01),
        use_bias=st.booleans(),
        use_scale=st.booleans(),
        bias_init=st.sampled_from(INITS),
        scale_init=st.sampled_from(INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence_group_size(self, **kwargs):
        self._test_equivalence(**kwargs)

    def test_call(self):
        x = np.random.uniform(size=(10, 32))
        module = tx.GroupNorm().init(42, x)

        y = module(x)

        assert y.shape == (10, 32)

    def test_tree(self):
        x = np.random.uniform(size=(10, 32))
        module = tx.GroupNorm().init(42, x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 2

    def test_slice(self):
        x = np.random.uniform(size=(10, 32))
        module = tx.GroupNorm().init(42, x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))
        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.BatchStat))
        assert len(flat) == 0

        flat = jax.tree_leaves(
            module.filter(lambda field: not issubclass(field.kind, tx.TreePart))
        )
        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 32))
        module = tx.GroupNorm().init(42, x)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 32)
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.Parameter)),
                jax.tree_leaves(module2.filter(tx.Parameter)),
            )
        )

    def test_eval(self):
        x = np.random.uniform(size=(10, 32))
        module = tx.GroupNorm().init(42, x)

        @jax.jit
        def f(module, x):
            module = module.eval()

            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 32)
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                jax.tree_leaves(module.filter(tx.Parameter)),
                jax.tree_leaves(module2.filter(tx.Parameter)),
            )
        )
