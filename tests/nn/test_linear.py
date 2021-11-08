import functools
import typing as tp
import unittest
from dataclasses import dataclass

import einops
import hypothesis as hp
import jax
import jax.numpy as jnp
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

A = tp.TypeVar("A")


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

        key = tx.Key(42)

        flax_module = linen.Dense(
            features=features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        treex_module = tx.Linear(
            features_out=features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
        ).train(training)

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key, x)

        assert np.allclose(variables["params"]["kernel"], treex_module.kernel)
        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

        y_flax, updates = flax_module.apply(variables, x, mutable=["batch_stats"])
        variables = variables.copy(updates)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        assert np.allclose(variables["params"]["kernel"], treex_module.kernel)
        if use_bias:
            assert np.allclose(variables["params"]["bias"], treex_module.bias)

    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(3).init(42, x)

        y = module(x)

        assert y.shape == (10, 3)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(3).init(42, x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 2

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(3).init(42, x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 2

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.Linear(3).init(42, x)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 3)
        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(module), jax.tree_leaves(module2))
        )

    def test_axis_name(self):
        class Linear3D(tx.Linear):
            def __init__(
                self,
                num_heads: int,
                *args,
                axis_name: tp.Any = "batch",
                **kwargs,
            ):
                super().__init__(*args, axis_name=axis_name, **kwargs)
                self.num_heads = num_heads

            def __call__(self, x):
                def call(module: "Linear3D", x):
                    x = super(module.__class__, module).__call__(x)
                    return x, module

                x = einops.repeat(x, "... -> head ...", head=self.num_heads)
                x, module = jax.vmap(call, axis_name=self.axis_name)(self, x)

                self.merge(module, inplace=True)

                return x

        num_heads = 4
        units = 3

        x = np.random.uniform(size=(5, 2))

        module = Linear3D(num_heads, units).init(42, x)

        y = module(x)

        assert y.shape == (num_heads, 5, units)

    def test_vmap_compact(self):
        @dataclass
        class Linear3D(tx.Module):
            num_heads: int
            units: int

            @tx.preserve_state(jax.vmap, axis_name="head")
            @tx.compact
            def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
                x = tx.Linear(self.units, axis_name="head")(x)
                return x

            def __call__(self, x):
                x = einops.repeat(x, "... -> head ...", head=self.num_heads)
                x = self._forward(x)
                return x

        num_heads = 4
        units = 3

        x = np.random.uniform(size=(5, 2))

        module = Linear3D(num_heads, units).init(42, x)

        assert module.linear.kernel.shape == (num_heads, 2, units)
        assert module.linear.bias.shape == (num_heads, units)

        y = module(x)

        assert y.shape == (num_heads, 5, 3)
