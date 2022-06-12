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


class TestEmbed:
    @hp.given(
        batch_size=st.integers(min_value=1, max_value=32),
        length=st.integers(min_value=1, max_value=32),
        num_embeddings=st.integers(min_value=1, max_value=32),
        training=st.booleans(),
        features_out=st.integers(min_value=1, max_value=32),  #
        embedding_init=st.sampled_from(KERNEL_INITS),
    )
    @hp.settings(deadline=None, max_examples=20)
    def test_equivalence(
        self,
        batch_size,
        length,
        num_embeddings,
        training,
        features_out,
        embedding_init,
    ):
        shape = (batch_size, length, length)

        x = np.random.randint(num_embeddings, size=shape)

        key = tx.Key(42)

        flax_module = linen.Embed(
            num_embeddings=num_embeddings,
            features=features_out,
            embedding_init=embedding_init,
        )
        treex_module = tx.Embed(
            num_embeddings=num_embeddings,
            features=features_out,
            embedding_init=embedding_init,
        ).train(training)

        flax_key, _ = tx.iter_split(key)  # emulate init split
        variables = flax_module.init(flax_key, x)
        treex_module = treex_module.init(key=key)(x)

        assert np.allclose(variables["params"]["embedding"], treex_module.embedding)

        y_flax, updates = flax_module.apply(variables, x, mutable=["batch_stats"])
        variables = variables.copy(updates)

        y_treex = treex_module(x)

        assert np.allclose(y_flax, y_treex)

        assert np.allclose(variables["params"]["embedding"], treex_module.embedding)

    def test_call(self):
        x = np.random.randint(2, size=(10,))
        module = tx.Embed(2, 3).init(key=42)(x)

        y = module(x)

        assert y.shape == (10, 3)

    def test_tree(self):
        x = np.random.randint(2, size=(10,))
        module = tx.Embed(2, 3).init(key=42)(x)

        flat = jax.tree_leaves(module)

        assert len(flat) == 1

    def test_slice(self):
        x = np.random.randint(2, size=(10,))
        module = tx.Embed(2, 3).init(key=42)(x)

        flat = jax.tree_leaves(module.filter(tx.Parameter))

        assert len(flat) == 1

        flat = jax.tree_leaves(module.filter(tx.State))

        assert len(flat) == 0

    def test_jit(self):
        x = np.random.randint(2, size=(10,))
        module = tx.Embed(2, 3).init(key=42)(x)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 3)
        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(module), jax.tree_leaves(module2))
        )
