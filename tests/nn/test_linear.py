import unittest

import jax

import treex as tx
import numpy as np


class LinearTest(unittest.TestCase):
    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.Linear(2, 3).init(42)

        y = module(x)

        assert y.shape == (10, 3)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.Linear(2, 3).init(42)

        flat = jax.tree_leaves(module)

        assert len(flat) == 2

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.Linear(2, 3).init(42)

        flat = jax.tree_leaves(module.slice(tx.Parameter))

        assert len(flat) == 2

        flat = jax.tree_leaves(module.slice(tx.State))

        assert len(flat) == 0

    def test_jit(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.Linear(2, 3).init(42)

        @jax.jit
        def f(module, x):
            return module, module(x)

        module2, y = f(module, x)

        assert y.shape == (10, 3)
        assert all(
            np.allclose(a, b)
            for a, b in zip(jax.tree_leaves(module), jax.tree_leaves(module2))
        )
