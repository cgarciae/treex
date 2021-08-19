import unittest

import jax
import numpy as np
import treex as tx


class BatchNormTest(unittest.TestCase):
    def test_call(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        y = module(x, use_running_average=False)

        assert y.shape == (10, 2)

    def test_tree(self):
        x = np.random.uniform(size=(10, 2))
        module = tx.nn.BatchNorm(2).init(42)

        flat = jax.tree_leaves(module)

        assert len(flat) == 4

    def test_slice(self):
        x = np.random.uniform(size=(10, 2))
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
            return module, module(x, use_running_average=False)

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
