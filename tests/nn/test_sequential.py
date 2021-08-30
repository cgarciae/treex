import jax
import numpy as np
import pytest

import treex as tx


class TestSequence:
    def test_basic(self):
        mlp = tx.Sequential(
            tx.Linear(2, 32),
            jax.nn.relu,
            tx.Linear(32, 8),
            jax.nn.relu,
            tx.Linear(8, 4),
        ).init(42)

        assert isinstance(mlp.layers[1], tx.Lambda)
        assert isinstance(mlp.layers[3], tx.Lambda)

        x = np.random.uniform(-1, 1, (10, 2))
        y = mlp(x)

        assert y.shape == (10, 4)

    def test_pytree(self):
        mlp = tx.Sequential(
            tx.Linear(2, 32),
            jax.nn.relu,
            tx.Linear(32, 8),
            jax.nn.relu,
            tx.Linear(8, 4),
        ).init(42)

        jax.tree_map(lambda x: 2 * x, mlp)
