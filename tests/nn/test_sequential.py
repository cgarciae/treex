import jax
import numpy as np
import pytest

import treex as tx


class TestSequence:
    def test_basic(self):
        x = np.random.uniform(-1, 1, (5, 2))

        mlp = tx.Sequential(
            tx.Linear(3),
            jax.nn.relu,
            tx.Linear(2),
            jax.nn.relu,
            tx.Linear(1),
        ).init(key=42)(x)

        assert isinstance(mlp.layers[1], tx.Lambda)
        assert isinstance(mlp.layers[3], tx.Lambda)

        y = mlp(x)

        assert y.shape == (5, 1)

    def test_pytree(self):
        x = np.random.uniform(-1, 1, (5, 2))

        mlp = tx.Sequential(
            tx.Linear(3),
            jax.nn.relu,
            tx.Linear(2),
            jax.nn.relu,
            tx.Linear(1),
        ).init(key=42)(x)

        leaves = jax.tree_leaves(mlp)

        assert len(leaves) == 6
