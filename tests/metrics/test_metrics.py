import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex import metrics


class TestAccuracy:
    def test_list(self):

        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metrics = tx.metrics.Metrics(
            [
                tx.metrics.Accuracy(num_classes=10),
                tx.metrics.Accuracy(num_classes=10),
            ]
        )
        y_true = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        y_pred = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == {"accuracy": 0.8, "accuracy2": 0.8}

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == {"accuracy": 0.8, "accuracy2": 0.8}

    def test_dict(self):

        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metrics = tx.metrics.Metrics(
            dict(
                a=tx.metrics.Accuracy(num_classes=10),
                b=tx.metrics.Accuracy(num_classes=10),
            )
        )
        y_true = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        y_pred = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == {"a/accuracy": 0.8, "b/accuracy": 0.8}

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == {"a/accuracy": 0.8, "b/accuracy": 0.8}
