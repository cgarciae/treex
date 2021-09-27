import inspect

import jax
import jax.numpy as jnp
import pytest

import treex as tx


class TestReduce:
    def test_jit(self):
        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metric = tx.metrics.Accuracy()
        y_true = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])

        metric = f(metric, y_true, y_pred)
        metric = f(metric, y_true, y_pred)

        assert metric.count == 20
        assert N == 1
