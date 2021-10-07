import inspect

import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex.metrics.classification.accuracy import Accuracy
from treex.metrics.utilities.enums import DataType


class TestAccuracy:
    def test_jit(self):
        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metric = Accuracy(num_classes=10)
        y_true = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y_pred = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])

        metric = f(metric, y_true, y_pred)
        assert N == 1
        assert metric.compute() == 0.8

        metric = f(metric, y_true, y_pred)
        # assert N == 1
        assert metric.compute() == 0.8

    def test_logits_preds(self):
        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metric = Accuracy()
        y_true = jnp.array([0, 0, 1, 1, 1])
        y_pred = jnp.array(
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [40.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )

        metric0 = metric
        metric = f(metric, y_true, y_pred)
        assert N == 1
        assert metric.compute() == 0.8

        metric = f(metric, y_true, y_pred)
        # assert N == 1
        assert metric.compute() == 0.8
