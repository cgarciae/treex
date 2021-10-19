import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex import metrics


class TestLosses:
    def test_list(self):

        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metrics = tx.metrics.Losses(
            [
                tx.losses.MeanSquaredError(),
                tx.losses.MeanSquaredError(),
            ]
        )
        y_true = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        y_pred = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == (
            2.0,
            {
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
            },
        )

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == (
            2.0,
            {
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
            },
        )

    def test_dict(self):

        N = 0

        @jax.jit
        def f(m, y_true, y_pred):
            nonlocal N
            N += 1
            m(y_true=y_true, y_pred=y_pred)
            return m

        metrics = tx.metrics.Losses(
            dict(
                a=tx.losses.MeanSquaredError(),
                b=tx.losses.MeanSquaredError(),
            )
        )
        y_true = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        y_pred = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == (
            2.0,
            {
                "a/mean_squared_error_loss": 1.0,
                "b/mean_squared_error_loss": 1.0,
            },
        )

        metrics = f(metrics, y_true, y_pred)
        assert N == 1
        assert metrics.compute() == (
            2.0,
            {
                "a/mean_squared_error_loss": 1.0,
                "b/mean_squared_error_loss": 1.0,
            },
        )
