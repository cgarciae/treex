import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex import losses


class MyModule(tx.Module):
    aux_loss: jnp.ndarray = tx.LossLog.node()
    aux_metric: jnp.ndarray = tx.MetricLog.node()
    some_value: jnp.ndarray = tx.node()

    def __init__(self) -> None:
        self.aux_loss = jnp.array(1.0, jnp.float32)
        self.aux_metric = jnp.array(2.0, jnp.float32)
        self.some_value = jnp.array(10.0, jnp.float32)

    def __call__(self):
        pass


class TestLossAndLogs:
    def test_basic(self):

        N = 0

        @jax.jit
        def f(
            module: MyModule,
            metrics: tx.metrics.LossesAndMetrics,
            target,
            preds,
        ):
            nonlocal N
            N += 1
            return metrics.update(
                target=target,
                preds=preds,
                aux_losses=module.loss_logs().as_logs(),
                aux_metrics=module.metric_logs().as_logs(),
            )

        module = MyModule()

        metrics = tx.metrics.LossesAndMetrics(
            losses=tx.metrics.Losses(
                [
                    tx.losses.MeanSquaredError(),
                    tx.losses.MeanSquaredError(),
                ]
            ).slice(target="losses", preds="losses"),
            metrics=tx.metrics.Metrics(
                dict(
                    a=tx.metrics.Accuracy(num_classes=10),
                    b=tx.metrics.Accuracy(num_classes=10),
                )
            ).slice(target="metrics", preds="metrics"),
            aux_losses=tx.metrics.AuxLosses(),
            aux_metrics=tx.metrics.AuxMetrics(),
        ).reset(
            aux_losses=module.loss_logs().as_logs(),
            aux_metrics=module.metric_logs().as_logs(),
        )
        target = dict(
            losses=jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :],
        )
        preds = dict(
            losses=jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :],
        )

        metrics = f(module, metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {
            "loss": 3.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 1.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 2.0,
        }

        module = module.replace(
            aux_loss=jnp.array(3.0, jnp.float32),
            aux_metric=jnp.array(4.0, jnp.float32),
        )

        metrics = f(module, metrics, target, preds)
        assert N == 1
        assert metrics.compute() == {
            "loss": 4.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 2.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 3.0,
        }

    def test_batch_loss(self):
        N = 0

        @jax.jit
        def f(
            module: MyModule,
            metrics: tx.metrics.LossesAndMetrics,
            target,
            preds,
        ):
            nonlocal N
            N += 1
            return metrics.loss_and_update(
                target=target,
                preds=preds,
                aux_losses=module.loss_logs().as_logs(),
                aux_metrics=module.metric_logs().as_logs(),
            )

        module = MyModule()

        metrics = tx.metrics.LossesAndMetrics(
            losses=tx.metrics.Losses(
                [
                    tx.losses.MeanSquaredError(),
                    tx.losses.MeanSquaredError(),
                ]
            ).slice(target="losses", preds="losses"),
            metrics=tx.metrics.Metrics(
                dict(
                    a=tx.metrics.Accuracy(num_classes=10),
                    b=tx.metrics.Accuracy(num_classes=10),
                )
            ).slice(target="metrics", preds="metrics"),
            aux_losses=tx.metrics.AuxLosses(),
            aux_metrics=tx.metrics.AuxMetrics(),
        ).reset(
            aux_losses=module.loss_logs().as_logs(),
            aux_metrics=module.metric_logs().as_logs(),
        )
        target = dict(
            losses=jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :],
        )
        preds = dict(
            losses=jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :],
            metrics=jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :],
        )

        loss, metrics = f(module, metrics, target, preds)
        assert N == 1
        assert loss == 3.0
        assert metrics.compute() == {
            "loss": 3.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 1.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 2.0,
        }

        module = module.replace(
            aux_loss=jnp.array(3.0, jnp.float32),
            aux_metric=jnp.array(4.0, jnp.float32),
        )

        loss, metrics = f(module, metrics, target, preds)
        assert N == 1
        assert loss == 5.0
        assert metrics.compute() == {
            "loss": 4.0,
            "mean_squared_error": 1.0,
            "mean_squared_error2": 1.0,
            "aux_loss": 2.0,
            "a": 0.8,
            "b": 0.8,
            "aux_metric": 3.0,
        }
