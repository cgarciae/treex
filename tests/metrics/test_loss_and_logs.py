import jax
import jax.numpy as jnp
import pytest

import treex as tx
from treex import losses


class TestLossAndLogs:
    def test_basic(self):
        class MyModule(tx.Module):
            aux_loss: jnp.ndarray = tx.LossLog.node()
            aux_metric: jnp.ndarray = tx.MetricLog.node()
            some_value: jnp.ndarray = tx.node()

            def __init__(self) -> None:
                self.aux_loss = jnp.array(1.0, jnp.float32)
                self.aux_metric = jnp.array(2.0, jnp.float32)
                self.some_value = jnp.array(10.0, jnp.float32)

        N = 0

        @jax.jit
        def f(
            module: MyModule,
            metrics: tx.metrics.LossAndLogs,
            target,
            preds,
            y_true_metrics,
            y_pred_metrics,
        ):
            nonlocal N
            N += 1
            metrics(
                target=target,
                preds=preds,
                metrics_kwargs=dict(
                    target=y_true_metrics,
                    preds=y_pred_metrics,
                ),
                aux_losses=module.loss_logs(),
                aux_metrics=module.metric_logs(),
            )
            return metrics

        module = MyModule()

        metrics = tx.metrics.LossAndLogs(
            losses=[
                tx.losses.MeanSquaredError(),
                tx.losses.MeanSquaredError(),
            ],
            metrics=dict(
                a=tx.metrics.Accuracy(num_classes=10),
                b=tx.metrics.Accuracy(num_classes=10),
            ),
            aux_losses=tx.metrics.AuxLosses(module.loss_logs()),
            aux_metrics=tx.metrics.AuxMetrics(module.metric_logs()),
        )
        target = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]
        y_true_metrics = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        y_pred_metrics = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        metrics = f(module, metrics, target, preds, y_true_metrics, y_pred_metrics)
        assert N == 1
        assert metrics.compute() == (
            3.0,
            {
                "loss": 3.0,
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
                "aux_loss": 1.0,
            },
            {
                "a/accuracy": 0.8,
                "b/accuracy": 0.8,
                "aux_metric": 2.0,
            },
        )

        module.aux_loss = jnp.array(3.0, jnp.float32)
        module.aux_metric = jnp.array(4.0, jnp.float32)

        metrics = f(module, metrics, target, preds, y_true_metrics, y_pred_metrics)
        assert N == 1
        assert metrics.compute() == (
            4.0,
            {
                "loss": 4.0,
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
                "aux_loss": 2.0,
            },
            {
                "a/accuracy": 0.8,
                "b/accuracy": 0.8,
                "aux_metric": 3.0,
            },
        )

    def test_batch_loss(self):
        class MyModule(tx.Module):
            aux_loss: jnp.ndarray = tx.LossLog.node()
            aux_metric: jnp.ndarray = tx.MetricLog.node()
            some_value: jnp.ndarray = tx.node()

            def __init__(self) -> None:
                self.aux_loss = jnp.array(1.0, jnp.float32)
                self.aux_metric = jnp.array(2.0, jnp.float32)
                self.some_value = jnp.array(10.0, jnp.float32)

        N = 0

        @jax.jit
        def f(
            module: MyModule,
            metrics: tx.metrics.LossAndLogs,
            target,
            preds,
            y_true_metrics,
            y_pred_metrics,
        ):
            nonlocal N
            N += 1
            loss, losses_logs, metrics_logs = metrics.batch_loss_epoch_logs(
                target=target,
                preds=preds,
                metrics_kwargs=dict(
                    target=y_true_metrics,
                    preds=y_pred_metrics,
                ),
                aux_losses=module.loss_logs(),
                aux_metrics=module.metric_logs(),
            )
            logs = {**losses_logs, **metrics_logs}
            return loss, logs, metrics

        module = MyModule()

        metrics = tx.metrics.LossAndLogs(
            losses=[
                tx.losses.MeanSquaredError(),
                tx.losses.MeanSquaredError(),
            ],
            metrics=dict(
                a=tx.metrics.Accuracy(num_classes=10),
                b=tx.metrics.Accuracy(num_classes=10),
            ),
            aux_losses=tx.metrics.AuxLosses(module.loss_logs()),
            aux_metrics=tx.metrics.AuxMetrics(module.metric_logs()),
        )
        target = jnp.array([0.0, 0.0, 0.0, 0.0])[None, None, None, :]
        preds = jnp.array([1.0, 1.0, 1.0, 1.0])[None, None, None, :]
        y_true_metrics = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[None, None, None, :]
        y_pred_metrics = jnp.array([0, 1, 2, 3, 0, 5, 6, 7, 0, 9])[None, None, None, :]

        loss, logs, metrics = f(
            module, metrics, target, preds, y_true_metrics, y_pred_metrics
        )
        assert N == 1
        assert (loss, logs) == (
            3.0,
            {
                "loss": 3.0,
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
                "a/accuracy": 0.8,
                "b/accuracy": 0.8,
                "aux_loss": 1.0,
                "aux_metric": 2.0,
            },
        )

        module.aux_loss = jnp.array(3.0, jnp.float32)
        module.aux_metric = jnp.array(4.0, jnp.float32)

        loss, logs, metrics = f(
            module, metrics, target, preds, y_true_metrics, y_pred_metrics
        )
        assert N == 1
        assert (loss, logs) == (
            5.0,
            {
                "loss": 4.0,
                "mean_squared_error_loss": 1.0,
                "mean_squared_error_loss2": 1.0,
                "a/accuracy": 0.8,
                "b/accuracy": 0.8,
                "aux_loss": 2.0,
                "aux_metric": 3.0,
            },
        )
