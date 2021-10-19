import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import metrics, types, utils
from treex.losses.loss import Loss
from treex.metrics.losses import AuxLosses, Losses
from treex.metrics.metric import Metric
from treex.metrics.metrics import AuxMetrics, Metrics
from treex.treex import Treex


class Logs(Metric):
    losses: Losses
    metrics: Metrics
    aux_losses: tp.Optional[AuxLosses]
    aux_metrics: tp.Optional[AuxMetrics]

    def __init__(
        self,
        losses: tp.Union[Losses, tp.Any],
        metrics: tp.Union[Metrics, tp.Any],
        aux_losses: tp.Optional[tp.Union[AuxLosses, tp.Any]] = None,
        aux_metrics: tp.Optional[tp.Union[AuxMetrics, tp.Any]] = None,
    ):
        self.losses = losses if isinstance(losses, Losses) else Losses(losses)
        self.metrics = metrics if isinstance(metrics, Metrics) else Metrics(metrics)
        self.aux_losses = (
            aux_losses
            if isinstance(aux_losses, AuxLosses)
            else AuxLosses(aux_losses)
            if aux_losses is not None
            else None
        )
        self.aux_metrics = (
            aux_metrics
            if isinstance(aux_metrics, AuxMetrics)
            else AuxMetrics(aux_metrics)
            if aux_metrics is not None
            else None
        )

    def update(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        aux_losses: tp.Optional[tp.Any] = None,
        aux_metrics: tp.Optional[tp.Any] = None,
        **losses_kwargs,
    ) -> None:

        if metrics_kwargs is None:
            metrics_kwargs = losses_kwargs

        self.losses.update(**losses_kwargs)
        self.metrics.update(**metrics_kwargs)

        if self.aux_losses is not None:
            if aux_losses is None:
                raise ValueError("`aux_losses` are expected, got None.")

            self.aux_losses.update(aux_losses)

        if self.aux_metrics is not None:
            if aux_metrics is None:
                raise ValueError("`aux_metrics` are expected, got None.")

            self.aux_metrics.update(aux_metrics)

    def compute(self) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:

        loss, losses_logs = self.losses.compute()
        metrics_logs = self.metrics.compute()

        logs = {**losses_logs, **metrics_logs}

        if self.aux_losses is not None:
            aux_loss, aux_losses_logs = self.aux_losses.compute()

            logs.update(aux_losses_logs)
            loss += aux_loss

        if self.aux_metrics is not None:
            aux_metrics_logs = self.aux_metrics.compute()
            logs.update(aux_metrics_logs)

        # add loss as first entry in logs
        logs = {"loss": loss, **logs}

        return loss, logs

    def __call__(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        aux_losses: tp.Optional[tp.Any] = None,
        aux_metrics: tp.Optional[tp.Any] = None,
        **losses_kwargs,
    ) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        return super().__call__(
            metrics_kwargs=metrics_kwargs,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
            **losses_kwargs,
            **losses_kwargs,
        )

    def batch_loss_epoch_logs(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        aux_logs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **losses_kwargs,
    ) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:

        batch_loss, batch_logs = self(**losses_kwargs)
        epoch_loss, epoch_logs = self.compute()

        return batch_loss, epoch_logs
