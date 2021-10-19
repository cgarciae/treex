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

Logs = tp.Dict[str, jnp.ndarray]


class LossAndLogs(Metric):
    losses: tp.Optional[Losses]
    metrics: tp.Optional[Metrics]
    aux_losses: tp.Optional[AuxLosses]
    aux_metrics: tp.Optional[AuxMetrics]

    def __init__(
        self,
        losses: tp.Optional[tp.Union[Losses, tp.Any]] = None,
        metrics: tp.Optional[tp.Union[Metrics, tp.Any]] = None,
        aux_losses: tp.Optional[tp.Union[AuxLosses, tp.Any]] = None,
        aux_metrics: tp.Optional[tp.Union[AuxMetrics, tp.Any]] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)
        self.losses = (
            losses
            if isinstance(losses, Losses)
            else Losses(losses)
            if losses is not None
            else None
        )
        self.metrics = (
            metrics
            if isinstance(metrics, Metrics)
            else Metrics(metrics)
            if metrics is not None
            else None
        )
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

        if self.losses is not None:
            self.losses.update(**losses_kwargs)

        if self.metrics is not None:
            self.metrics.update(**metrics_kwargs)

        if self.aux_losses is not None:
            if aux_losses is None:
                raise ValueError("`aux_losses` are expected, got None.")

            self.aux_losses.update(aux_losses)

        if self.aux_metrics is not None:
            if aux_metrics is None:
                raise ValueError("`aux_metrics` are expected, got None.")

            self.aux_metrics.update(aux_metrics)

    def compute(self) -> tp.Tuple[jnp.ndarray, Logs, Logs]:

        if self.losses is not None:
            loss, losses_logs = self.losses.compute()
        else:
            loss = jnp.zeros(0.0, dtype=jnp.float32)
            losses_logs = {}

        if self.metrics is not None:
            metrics_logs = self.metrics.compute()
        else:
            metrics_logs = {}

        if self.aux_losses is not None:
            aux_loss, aux_losses_logs = self.aux_losses.compute()

            losses_logs.update(aux_losses_logs)
            loss += aux_loss

        if self.aux_metrics is not None:
            aux_metrics_logs = self.aux_metrics.compute()
            metrics_logs.update(aux_metrics_logs)

        losses_logs = {"loss": loss, **losses_logs}

        return loss, losses_logs, metrics_logs

    def __call__(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        aux_losses: tp.Optional[tp.Any] = None,
        aux_metrics: tp.Optional[tp.Any] = None,
        **losses_kwargs,
    ) -> tp.Tuple[jnp.ndarray, Logs, Logs]:
        return super().__call__(
            metrics_kwargs=metrics_kwargs,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
            **losses_kwargs,
        )

    def batch_loss_epoch_logs(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        aux_losses: tp.Optional[tp.Any] = None,
        aux_metrics: tp.Optional[tp.Any] = None,
        **losses_kwargs,
    ) -> tp.Tuple[jnp.ndarray, Logs, Logs]:
        batch_loss, *_ = self(
            metrics_kwargs=metrics_kwargs,
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
            **losses_kwargs,
        )
        epoch_loss, losses_logs, metrics_logs = self.compute()

        return batch_loss, losses_logs, metrics_logs
