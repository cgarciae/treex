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

M = tp.TypeVar("M", bound="LossesAndMetrics")


class LossesAndMetrics(Metric):
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
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(name=name, dtype=dtype)
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

    def reset(
        self: M,
        aux_losses: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
        aux_metrics: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
    ) -> M:
        if self.losses is not None:
            losses = self.losses.reset()
        else:
            losses = None
        if self.metrics is not None:
            metrics = self.metrics.reset()
        else:
            metrics = None

        if self.aux_losses is not None:
            aux_losses_ = self.aux_losses.reset(aux_losses)
        else:
            aux_losses_ = None

        if self.aux_metrics is not None:
            aux_metrics_ = self.aux_metrics.reset(aux_metrics)
        else:
            aux_metrics_ = None

        return self.replace(
            losses=losses,
            metrics=metrics,
            aux_losses=aux_losses_,
            aux_metrics=aux_metrics_,
        )

    def update(
        self: M,
        aux_losses: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
        aux_metrics: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
        **kwargs,
    ) -> M:

        if self.losses is not None:
            losses = self.losses.update(**kwargs)
        else:
            losses = None

        if self.metrics is not None:
            metrics = self.metrics.update(**kwargs)
        else:
            metrics = None

        if self.aux_losses is not None:
            if aux_losses is None:
                raise ValueError("`aux_losses` are expected, got None.")

            aux_losses_ = self.aux_losses.update(aux_values=aux_losses)
        else:
            aux_losses_ = None

        if self.aux_metrics is not None:
            if aux_metrics is None:
                raise ValueError("`aux_metrics` are expected, got None.")

            aux_metrics_ = self.aux_metrics.update(aux_values=aux_metrics)
        else:
            aux_metrics_ = None

        return self.replace(
            losses=losses,
            metrics=metrics,
            aux_losses=aux_losses_,
            aux_metrics=aux_metrics_,
        )

    def compute(self) -> tp.Dict[str, jnp.ndarray]:

        if self.losses is not None:
            losses_logs = self.losses.compute()
        else:
            losses_logs = {}

        if self.metrics is not None:
            metrics_logs = self.metrics.compute()
        else:
            metrics_logs = {}

        if self.aux_losses is not None:
            aux_losses_logs = self.aux_losses.compute()
        else:
            aux_losses_logs = {}

        if self.aux_metrics is not None:
            aux_metrics_logs = self.aux_metrics.compute()
        else:
            aux_metrics_logs = {}

        loss = self.total_loss()

        return {
            "loss": loss,
            **losses_logs,
            **metrics_logs,
            **aux_losses_logs,
            **aux_metrics_logs,
        }

    def compute_logs(self) -> tp.Dict[str, jnp.ndarray]:
        return self.compute()

    def __call__(
        self: M,
        aux_losses: tp.Optional[tp.Any] = None,
        aux_metrics: tp.Optional[tp.Any] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Dict[str, jnp.ndarray], M]:
        return super().__call__(
            aux_losses=aux_losses,
            aux_metrics=aux_metrics,
            **kwargs,
        )

    def total_loss(self) -> jnp.ndarray:
        loss = jnp.array(0.0, dtype=jnp.float32)

        if self.losses is not None:
            loss += self.losses.total_loss()

        if self.aux_losses is not None:
            loss += self.aux_losses.total_loss()

        return loss

    def loss_and_update(self: M, **kwargs) -> tp.Tuple[jnp.ndarray, M]:
        batch_updates = self.batch_updates(**kwargs)
        loss = batch_updates.total_loss()
        metrics = self.merge(batch_updates)

        return loss, metrics
