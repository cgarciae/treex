import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import metrics, types, utils
from treex.losses.loss import Loss
from treex.metrics.losses import Losses
from treex.metrics.metric import Metric
from treex.metrics.metrics import Metrics
from treex.treex import Treex


class LossesAndMetrics(Treex):
    losses: Losses
    metrics: Metrics

    def __init__(
        self,
        losses: tp.Union[Losses, tp.Any],
        metrics: tp.Union[Metrics, tp.Any],
    ):
        self.losses = losses if isinstance(losses, Losses) else Losses(losses)
        self.metrics = metrics if isinstance(metrics, Metrics) else Metrics(metrics)

    def __call__(
        self,
        metrics_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        **loss_kwargs,
    ) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:

        if metrics_kwargs is None:
            metrics_kwargs = loss_kwargs

        batch_loss, _ = self.losses(**loss_kwargs)
        total_loss, loss_logs = self.losses.compute()

        self.metrics(**metrics_kwargs)
        metrics_logs = self.metrics.compute()

        logs = {"loss": total_loss, **loss_logs, **metrics_logs}

        return batch_loss, logs
