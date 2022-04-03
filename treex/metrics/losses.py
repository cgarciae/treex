import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to

from treex import types, utils
from treex.losses.loss import Loss
from treex.metrics.metric import Metric
from treex.metrics.metrics import AuxMetrics

M = tp.TypeVar("M", bound="Losses")
A = tp.TypeVar("A", bound="AuxLosses")


class Losses(Metric):
    losses: tp.Dict[str, Loss] = to.static()
    totals: tp.Optional[tp.Dict[str, jnp.ndarray]] = types.MetricState.node()
    counts: tp.Optional[tp.Dict[str, jnp.ndarray]] = types.MetricState.node()

    def __init__(
        self,
        losses: tp.Any,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(name=name, dtype=dtype)

        names: tp.Set[str] = set()

        def get_name(path, metric, parent_iterable):
            name = utils._get_name(metric)
            if path:
                if parent_iterable:
                    return f"{path}/{name}"
                else:
                    return path
            else:
                return name

        self.losses = {
            utils._unique_name(names, get_name(path, loss, parent_iterable)): loss
            for path, loss, parent_iterable in utils._flatten_names(losses)
        }
        self.totals = None
        self.counts = None

    def reset(self: M) -> M:
        totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.losses}
        counts = {name: jnp.array(0, dtype=jnp.uint32) for name in self.losses}
        return self.replace(totals=totals, counts=counts)

    def update(self: M, **kwargs) -> M:
        if self.totals is None or self.counts is None:
            raise ValueError("Losses not initialized, call 'reset()' first")

        totals = {
            name: self.totals[name] + loss(**kwargs)
            for name, loss in self.losses.items()
        }
        counts = {name: self.counts[name] + 1 for name in self.losses.keys()}

        return self.replace(totals=totals, counts=counts)

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        if self.totals is None or self.counts is None:
            raise ValueError("Losses not initialized, call 'reset()' first")

        outputs = {}
        names = set()

        for name in self.losses.keys():

            value = self.totals[name] / self.counts[name]

            for path, value, parent_iterable in utils._flatten_names(value):
                name = utils._unique_name(names, name)

                if path:
                    if parent_iterable:
                        name = f"{path}/{name}"
                    else:
                        name = path

                outputs[name] = value

        return outputs

    def __call__(self: M, **kwargs) -> tp.Tuple[tp.Dict[str, jnp.ndarray], M]:
        return super().__call__(**kwargs)

    def total_loss(self) -> jnp.ndarray:
        return sum(self.compute().values(), jnp.array(0.0))

    def slice(self, **kwargs: types.IndexLike) -> "Losses":
        losses = {name: loss.slice(**kwargs) for name, loss in self.losses.items()}
        return self.replace(losses=losses)

    def loss_and_update(self: M, **kwargs) -> tp.Tuple[jnp.ndarray, M]:
        batch_updates = self.batch_updates(**kwargs)
        loss = batch_updates.total_loss()
        metrics = self.merge(batch_updates)

        return loss, metrics


class AuxLosses(AuxMetrics):
    def total_loss(self) -> jnp.ndarray:
        return sum(self.compute().values(), jnp.array(0.0))

    def loss_and_update(self: A, **kwargs) -> tp.Tuple[jnp.ndarray, A]:
        batch_updates = self.batch_updates(**kwargs)
        loss = batch_updates.total_loss()
        metrics = self.merge(batch_updates)

        return loss, metrics
