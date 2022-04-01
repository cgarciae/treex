import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to

from treex import types, utils
from treex.metrics.metric import Metric

M = tp.TypeVar("M", bound="Metrics")
A = tp.TypeVar("A", bound="AuxMetrics")


class Metrics(Metric):
    metrics: tp.Dict[str, Metric]

    def __init__(
        self,
        metrics: tp.Any,
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

        self.metrics = {
            utils._unique_name(names, get_name(path, metric, parent_iterable)): metric
            for path, metric, parent_iterable in utils._flatten_names(metrics)
        }

    def reset(self: M) -> M:
        metrics = {name: metric.reset() for name, metric in self.metrics.items()}
        return self.replace(metrics=metrics)

    def update(self: M, **kwargs) -> M:
        metrics = {
            name: metric.update(**kwargs) for name, metric in self.metrics.items()
        }
        return self.replace(metrics=metrics)

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        outputs = {}
        names = set()

        for name, metric in self.metrics.items():

            value = metric.compute()

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


class AuxMetrics(Metric):
    totals: tp.Optional[tp.Dict[str, jnp.ndarray]] = types.MetricState.node()
    counts: tp.Optional[tp.Dict[str, jnp.ndarray]] = types.MetricState.node()

    def __init__(
        self,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(name=name, dtype=dtype)
        self.totals = None
        self.counts = None

    def reset(self: A, aux_values: tp.Optional[tp.Dict[str, jnp.ndarray]] = None) -> A:
        if self.totals is None or self.counts is None:
            if aux_values is None:
                raise ValueError(
                    "'aux_values' must be provided on the first call to 'reset()"
                )

            names = aux_values.keys()
        else:
            names = self.totals.keys()

        totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in names}
        counts = {name: jnp.array(0, dtype=jnp.uint32) for name in names}

        return self.replace(totals=totals, counts=counts)

    def update(self: A, aux_values: tp.Dict[str, jnp.ndarray], **_) -> A:
        if self.totals is None or self.counts is None:
            raise ValueError("AuxMetrics not initialized, call 'reset()' first")

        totals = {
            name: (self.totals[name] + aux_values[name]).astype(self.totals[name].dtype)
            for name in self.totals
        }
        counts = {
            name: (self.counts[name] + np.prod(aux_values[name].shape)).astype(
                self.counts[name].dtype
            )
            for name in self.counts
        }

        return self.replace(totals=totals, counts=counts)

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        if self.totals is None or self.counts is None:
            raise ValueError("AuxMetrics not initialized, call `reset()` first")

        return {name: self.totals[name] / self.counts[name] for name in self.totals}

    def __call__(self: A, aux_values: tp.Any) -> tp.Tuple[tp.Dict[str, jnp.ndarray], A]:
        return super().__call__(aux_values=aux_values)
