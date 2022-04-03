import functools
import typing as tp
from abc import abstractmethod
from ast import Slice

import jax
import jax.numpy as jnp
import treeo as to
from rich.text import Text

from treex import types, utils
from treex.treex import Treex

M = tp.TypeVar("M", bound="Metric")


class Metric(Treex):
    """
    Encapsulates metric logic and state. Metrics accumulate state between calls such
    that their output value reflect the metric as if calculated on the whole data
    given up to that point.
    """

    def __init__(
        self,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        name: name of the metric
        dtype: dtype of the metric
        """

        self.name = name if name is not None else utils._get_name(self)
        self.dtype = dtype if dtype is not None else jnp.float32

    def __call__(self: M, **kwargs) -> tp.Tuple[tp.Any, M]:
        metric: M = self

        batch_updates = metric.batch_updates(**kwargs)
        batch_values = batch_updates.compute()

        metric = metric.merge(batch_updates)

        return batch_values, metric

    @abstractmethod
    def reset(self: M) -> M:
        ...

    @abstractmethod
    def update(self: M, **kwargs) -> M:
        ...

    @abstractmethod
    def compute(self) -> tp.Any:
        ...

    def batch_updates(self: M, **kwargs) -> M:
        return self.reset().update(**kwargs)

    def aggregate(self: M) -> M:
        return jax.tree_map(lambda x: jnp.sum(x, axis=0), self)

    def merge(self: M, other: M) -> M:
        stacked = jax.tree_map(lambda *xs: jnp.stack(xs), self, other)
        return stacked.aggregate()

    def slice(self, **kwargs: types.IndexLike) -> "SliceParams":
        return SliceParams(self, kwargs)

    def _not_initialized_error(self):
        return ValueError(
            f"Metric '{self.name}' has not been initialized, call 'reset()' first"
        )


Slice = tp.Tuple[tp.Union[int, str], ...]


class SliceParams(Metric):
    arg_slice: tp.Dict[str, Slice]
    metric: Metric = types.MetricState.node()

    def __init__(
        self,
        metric: Metric,
        arg_slice: tp.Dict[str, types.IndexLike],
    ):
        super().__init__(name=metric.name, dtype=metric.dtype)
        self.metric = metric
        self.arg_slice = {
            key: tuple([index])
            if not isinstance(index, (list, tuple))
            else tuple(index)
            for key, index in arg_slice.items()
        }

    def reset(self) -> "SliceParams":
        return self.replace(metric=self.metric.reset())

    def update(self, **kwargs) -> "SliceParams":

        # slice the arguments
        for key, slices in self.arg_slice.items():
            for index in slices:
                kwargs[key] = kwargs[key][index]

        return self.replace(metric=self.metric.update(**kwargs))

    def compute(self) -> tp.Any:
        return self.metric.compute()
