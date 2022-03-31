import functools
import typing as tp
from abc import abstractmethod

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
        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
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
