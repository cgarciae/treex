import typing as tp

import jax.numpy as jnp

from treex import types, utils
from treex.metrics.metric import Metric


class Metrics(Metric):
    metrics: tp.Dict[str, Metric]

    def __init__(
        self,
        metrics: tp.Any,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)

        names: tp.Set[str] = set()

        def get_name(path, metric):
            name = utils._get_name(metric)
            return f"{path}/{name}" if path else name

        self.metrics = {
            utils._unique_name(names, get_name(path, metric)): metric
            for path, metric in utils._flatten_names(metrics)
        }

    def update(self, **kwargs) -> None:
        for name, metric in self.metrics.items():
            arg_names = utils._function_argument_names(metric.update)

            if arg_names is None:
                metric_kwargs = kwargs
            else:
                metric_kwargs = {arg: kwargs[arg] for arg in arg_names if arg in kwargs}

            metric.update(**metric_kwargs)

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        outputs = {}
        names = set()

        for name, metric in self.metrics.items():

            value = metric.compute()

            for path, value in utils._flatten_names(value):
                name = f"{name}/{path}" if path else name
                name = utils._unique_name(names, name)

                outputs[name] = value

        return outputs

    def __call__(self, **kwargs) -> tp.Dict[str, jnp.ndarray]:
        return super().__call__(**kwargs)
