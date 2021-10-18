import typing as tp

import jax.numpy as jnp

from treex import types, utils
from treex.metrics.metric import Metric


class Metrics(Metric):
    metrics: tp.Dict[str, Metric]

    def __init__(
        self,
        modules: tp.Any,
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
            for path, metric in utils._flatten_names(modules)
        }

    def update(self, **kwargs) -> None:
        for name, metric in self.metrics.items():
            update_kwargs = utils._function_argument_names(metric.update)

            if update_kwargs is None:
                metric_kwargs = kwargs

            else:
                metric_kwargs = {}

                for arg in update_kwargs:
                    if arg not in kwargs:
                        raise ValueError(f"Missing argument {arg} for metric {name}")

                    metric_kwargs[arg] = kwargs[arg]

            metric.update(**metric_kwargs)

    def compute(self) -> tp.Any:
        outputs = {}
        names = set()

        for name, metric in self.metrics.items():

            value = metric.compute()

            for path, value in utils._flatten_names(value):
                name = f"{name}/{path}" if path else name
                name = utils._unique_name(names, name)

                outputs[name] = value

        return outputs
