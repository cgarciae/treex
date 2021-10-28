import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to

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


class AuxMetrics(Metric):
    totals: tp.Dict[str, jnp.ndarray] = types.MetricState.node()
    counts: tp.Dict[str, jnp.ndarray] = types.MetricState.node()

    def __init__(
        self,
        aux_metrics: tp.Any,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)
        logs = self.as_logs(aux_metrics)
        self.totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in logs}
        self.counts = {name: jnp.array(0, dtype=jnp.uint32) for name in logs}

    def update(self, aux_metrics: tp.Any) -> None:
        logs = self.as_logs(aux_metrics)

        self.totals = {
            name: (self.totals[name] + logs[name]).astype(self.totals[name].dtype)
            for name in self.totals
        }
        self.counts = {
            name: (self.counts[name] + np.prod(logs[name].shape)).astype(
                self.counts[name].dtype
            )
            for name in self.counts
        }

    def compute(self) -> tp.Dict[str, jnp.ndarray]:
        return {name: self.totals[name] / self.counts[name] for name in self.totals}

    def __call__(self, aux_metrics: tp.Any) -> tp.Dict[str, jnp.ndarray]:
        return super().__call__(aux_metrics=aux_metrics)

    @staticmethod
    def metric_name(field_info: to.FieldInfo) -> str:
        return (
            field_info.value.name
            if isinstance(field_info.value, types.Named)
            else field_info.name
            if field_info.name is not None
            else "aux_metric"
        )

    def as_logs(self, tree: tp.Any) -> tp.Dict[str, jnp.ndarray]:

        names: tp.Set[str] = set()

        with to.add_field_info():
            fields_info: tp.List[to.FieldInfo] = jax.tree_flatten(
                tree,
                is_leaf=lambda x: isinstance(x, types.Named)
                and not isinstance(x.value, to.Nothing),
            )[0]

        # pretend Named values are leaves
        for i, x in enumerate(fields_info):
            if isinstance(x, types.Named):
                field_info = x.value
                field_info.value = types.Named(x.name, field_info.value)
                fields_info[i] = field_info

        metrics = {
            self.metric_name(field_info): field_info.value.value
            if isinstance(field_info.value, types.Named)
            else field_info.value
            for field_info in fields_info
        }
        metrics = {
            utils._unique_name(names, name): value for name, value in metrics.items()
        }

        return metrics
