import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from jax._src.numpy.lax_numpy import ndarray
from jax.core import Value
from treeo.types import Nothing
from treeo.utils import field

from treex import types, utils
from treex.losses.loss import Loss
from treex.metrics.metric import Metric


class Losses(Metric):
    totals: tp.Dict[str, jnp.ndarray] = types.MetricState.node()
    counts: tp.Dict[str, jnp.ndarray] = types.MetricState.node()
    losses: tp.Dict[str, Loss]

    def __init__(
        self,
        losses: tp.Any,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)

        names: tp.Set[str] = set()

        def get_name(path, metric):
            name = utils._get_name(metric)
            return f"{path}/{name}" if path else name

        names_losses = [
            (get_name(path, loss), loss) for path, loss in utils._flatten_names(losses)
        ]
        self.losses = {
            utils._unique_name(
                names, f"{name}_loss" if not name.endswith("loss") else name
            ): loss
            for name, loss in names_losses
        }
        self.totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in self.losses}
        self.counts = {name: jnp.array(0, dtype=jnp.uint32) for name in self.losses}

    def update(self, **kwargs) -> None:
        for name, loss in self.losses.items():
            arg_names = utils._function_argument_names(loss.call)

            if arg_names is None:
                loss_kwargs = kwargs
            else:
                loss_kwargs = {arg: kwargs[arg] for arg in arg_names if arg in kwargs}

            value = loss(**loss_kwargs)

            self.totals[name] = (self.totals[name] + value).astype(jnp.float32)
            self.counts[name] = (self.counts[name] + 1).astype(jnp.uint32)

    def compute(self) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        losses = {name: self.totals[name] / self.counts[name] for name in self.totals}
        total_loss = sum(losses.values(), jnp.array(0.0, dtype=jnp.float32))

        return total_loss, losses

    def __call__(self, **kwargs) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        return super().__call__(**kwargs)


class AuxLosses(Metric):
    totals: tp.Dict[str, jnp.ndarray] = types.MetricState.node()
    counts: tp.Dict[str, jnp.ndarray] = types.MetricState.node()

    def __init__(
        self,
        aux_losses: tp.Any,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)
        logs = self.as_logs(aux_losses)
        self.totals = {name: jnp.array(0.0, dtype=jnp.float32) for name in logs}
        self.counts = {name: jnp.array(0, dtype=jnp.uint32) for name in logs}

    def update(self, aux_losses: tp.Any) -> None:
        logs = self.as_logs(aux_losses)

        self.totals = {
            name: (self.totals[name] + logs[name]).astype(jnp.float32)
            for name in self.totals
        }
        self.counts = {
            name: (self.counts[name] + 1).astype(dtype=jnp.uint32)
            for name in self.counts
        }

    def compute(self) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        losses = {name: self.totals[name] / self.counts[name] for name in self.totals}
        total_loss = sum(losses.values(), jnp.array(0.0, dtype=jnp.float32))

        return total_loss, losses

    def __call__(
        self, aux_losses: tp.Any
    ) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        return super().__call__(aux_losses=aux_losses)

    @staticmethod
    def loss_name(field_info: to.FieldInfo) -> str:
        return (
            field_info.value.name
            if isinstance(field_info.value, types.Named)
            else field_info.name
            if field_info.name is not None
            else "aux_loss"
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

        losses = {
            self.loss_name(field_info): field_info.value.value
            if isinstance(field_info.value, types.Named)
            else field_info.value
            for field_info in fields_info
        }
        losses = {
            utils._unique_name(
                names,
                f"{name}_loss" if not name.endswith("loss") else name,
            ): value
            for name, value in losses.items()
        }

        return losses
