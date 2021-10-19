import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.numpy.lax_numpy import ndarray

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

        names_losses = (
            (get_name(path, loss), loss) for path, loss in utils._flatten_names(losses)
        )
        self.losses = {
            utils._unique_name(
                names, f"{name}_loss" if not name.endswith("_loss") else name
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

            self.totals[name] = (self.totals[name] + value).astype(
                self.totals[name].dtype
            )
            self.counts[name] = (self.counts[name] + np.prod(value.shape)).astype(
                self.counts[name].dtype
            )

    def compute(self) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        losses = {
            name: total / count
            for name, total, count in zip(
                self.losses.keys(),
                self.totals.values(),
                self.counts.values(),
            )
        }
        total_loss = sum(losses.values(), jnp.array(0.0, dtype=jnp.float32))

        return total_loss, losses

    def __call__(self, **kwargs) -> tp.Tuple[jnp.ndarray, tp.Dict[str, jnp.ndarray]]:
        return super().__call__(**kwargs)
