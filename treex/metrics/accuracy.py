import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types
from treex.metrics.metric import Metric


def accuracy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    # [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values(
    #     [y_pred, y_true]
    # )
    # y_pred.shape.assert_is_compatible_with(y_true.shape)

    if y_true.dtype != y_pred.dtype:
        y_pred = y_pred.astype(y_true.dtype)

    return (y_true == y_pred).astype(jnp.float32)


# NOTE: this class is experimental, its is just here to demonstrate and test the Metric API.
# for a more serious use implementation porting torchmetrics.Accuracy would be a good idea:
# https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/classification/accuracy.py
class Accuracy(Metric):

    count: jnp.ndarray = types.ModelState.node()
    total: jnp.ndarray = types.ModelState.node()

    def __init__(
        self,
        argmax_preds: bool = False,
        argmax_labels: bool = False,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(on=on, name=name, dtype=dtype)
        self.argmax_preds = argmax_preds
        self.argmax_labels = argmax_labels

        self.count = jnp.array(0, dtype=jnp.uint64)
        self.total = jnp.array(0.0, dtype=jnp.float64)

    def update(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
    ):
        """ """

        if self.argmax_preds:
            y_pred = jnp.argmax(y_pred, axis=-1)

        if self.argmax_labels:
            y_true = jnp.argmax(y_true, axis=-1)

        values = accuracy(y_true=y_true, y_pred=y_pred)

        self.count = (self.count + np.prod(values.shape)).astype(self.count.dtype)
        self.total = (self.total + jnp.sum(values)).astype(self.total.dtype)

    def compute(self) -> tp.Any:
        return self.total / self.count
