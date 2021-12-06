import typing as tp

import jax.numpy as jnp

from treex import types
from treex.metrics.mean import Mean


def _mean_square_error(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Calculates values required to update/compute Mean Square Error. Cast y_pred to have the same type as y_true.

    Args:
        y_pred: Predicted tensor
        y_true: Ground truth tensor

    Returns:
        jnp.ndarray values needed to update Mean Square Error
    """

    y_true = y_true.astype(y_pred.dtype)
    return jnp.square(y_pred - y_true)


class MeanSquareError(Mean):
    def __init__(
        self,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        `Computes Mean Square Error`_ (MSE):
        .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2
        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

        Args:
            on:
                A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
            name:
                Module name
            dtype:
                Metrics states initialization dtype


        Example:
        >>> import jax.numpy as jnp
        >>> from treex.metrics.mean_square_error import MeanSquareError

        >>> y_true = jnp.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = jnp.array([3.0, -0.5, 2.0, 7.0])

        >>> mse = MeanSquareError()
        >>> mse(y_pred, y_true)

        """
        super().__init__(on=on, name=name, dtype=dtype)

    def update(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: jnp.ndarray = None,
    ) -> tp.Any:
        """
        Accumulates metric statistics. `y_true` and `y_pred` should have the same shape.

        Arguments:
            y_true:
                Ground truth values. shape = `[batch_size, d0, .. dN]`.
            y_pred:
                The predicted values. shape = `[batch_size, d0, .. dN]`
            sample_weight:
                Optional weighting of each example. Defaults to 1. shape = `[batch_size, d0, .. dN]`
        Returns:
            Array with the cumulative mean absolute error.
        """
        values = _mean_square_error(y_pred, y_true)
        return super().update(values, sample_weight)
