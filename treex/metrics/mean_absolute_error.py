import typing as tp

import jax.numpy as jnp

from treex import types
from treex.metrics.mean import Mean


def _mean_absolute_error(preds: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Calculates values required to update/compute Mean Absolute Error. Cast preds to have the same type as target.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    Returns:
        jnp.ndarray values needed to update Mean Absolute Error
    """

    target = target.astype(preds.dtype)
    return jnp.abs(preds - target)


class MeanAbsoluteError(Mean):
    def __init__(
        self,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        """
        `Computes Mean Absolute Error`_ (MAE):
        .. math:: \text{MAE} = \frac{1}{N}\sum_i^N | y_i - \hat{y_i} |
        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

        Args:
            on:
                A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
            name:
                Module name
            dtype:
                Metrics states initialization dtype


        Example:
        >>> import jax.numpy as jnp
        >>> from treex.metrics.mean_absolute_error import MeanAbsolutError

        >>> target = jnp.array([3.0, -0.5, 2.0, 7.0])
        >>> preds = jnp.array([3.0, -0.5, 2.0, 7.0])

        >>> mae = MeanAbsolutError()
        >>> mae(preds, target)

        """
        super().__init__(on=on, name=name, dtype=dtype)

    def update(
        self,
        target: jnp.ndarray,
        preds: jnp.ndarray,
        sample_weight: jnp.ndarray = None,
    ) -> tp.Any:
        """
        Accumulates metric statistics. `target` and `preds` should have the same shape.

        Arguments:
            target:
                Ground truth values. shape = `[batch_size, d0, .. dN]`.
            preds:
                The predicted values. shape = `[batch_size, d0, .. dN]`
            sample_weight:
                Optional weighting of each example. Defaults to 1. shape = `[batch_size, d0, .. dN]`
        Returns:
            Array with the cumulative mean absolute error.
        """
        values = _mean_absolute_error(preds, target)
        return super().update(values, sample_weight)
