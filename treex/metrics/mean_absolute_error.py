import typing as tp

import jax.numpy as jnp

from treex import types
from treex.metrics.mean import Mean


def _mean_absolute_error(y_pred: jnp.ndarray, y_true: jnp.ndarray):
    """
    Computes the mean absolute error between labels and predictions.

    After computing the absolute distance between the inputs, the mean value over
    the last dimension is returned.

    ```python
    loss = mean(abs(y_true - y_pred), axis=-1)
    ```

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    y_pred = jax.random.uniform(rng, shape=(2, 3))
    y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)

    loss = elegy.losses.mean_absolute_error(y_true, y_pred)

    assert loss.shape == (2,)
    assert jnp.array_equal(loss, jnp.mean(jnp.abs(y_true - y_pred), axis=-1))

    ```
    Arguments:
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        
    Returns:
        Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
    """

    y_true = y_true.astype(y_pred.dtype)
    return jnp.mean(jnp.abs(y_pred - y_true), axis=-1)


class MeanAbsolutError(Mean):
    def __init__(
        self,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
        **kwards
    ):
        """
        Creates a `MeanAbsoluteError` instance.

        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `y_true` and `y_pred`
                arguments before passing them to `call`. For example if `on = "a"` then
                `y_true = y_true["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `y_true = y_true["a"][0]["b"]`, same for `y_pred`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).

            kwargs: Additional keyword arguments passed to Module.
            """
        super().__init__(on=on, name=name, dtype=dtype)

    def update(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> tp.Any:
        """
        Accumulates metric statistics. `y_true` and `y_pred` should have the same shape.

        Arguments:
            y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
            y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
            sample_weight: Optional `sample_weight` acts as a
                coefficient for the metric. If a scalar is provided, then the metric is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the metric for each sample of the batch is rescaled
                by the corresponding element in the `sample_weight` vector. If the shape
                of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
                to this shape), then each metric element of `y_pred` is scaled by the
                corresponding value of `sample_weight`. (Note on `dN-1`: all metric
                functions reduce by 1 dimension, usually the last axis (-1)).
        Returns:
            Array with the cumulative mean absolute error.
        """
        values = _mean_absolute_error(self, y_pred, y_true, sample_weight)
        return super().update(values, sample_weight)

    def compute(self):
        return super().compute(self)