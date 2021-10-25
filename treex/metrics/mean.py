import typing as tp

import jax.numpy as jnp

from treex import types
from treex.metrics.reduce import Reduce, Reduction


class Mean(Reduce):
    """
    Computes the (weighted) mean of the given values.

    For example, if values is `[1, 3, 5, 7]` then the mean is `4`.
    If the weights were specified as `[1, 1, 0, 0]` then the mean would be `2`.
    This metric creates two variables, `total` and `count` that are used to
    compute the average of `values`. This average is ultimately returned as `mean`
    which is an idempotent operation that simply divides `total` by `count`.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Usage:

    ```python
    mean = elegy.metrics.Mean()
    result = mean([1, 3, 5, 7])  # 16 / 4
    assert result == 4.0


    result = mean([4, 10])  # 30 / 6
    assert result == 5.0
    ```

    Usage with elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.MeanSquaredError(),
        metrics=elegy.metrics.Mean(),
    )
    ```
    """

    def __init__(
        self,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        """Creates a `Mean` instance.
        Arguments:
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
            kwargs: Additional keyword arguments passed to Module.
        """
        super().__init__(
            reduction=Reduction.weighted_mean,
            on=on,
            name=name,
            dtype=dtype,
        )

    def update(
        self,
        values: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ):
        """
        Accumulates the mean statistic over various batches.

        Arguments:
            values: Per-example value.
            sample_weight: Optional weighting of each example.

        Returns:
            Array with the cumulative mean.
        """

        super().update(
            values=values,
            sample_weight=sample_weight,
        )
