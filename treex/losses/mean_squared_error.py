import typing as tp

import jax.numpy as jnp

from treex import types, utils
from treex.losses.loss import Loss, Reduction


def mean_squared_error(target: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the mean squared error between target and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    ```python
    loss = mean(square(target - preds), axis=-1)
    ```

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = tx.losses.mean_squared_error(target, preds)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, jnp.mean(jnp.square(target - preds), axis=-1))
    ```

    Arguments:
        target: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        preds: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """

    target = target.astype(preds.dtype)

    return jnp.mean(jnp.square(preds - target), axis=-1)


class MeanSquaredError(Loss):
    """
    Computes the mean of squares of errors between target and predictions.

    `loss = square(target - preds)`

    Usage:

    ```python
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    mse = tx.losses.MeanSquaredError()

    assert mse(target, preds) == 0.5

    # Calling with 'sample_weight'.
    assert mse(target, preds, sample_weight=jnp.array([0.7, 0.3])) == 0.25

    # Using 'sum' reduction type.
    mse = tx.losses.MeanSquaredError(reduction=tx.losses.Reduction.SUM)

    assert mse(target, preds) == 1.0

    # Using 'none' reduction type.
    mse = tx.losses.MeanSquaredError(reduction=tx.losses.Reduction.NONE)

    assert list(mse(target, preds)) == [0.5, 0.5]
    ```
    Usage with the Elegy API:

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
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):
        """
        Initializes `Mean` class.

        Arguments:
            reduction: (Optional) Type of `tx.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """

        return super().__init__(reduction=reduction, weight=weight, on=on, name=name)

    def call(
        self,
        target: jnp.ndarray,
        preds: jnp.ndarray,
        sample_weight: tp.Optional[
            jnp.ndarray
        ] = None,  # not used, __call__ handles it, left for documentation purposes.
    ) -> jnp.ndarray:
        """
        Invokes the `MeanSquaredError` instance.

        Arguments:
            target: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
            preds: The predicted values. shape = `[batch_size, d0, .. dN]`
            sample_weight: Optional `sample_weight` acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `preds` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)

        Returns:
            Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
                shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
                because all loss functions reduce by 1 dimension, usually axis=-1.)

        Raises:
            ValueError: If the shape of `sample_weight` is invalid.
        """
        return mean_squared_error(target, preds)
