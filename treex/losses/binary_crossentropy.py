import typing as tp

import jax
import jax.numpy as jnp

from treex import types, utils
from treex.losses.loss import Loss, Reduction


def binary_crossentropy(
    target: jnp.ndarray,
    preds: jnp.ndarray,
    from_logits: bool = False,
    label_smoothing: float = 0,
) -> jnp.ndarray:
    assert abs(preds.ndim - target.ndim) <= 1

    target, preds = utils._maybe_expand_dims(target, preds)

    if label_smoothing:
        target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        return -jnp.mean(target * preds - jnp.logaddexp(0.0, preds), axis=-1)

    preds = jnp.clip(preds, types.EPSILON, 1.0 - types.EPSILON)
    return -jnp.mean(
        target * jnp.log(preds) + (1 - target) * jnp.log(1 - preds), axis=-1
    )


class BinaryCrossentropy(Loss):
    """
    Computes the cross-entropy loss between true target and predicted target.
    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.
    In the snippet below, each of the four examples has only a single
    floating-pointing value, and both `preds` and `target` have the shape
    `[batch_size]`.

    Usage:
    ```python
    target = jnp.array([[0., 1.], [0., 0.]])
    preds = jnp.array[[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = tx.losses.BinaryCrossentropy()
    result = bce(target, preds)
    assert np.isclose(result, 0.815, rtol=0.01)

    # Calling with 'sample_weight'.
    bce = tx.losses.BinaryCrossentropy()
    result = bce(target, preds, sample_weight=jnp.array([1, 0]))
    assert np.isclose(result, 0.458, rtol=0.01)

    # Using 'sum' reduction type.
    bce = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.SUM)
    result = bce(target, preds)
    assert np.isclose(result, 1.630, rtol=0.01)

    # Using 'none' reduction type.
    bce = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.NONE)
    result = bce(target, preds)
    assert jnp.all(np.isclose(result, [0.916, 0.713], rtol=0.01))
    ```


    Usage with the `Elegy` API:
    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.BinaryCrossentropy(),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
    )
    ```
    """

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `CategoricalCrossentropy` instance.

        Arguments:
            from_logits: Whether `preds` is expected to be a logits tensor. By
                default, we assume that `preds` encodes a probability distribution.
                **Note - Using from_logits=True is more numerically stable.**
            label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
                meaning the confidence on label values are relaxed. e.g.
                `label_smoothing=0.2` means that we will use a value of `0.1` for label
                `0` and `0.9` for label `1`"
            reduction: (Optional) Type of `tx.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. Indicates that the reduction
                option will be determined by the usage context. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `elegy` `compile` and `fit`, ` or `SUM_OVER_BATCH_SIZE`
                will raise an error.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            on: A string or integer, or iterable of string or integers, that
                indicate how to index/filter the `target` and `preds`
                arguments before passing them to `call`. For example if `on = "a"` then
                `target = target["a"]`. If `on` is an iterable
                the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
                then `target = target["a"][0]["b"]`, same for `preds`. For more information
                check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
        """
        super().__init__(reduction=reduction, weight=weight, on=on, **kwargs)
        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(
        self,
        target: jnp.ndarray,
        preds: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Invokes the `BinaryCrossentropy` instance.

        Arguments:
            target: Ground truth values.
            preds: The predicted values.
            sample_weight: Acts as a
                coefficient for the loss. If a scalar is provided, then the loss is
                simply scaled by the given value. If `sample_weight` is a tensor of size
                `[batch_size]`, then the total loss for each sample of the batch is
                rescaled by the corresponding element in the `sample_weight` vector. If
                the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
                broadcasted to this shape), then each loss element of `preds` is scaled
                by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
                functions reduce by 1 dimension, usually axis=-1.)
        Returns:
            Loss values per sample.
        """

        return binary_crossentropy(
            target,
            preds,
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
        )
