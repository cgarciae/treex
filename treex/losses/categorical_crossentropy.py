import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types, utils
from treex.losses.loss import Loss, Reduction


def smooth_labels(
    target: jnp.ndarray,
    smoothing: jnp.ndarray,
) -> jnp.ndarray:
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / target.shape[-1]
    return smooth_positives * target + smooth_negatives


def categorical_crossentropy(
    target: jnp.ndarray,
    preds: jnp.ndarray,
    from_logits: bool = False,
    label_smoothing: tp.Optional[jnp.ndarray] = None,
) -> jnp.ndarray:

    if label_smoothing is not None:
        target = smooth_labels(target, label_smoothing)

    if from_logits:
        preds = jax.nn.log_softmax(preds)

    else:
        preds = jnp.maximum(preds, types.EPSILON)
        preds = jnp.log(preds)

    return -jnp.sum(target * preds, axis=-1)


class CategoricalCrossentropy(Loss):
    """
    Computes the crossentropy loss between the target and predictions.
    Use this crossentropy loss function when there are two or more label classes.
    We expect target to be provided in a `one_hot` representation. If you want to
    provide target as integers, please use `SparseCategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature.
    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `preds` and `target` are
    `[batch_size, num_classes]`.

    Usage:
    ```python
    target = jnp.array([[0, 1, 0], [0, 0, 1]])
    preds = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cce = tx.losses.CategoricalCrossentropy()

    assert cce(target, preds) == 1.177
    # Calling with 'sample_weight'.
    assert cce(target, preds, sample_weight=tf.constant([0.3, 0.7])) == 0.814
    # Using 'sum' reduction type.
    cce = tx.losses.CategoricalCrossentropy(
        reduction=tx.losses.Reduction.SUM
    )
    assert cce(target, preds) == 2.354
    # Using 'none' reduction type.
    cce = tx.losses.CategoricalCrossentropy(
        reduction=tx.losses.Reduction.NONE
    )

    assert list(cce(target, preds)) == [0.0513, 2.303]
    ```

    Usage with the `Elegy` API:
    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.CategoricalCrossentropy(),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
    )
    ```
    """

    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: tp.Optional[types.ScalarLike] = None,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[types.ScalarLike] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs,
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
        super().__init__(reduction=reduction, weight=weight, on=on, **kwargs)

        self._from_logits = from_logits
        self._label_smoothing = (
            jnp.asarray(label_smoothing, dtype=jnp.float32)
            if label_smoothing is not None
            else None
        )

    def call(
        self,
        target: jnp.ndarray,
        preds: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Invokes the `CategoricalCrossentropy` instance.

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

        return categorical_crossentropy(
            target,
            preds,
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
        )
