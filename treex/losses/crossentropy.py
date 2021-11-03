import typing as tp

import jax
import jax.numpy as jnp
import optax

from treex import types, utils
from treex.losses.loss import Loss, Reduction


def smooth_labels(
    target: jnp.ndarray,
    smoothing: jnp.ndarray,
) -> jnp.ndarray:
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / target.shape[-1]
    return smooth_positives * target + smooth_negatives


def crossentropy(
    target: jnp.ndarray,
    preds: jnp.ndarray,
    *,
    binary: bool = False,
    from_logits: bool = True,
    label_smoothing: tp.Optional[float] = None,
    check_bounds: bool = True,
) -> jnp.ndarray:

    n_classes = preds.shape[-1]

    if target.ndim == preds.ndim - 1:
        if target.shape != preds.shape[:-1]:
            raise ValueError(
                f"Target shape '{target.shape}' does not match preds shape '{preds.shape}'"
            )
        target = jax.nn.one_hot(target, n_classes)
    else:
        if target.ndim != preds.ndim:
            raise ValueError(
                f"Target shape '{target.shape}' does not match preds shape '{preds.shape}'"
            )

    if label_smoothing is not None:
        target = optax.smooth_labels(target, label_smoothing)

    if from_logits:
        if binary:
            loss = optax.sigmoid_binary_cross_entropy(preds, target).mean(axis=-1)
        else:
            loss = optax.softmax_cross_entropy(preds, target)
    else:
        preds = jnp.clip(preds, types.EPSILON, 1.0 - types.EPSILON)

        if binary:
            loss = target * jnp.log(preds)  # + types.EPSILON)
            loss += (1 - target) * jnp.log(1 - preds)  # + types.EPSILON)
            loss = -loss.mean(axis=-1)
        else:
            loss = -(target * jnp.log(preds)).sum(axis=-1)

    # TODO: implement check_bounds
    # if check_bounds:
    #     # set NaN where target is negative or larger/equal to the number of preds channels
    #     loss = jnp.where(target < 0, jnp.nan, loss)
    #     loss = jnp.where(target >= n_classes, jnp.nan, loss)

    return loss


class Crossentropy(Loss):
    """
    Computes the crossentropy loss between the target and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect target to be provided as integers. If you want to provide target
    using `one-hot` representation, please use `CategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature for `preds`
    and a single floating point value per feature for `target`.
    In the snippet below, there is a single floating point value per example for
    `target` and `# classes` floating pointing values per example for `preds`.
    The shape of `target` is `[batch_size]` and the shape of `preds` is
    `[batch_size, num_classes]`.

    Usage:
    ```python
    target = jnp.array([1, 2])
    preds = jnp.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce = tx.losses.Crossentropy()
    result = scce(target, preds)  # 1.177
    assert np.isclose(result, 1.177, rtol=0.01)

    # Calling with 'sample_weight'.
    result = scce(target, preds, sample_weight=jnp.array([0.3, 0.7]))  # 0.814
    assert np.isclose(result, 0.814, rtol=0.01)

    # Using 'sum' reduction type.
    scce = tx.losses.Crossentropy(
        reduction=tx.losses.Reduction.SUM
    )
    result = scce(target, preds)  # 2.354
    assert np.isclose(result, 2.354, rtol=0.01)

    # Using 'none' reduction type.
    scce = tx.losses.Crossentropy(
        reduction=tx.losses.Reduction.NONE
    )
    result = scce(target, preds)  # [0.0513, 2.303]
    assert jnp.all(np.isclose(result, [0.0513, 2.303], rtol=0.01))
    ```

    Usage with the `Elegy` API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.Crossentropy(),
        metrics=elegy.metrics.Accuracy(),
        optimizer=optax.adam(1e-3),
    )

    ```
    """

    def __init__(
        self,
        *,
        from_logits: bool = True,
        binary: bool = False,
        label_smoothing: tp.Optional[float] = None,
        reduction: tp.Optional[Reduction] = None,
        check_bounds: bool = True,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):
        """
        Initializes `SparseCategoricalCrossentropy` instance.

        Arguments:
            from_logits: Whether `preds` is expected to be a logits tensor. By
                default, we assume that `preds` encodes a probability distribution.
                **Note - Using from_logits=True is more numerically stable.**
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
            check_bounds: If `True` (default), checks `target` for negative values and values
                larger or equal than the number of channels in `preds`. Sets loss to NaN
                if this is the case. If `False`, the check is disabled and the loss may contain
                incorrect values.
        """
        super().__init__(reduction=reduction, weight=weight, on=on, name=name)

        self._from_logits = from_logits
        self._check_bounds = check_bounds
        self._binary = binary
        self._label_smoothing = label_smoothing

    def call(
        self, target, preds, sample_weight: tp.Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Invokes the `SparseCategoricalCrossentropy` instance.

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

        return crossentropy(
            target,
            preds,
            binary=self._binary,
            from_logits=self._from_logits,
            label_smoothing=self._label_smoothing,
            check_bounds=self._check_bounds,
        )
