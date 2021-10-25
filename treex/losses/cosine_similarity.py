import typing as tp

import jax.numpy as jnp

from treex import types, utils
from treex.losses.loss import Loss, Reduction


def cosine_similarity(
    target: jnp.ndarray, preds: jnp.ndarray, axis: int
) -> jnp.ndarray:
    """
    Computes the cosine similarity between target and predictions.

    ```python
    loss = -sum(l2_norm(target) * l2_norm(preds))
    ```

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = tx.losses.cosine_similarity(target, preds, axis=1)
    assert loss.shape == (2,)

    target = target / jnp.maximum(jnp.linalg.norm(target, axis=1, keepdims=True), jnp.sqrt(types.EPSILON))
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, axis=1, keepdims=True), jnp.sqrt(types.EPSILON))
    assert jnp.array_equal(loss, -jnp.sum(target * preds, axis=1))
    ```

    Arguments:
        target: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        preds: The predicted values. shape = `[batch_size, d0, .. dN]`.
        axis: The dimension along which the cosine similarity is computed.

    Returns:
          cosine similarity Values. If reduction is NONE, this has
         shape [batch_size, d0, .. dN-1]; otherwise, it is scalar.
         (Note dN-1 because all loss functions reduce by 1 dimension, usually axis=-1.)
    """
    target = target / jnp.maximum(
        jnp.linalg.norm(target, axis=axis, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    preds = preds / jnp.maximum(
        jnp.linalg.norm(preds, axis=axis, keepdims=True), jnp.sqrt(types.EPSILON)
    )
    return -jnp.sum(target * preds, axis=axis)


class CosineSimilarity(Loss):
    """
    Computes the mean squared logarithmic errors between target and predictions.

    `loss = -sum(l2_norm(target) * l2_norm(preds))`

    Usage:

    ```python
    target = jnp.array([[0., 1.], [1., 1.]])
    preds = jnp.array([[1., 0.], [1., 1.]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    cosine_loss = tx.losses.CosineSimilarity(axis=1)
    assert cosine_loss(target, preds) == -0.49999997

    # Calling with 'sample_weight'.
    assert cosine_loss(target, preds, sample_weight=jnp.array([0.8, 0.2])) == -0.099999994

    # Using 'sum' reduction type.
    cosine_loss = tx.losses.CosineSimilarity(axis=1,
        reduction=tx.losses.Reduction.SUM
    )
    assert cosine_loss(target, preds) == -0.99999994

    # Using 'none' reduction type.
    cosine_loss = tx.losses.CosineSimilarity(axis=1,
        reduction=tx.losses.Reduction.NONE
    )

    assert jnp.equal(cosine_loss(target, preds), jnp.array([-0., -0.99999994])).all()
    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.CosineSimilarity(axis=1),
        metrics=elegy.metrics.Mean(),
    )
    ```
    """

    def __init__(
        self,
        axis: int = -1,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `Mean` class.

        Arguments:
            axis: (Optional) Defaults to -1. The dimension along which the cosine
                   similarity is computed.
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
        self.axis = axis
        return super().__init__(reduction=reduction, weight=weight, on=on, **kwargs)

    def call(
        self,
        target: jnp.ndarray,
        preds: jnp.ndarray,
        sample_weight: tp.Optional[
            jnp.ndarray
        ] = None,  # not used, __call__ handles it, left for documentation purposes.
    ) -> jnp.ndarray:
        """
        Invokes the `CosineSimilarity` instance.

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
        return cosine_similarity(target, preds, self.axis)
