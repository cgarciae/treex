import typing as tp

import jax.numpy as jnp

from treex import types
from treex.losses.loss import Loss, Reduction


def huber(target: jnp.ndarray, preds: jnp.ndarray, delta: float) -> jnp.ndarray:
    r"""
    Computes the Huber loss between target and predictions.
    
    For each value x in error = target - preds:

    $$
    loss =
    \begin{cases}
    \ 0.5 \times x^2,\hskip8em\text{if } |x|\leq d\\
    0.5 \times d^2 + d \times (|x| - d),\hskip1.7em \text{otherwise} 
    \end{cases}
    $$
    
    where d is delta. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = tx.losses.huber(target, preds, delta=1.0)
    assert loss.shape == (2,)

    preds = preds.astype(float)
    target = target.astype(float)
    delta = 1.0
    error = jnp.subtract(preds, target)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    assert jnp.array_equal(loss, jnp.mean(
      jnp.add(
          jnp.multiply(
              0.5,
              jnp.multiply(quadratic, quadratic)
              ),
              jnp.multiply(delta, linear)), axis=-1
    ))
    ```

    Arguments:
        target: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        preds: The predicted values. shape = `[batch_size, d0, .. dN]`.
        delta: A float, the point where the Huber loss function changes from a quadratic to linear.

    Returns:
          huber loss Values. If reduction is NONE, this has
         shape [batch_size, d0, .. dN-1]; otherwise, it is scalar.
         (Note dN-1 because all loss functions reduce by 1 dimension, usually axis=-1.)
    """
    preds = preds.astype(float)
    target = target.astype(float)
    delta = float(delta)
    error = jnp.subtract(preds, target)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = jnp.subtract(abs_error, quadratic)
    return jnp.mean(
        jnp.add(
            jnp.multiply(0.5, jnp.multiply(quadratic, quadratic)),
            jnp.multiply(delta, linear),
        ),
        axis=-1,
    )


class Huber(Loss):
    r"""
    Computes the Huber loss  between target and predictions.
    
    For each value x in error = target - preds:

    $$
    loss =
    \begin{cases}
    \ 0.5 \times x^2,\hskip8em\text{if } |x|\leq d\\
    0.5 \times d^2 + d \times (|x| - d),\hskip1.7em \text{otherwise} 
    \end{cases}
    $$
    
    where d is delta. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage:

    ```python
    target = jnp.array([[0, 1], [0, 0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    huber_loss = tx.losses.Huber()
    assert huber_loss(target, preds) == 0.155

    # Calling with 'sample_weight'.
    assert (
        huber_loss(target, preds, sample_weight=jnp.array([0.8, 0.2])) == 0.08500001
    )

    # Using 'sum' reduction type.
    huber_loss = tx.losses.Huber(
        reduction=tx.losses.Reduction.SUM
    )
    assert huber_loss(target, preds) == 0.31

    # Using 'none' reduction type.
    huber_loss = tx.losses.Huber(
        reduction=tx.losses.Reduction.NONE
    )

    assert jnp.equal(huber_loss(target, preds), jnp.array([0.18, 0.13000001])).all()
    ```
    Usage with the Elegy API:

    ```python
    model = elegy.Model(
        module_fn,
        loss=tx.losses.Huber(delta=1.0),
        metrics=elegy.metrics.Mean(),
    )
    ```
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        **kwargs
    ):
        """
        Initializes `Mean` class.

        Arguments:
            delta: (Optional) Defaults to 1.0. A float, the point where the Huber loss function changes from a quadratic to linear.
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
        self.delta = delta
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
        Invokes the `Huber` instance.

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
        return huber(target, preds, self.delta)
