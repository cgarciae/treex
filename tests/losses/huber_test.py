import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.keras as tfk

import treex as tx

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():

    target = jnp.array([[0, 1], [0, 0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    huber_loss = tx.losses.Huber()
    assert huber_loss(target=target, preds=preds) == 0.155

    # Calling with 'sample_weight'.
    assert (
        huber_loss(target=target, preds=preds, sample_weight=jnp.array([0.8, 0.2]))
        == 0.08500001
    )

    # Using 'sum' reduction type.
    huber_loss = tx.losses.Huber(reduction=tx.losses.Reduction.SUM)
    assert huber_loss(target=target, preds=preds) == 0.31

    # Using 'none' reduction type.
    huber_loss = tx.losses.Huber(reduction=tx.losses.Reduction.NONE)

    assert jnp.equal(
        huber_loss(target=target, preds=preds), jnp.array([0.18, 0.13000001])
    ).all()


def test_function():

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
    assert jnp.array_equal(
        loss,
        jnp.mean(
            jnp.add(
                jnp.multiply(0.5, jnp.multiply(quadratic, quadratic)),
                jnp.multiply(delta, linear),
            ),
            axis=-1,
        ),
    )


def test_compatibility():
    # Input:  true (target) and predicted (preds) tensors
    rng = jax.random.PRNGKey(121)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    target = target.astype(dtype=jnp.float32)
    preds = jax.random.uniform(rng, shape=(2, 3))

    # cosine_loss using sample_weight
    huber_loss = tx.losses.Huber(delta=1.0)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0)

    assert np.isclose(
        huber_loss(target=target, preds=preds, sample_weight=jnp.array([1, 0])),
        huber_loss_tfk(target, preds, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # cosine_loss with reduction method: SUM
    huber_loss = tx.losses.Huber(delta=1.0, reduction=tx.losses.Reduction.SUM)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0, reduction=tfk.losses.Reduction.SUM)
    assert np.isclose(
        huber_loss(target=target, preds=preds),
        huber_loss_tfk(target, preds),
        rtol=0.0001,
    )

    # cosine_loss with reduction method: NONE
    huber_loss = tx.losses.Huber(delta=1.0, reduction=tx.losses.Reduction.NONE)
    huber_loss_tfk = tfk.losses.Huber(delta=1.0, reduction=tfk.losses.Reduction.NONE)
    assert jnp.all(
        np.isclose(
            huber_loss(target=target, preds=preds),
            huber_loss_tfk(target, preds),
            rtol=0.0001,
        )
    )


if __name__ == "__main__":

    test_basic()
    test_function()
    test_compatibility()
