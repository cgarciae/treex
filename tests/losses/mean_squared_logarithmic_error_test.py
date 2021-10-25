import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.keras as tfk

import treex as tx
from treex import types, utils

# import debugpy

# print("Waiting for debugger...")
# debugpy.listen(5679)
# debugpy.wait_for_client()


def test_basic():

    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])

    # Using 'auto'/'sum_over_batch_size' reduction type.
    msle = tx.losses.MeanSquaredLogarithmicError()

    assert msle(target=target, preds=preds) == 0.24022643

    # Calling with 'sample_weight'.
    assert (
        msle(target=target, preds=preds, sample_weight=jnp.array([0.7, 0.3]))
        == 0.12011322
    )

    # Using 'sum' reduction type.
    msle = tx.losses.MeanSquaredLogarithmicError(reduction=tx.losses.Reduction.SUM)

    assert msle(target=target, preds=preds) == 0.48045287

    # Using 'none' reduction type.
    msle = tx.losses.MeanSquaredLogarithmicError(reduction=tx.losses.Reduction.NONE)

    assert jnp.equal(
        msle(target=target, preds=preds), jnp.array([0.24022643, 0.24022643])
    ).all()


def test_function():

    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = tx.losses.mean_squared_logarithmic_error(target, preds)

    assert loss.shape == (2,)

    first_log = jnp.log(jnp.maximum(target, types.EPSILON) + 1.0)
    second_log = jnp.log(jnp.maximum(preds, types.EPSILON) + 1.0)
    assert jnp.array_equal(loss, jnp.mean(jnp.square(first_log - second_log), axis=-1))


def test_compatibility():
    # Input:  true (target) and predicted (preds) tensors
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # MSLE using sample_weight
    msle_elegy = tx.losses.MeanSquaredLogarithmicError()
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError()
    assert np.isclose(
        msle_elegy(target=target, preds=preds, sample_weight=jnp.array([1, 0])),
        msle_tfk(target, preds, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # MSLE with reduction method: SUM
    msle_elegy = tx.losses.MeanSquaredLogarithmicError(
        reduction=tx.losses.Reduction.SUM
    )
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError(
        reduction=tfk.losses.Reduction.SUM
    )
    assert np.isclose(
        msle_elegy(target=target, preds=preds), msle_tfk(target, preds), rtol=0.0001
    )

    # MSLE with reduction method: NONE
    msle_elegy = tx.losses.MeanSquaredLogarithmicError(
        reduction=tx.losses.Reduction.NONE
    )
    msle_tfk = tfk.losses.MeanSquaredLogarithmicError(
        reduction=tfk.losses.Reduction.NONE
    )
    assert jnp.all(
        np.isclose(
            msle_elegy(target=target, preds=preds),
            msle_tfk(target, preds),
            rtol=0.0001,
        )
    )


if __name__ == "__main__":

    test_basic()
    test_function()
