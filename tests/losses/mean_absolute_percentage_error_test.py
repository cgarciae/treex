from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.keras as tfk

import treex as tx
from treex import types, utils


class MeanAbsolutePercentageErrorTest(TestCase):
    #
    def test_basic(self):
        y_true = jnp.array([[1.0, 1.0], [0.9, 0.0]])
        y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

        # Using 'auto'/'sum_over_batch_size' reduction type.
        mape = tx.losses.MeanAbsolutePercentageError()
        result = mape(y_true, y_pred)
        assert np.isclose(result, 2.78, rtol=0.01)

        # Calling with 'sample_weight'.
        assert np.isclose(
            mape(y_true, y_pred, sample_weight=jnp.array([0.1, 0.9])), 2.5, rtol=0.01
        )

        # Using 'sum' reduction type.
        mape = tx.losses.MeanAbsolutePercentageError(reduction=tx.losses.Reduction.SUM)

        assert np.isclose(mape(y_true, y_pred), 5.6, rtol=0.01)

        # Using 'none' reduction type.
        mape = tx.losses.MeanAbsolutePercentageError(reduction=tx.losses.Reduction.NONE)

        result = mape(y_true, y_pred)
        assert jnp.all(np.isclose(result, [0.0, 5.6], rtol=0.01))

    #
    def test_function(self):

        y_true = jnp.array([[1.0, 1.0], [0.9, 0.0]])
        y_pred = jnp.array([[1.0, 1.0], [1.0, 0.0]])

        ## Standard MAPE
        mape_elegy = tx.losses.MeanAbsolutePercentageError()
        mape_tfk = tfk.losses.MeanAbsolutePercentageError()
        assert np.isclose(
            mape_elegy(y_true, y_pred), mape_tfk(y_true, y_pred), rtol=0.0001
        )

        ## MAPE using sample_weight
        assert np.isclose(
            mape_elegy(y_true, y_pred, sample_weight=jnp.array([1, 0])),
            mape_tfk(y_true, y_pred, sample_weight=jnp.array([1, 0])),
            rtol=0.0001,
        )

        ## MAPE with reduction method: SUM
        mape_elegy = tx.losses.MeanAbsolutePercentageError(
            reduction=tx.losses.Reduction.SUM
        )
        mape_tfk = tfk.losses.MeanAbsolutePercentageError(
            reduction=tfk.losses.Reduction.SUM
        )
        assert np.isclose(
            mape_elegy(y_true, y_pred), mape_tfk(y_true, y_pred), rtol=0.0001
        )

        ## MAPE with reduction method: NONE
        mape_elegy = tx.losses.MeanAbsolutePercentageError(
            reduction=tx.losses.Reduction.NONE
        )
        mape_tfk = tfk.losses.MeanAbsolutePercentageError(
            reduction=tfk.losses.Reduction.NONE
        )
        assert jnp.all(
            np.isclose(
                mape_elegy(y_true, y_pred), mape_tfk(y_true, y_pred), rtol=0.0001
            )
        )

        ## Prove the loss function
        rng = jax.random.PRNGKey(42)
        y_true = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
        y_pred = jax.random.uniform(rng, shape=(2, 3))
        y_true = y_true.astype(y_pred.dtype)
        loss = tx.losses.mean_absolute_percentage_error(y_true, y_pred)
        assert loss.shape == (2,)
        assert jnp.array_equal(
            loss,
            100
            * jnp.mean(
                jnp.abs(
                    (y_pred - y_true) / jnp.maximum(jnp.abs(y_true), types.EPSILON)
                ),
                axis=-1,
            ),
        )
