import jax.numpy as jnp
import numpy as np
import tensorflow.keras as tfk

import treex as tx


def test_basic():
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    bce = tx.losses.BinaryCrossentropy()
    result = bce(target=target, preds=preds)
    assert np.isclose(result, 0.815, rtol=0.01)

    y_logits = jnp.log(preds) - jnp.log(1 - preds)
    bce = tx.losses.BinaryCrossentropy(from_logits=True)
    result_from_logits = bce(target=target, preds=y_logits)
    assert np.isclose(result_from_logits, 0.815, rtol=0.01)
    assert np.isclose(result_from_logits, result, rtol=0.01)

    bce = tx.losses.BinaryCrossentropy()
    result = bce(target=target, preds=preds, sample_weight=jnp.array([1, 0]))
    assert np.isclose(result, 0.458, rtol=0.01)

    bce = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.SUM)
    result = bce(target=target, preds=preds)
    assert np.isclose(result, 1.630, rtol=0.01)

    bce = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.NONE)
    result = bce(target=target, preds=preds)
    assert jnp.all(np.isclose(result, jnp.array([0.916, 0.713]), rtol=0.01))


def test_compatibility():

    # Input:  true (target) and predicted (preds) tensors
    target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

    # Standard BCE, considering prediction tensor as probabilities
    bce_treex = tx.losses.BinaryCrossentropy()
    bce_tfk = tfk.losses.BinaryCrossentropy()
    assert np.isclose(
        bce_treex(target=target, preds=preds), bce_tfk(target, preds), rtol=0.0001
    )

    # Standard BCE, considering prediction tensor as logits
    y_logits = jnp.log(preds) - jnp.log(1 - preds)
    bce_treex = tx.losses.BinaryCrossentropy(from_logits=True)
    bce_tfk = tfk.losses.BinaryCrossentropy(from_logits=True)
    assert np.isclose(
        bce_treex(target=target, preds=y_logits),
        bce_tfk(target, y_logits),
        rtol=0.0001,
    )

    # BCE using sample_weight
    bce_treex = tx.losses.BinaryCrossentropy()
    bce_tfk = tfk.losses.BinaryCrossentropy()
    assert np.isclose(
        bce_treex(target=target, preds=preds, sample_weight=jnp.array([1, 0])),
        bce_tfk(target, preds, sample_weight=jnp.array([1, 0])),
        rtol=0.0001,
    )

    # BCE with reduction method: SUM
    bce_treex = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.SUM)
    bce_tfk = tfk.losses.BinaryCrossentropy(reduction=tfk.losses.Reduction.SUM)
    assert np.isclose(
        bce_treex(target=target, preds=preds), bce_tfk(target, preds), rtol=0.0001
    )

    # BCE with reduction method: NONE
    bce_treex = tx.losses.BinaryCrossentropy(reduction=tx.losses.Reduction.NONE)
    bce_tfk = tfk.losses.BinaryCrossentropy(reduction=tfk.losses.Reduction.NONE)
    assert jnp.all(
        np.isclose(
            bce_treex(target=target, preds=preds),
            bce_tfk(target, preds),
            rtol=0.0001,
        )
    )

    # BCE with label smoothing
    bce_treex = tx.losses.BinaryCrossentropy(label_smoothing=0.9)
    bce_tfk = tfk.losses.BinaryCrossentropy(label_smoothing=0.9)
    assert np.isclose(
        bce_treex(target=target, preds=preds), bce_tfk(target, preds), rtol=0.0001
    )
