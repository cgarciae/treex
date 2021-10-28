import enum

import hypothesis as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import binary

import treex as tx


class Mode(enum.Enum):
    SPARSE_CATEGORICAL = enum.auto()
    CATEGORICAL = enum.auto()
    BINARY = enum.auto()


class TestCrossentropy:
    @hp.given(
        mode=st.sampled_from(list(Mode)),
        from_logits=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_compatibility(self, mode, from_logits):

        binary = mode == Mode.BINARY

        TFClass = (
            tf.losses.BinaryCrossentropy
            if mode == Mode.BINARY
            else tf.losses.CategoricalCrossentropy
            if mode == Mode.CATEGORICAL
            else tf.losses.SparseCategoricalCrossentropy
        )

        # Input:  true (target) and predicted (preds) tensors
        if mode == Mode.SPARSE_CATEGORICAL:
            target = jnp.array([1, 0])
        else:
            target = jnp.array([[0.0, 1.0], [0.0, 0.0]])

        preds = jnp.array([[0.6, 0.4], [0.4, 0.6]])

        # Standard BCE, considering prediction tensor as probabilities
        bce_treex = tx.losses.Crossentropy(binary=binary, from_logits=from_logits)
        bce_tfk = TFClass(from_logits=from_logits)
        assert np.isclose(
            bce_treex(target=target, preds=preds), bce_tfk(target, preds), rtol=0.0001
        )

        # BCE using sample_weight
        bce_treex = tx.losses.Crossentropy(binary=binary, from_logits=from_logits)
        bce_tfk = TFClass(from_logits=from_logits)
        assert np.isclose(
            bce_treex(target=target, preds=preds, sample_weight=jnp.array([1, 0])),
            bce_tfk(target, preds, sample_weight=jnp.array([1, 0])),
            rtol=0.0001,
        )

        # BCE with reduction method: SUM
        bce_treex = tx.losses.Crossentropy(
            binary=binary, from_logits=from_logits, reduction=tx.losses.Reduction.SUM
        )
        bce_tfk = TFClass(from_logits=from_logits, reduction=tf.losses.Reduction.SUM)
        assert np.isclose(
            bce_treex(target=target, preds=preds), bce_tfk(target, preds), rtol=0.0001
        )

        # BCE with reduction method: NONE
        bce_treex = tx.losses.Crossentropy(
            binary=binary, from_logits=from_logits, reduction=tx.losses.Reduction.NONE
        )
        bce_tfk = TFClass(from_logits=from_logits, reduction=tf.losses.Reduction.NONE)
        assert jnp.all(
            np.isclose(
                bce_treex(target=target, preds=preds),
                bce_tfk(target, preds),
                rtol=0.0001,
            )
        )

        if mode != Mode.SPARSE_CATEGORICAL:
            # BCE with label smoothing
            bce_treex = tx.losses.Crossentropy(
                binary=binary, from_logits=from_logits, label_smoothing=0.9
            )
            bce_tfk = TFClass(from_logits=from_logits, label_smoothing=0.9)
            assert np.isclose(
                bce_treex(target=target, preds=preds),
                bce_tfk(target, preds),
                rtol=0.0001,
            )
