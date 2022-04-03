from unittest import TestCase

import jax.numpy as jnp
import numpy as np
import pytest

import treex as tx


class LossTest(TestCase):
    def test_basic(self):
        class MAE(tx.Loss):
            def call(self, target, preds):
                return jnp.abs(target - preds)

        target = jnp.array([1.0, 2.0, 3.0])
        preds = jnp.array([2.0, 3.0, 4.0])

        mae = MAE()

        sample_loss = mae.call(target, preds)
        loss = mae(target=target, preds=preds)

        assert jnp.alltrue(sample_loss == jnp.array([1.0, 1.0, 1.0]))
        assert loss == 1

    def test_slice(self):
        class MAE(tx.Loss):
            def call(self, target, preds):
                return jnp.abs(target - preds)

        target = dict(a=jnp.array([1.0, 2.0, 3.0]))
        preds = dict(a=jnp.array([2.0, 3.0, 4.0]))

        mae = MAE().slice(target="a", preds="a")

        # raises because it doesn't use kwargs
        with pytest.raises(BaseException):
            sample_loss = mae(target, preds)

        # raises because it doesn't use __call__ which filters
        with pytest.raises(BaseException):
            sample_loss = mae.call(target=target, preds=preds)

        loss = mae(target=target, preds=preds)

        assert loss == 1
