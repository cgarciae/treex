import hypothesis as hp
import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics as tm
from hypothesis import strategies as st

import treex as tx


class TestMSE:
    def test_mse_basic(self):

        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mse_tx = tx.metrics.MeanSquareError()
        mse_tx_value = mse_tx(target=target, preds=preds)

        mse_tm = tm.MeanSquaredError()
        mse_tm_value = mse_tm(torch.from_numpy(preds), torch.from_numpy(target))
        assert np.isclose(np.array(mse_tx_value), mse_tm_value.numpy())

    def test_accumulative_mse(self):
        mse_tx = tx.metrics.MeanSquareError()
        mse_tm = tm.MeanSquaredError()

        for batch in range(2):
            target = np.random.randn(8, 5, 5)
            preds = np.random.randn(8, 5, 5)

            mse_tx(target=target, preds=preds)
            mse_tm(torch.from_numpy(preds), torch.from_numpy(target))

        assert np.isclose(
            np.array(mse_tx.compute()),
            mse_tm.compute().numpy(),
        )

    def test_mse_short(self):

        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        mse_tx_long = tx.metrics.MeanSquareError()(target=target, preds=preds)
        mse_tx_short = tx.metrics.MSE()(target=target, preds=preds)
        assert np.isclose(np.array(mse_tx_long), np.array(mse_tx_short))

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mse_weights(self, use_sample_weight):

        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        params = {"target": target, "preds": preds}
        mse_tx = tx.metrics.MeanSquareError()

        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8)
            while sample_weight.sum() == 0:
                sample_weight = np.random.choice([0, 1], 8)
            params.update({"sample_weight": sample_weight})

        mse_tx_value = mse_tx(**params)

        if use_sample_weight:
            target, preds = target[sample_weight == 1], preds[sample_weight == 1]

        mse_tx = tx.metrics.MeanSquareError()
        mse_tx_no_sample_weight = mse_tx(**params)

        assert np.isclose(mse_tx_value, mse_tx_no_sample_weight)

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mse_weights_values_dim(self, use_sample_weight):

        target = np.random.randn(8, 20, 20)
        preds = np.random.randn(8, 20, 20)

        params = {"target": target, "preds": preds}
        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8 * 20).reshape((8, 20))
            params.update({"sample_weight": sample_weight})

        mse_tx = tx.metrics.MeanSquareError()(**params)

        assert isinstance(mse_tx, jnp.ndarray)
