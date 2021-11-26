import hypothesis as hp
import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics as tm
from hypothesis import strategies as st

import treex as tx


class TestMAE:
    def test_mae_basic(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        mae_tx = tx.metrics.MeanAbsoluteError()
        mae_tx_value = mae_tx(y_true=y_true, y_pred=y_pred)

        mae_tm = tm.MeanAbsoluteError()
        mae_tm_value = mae_tm(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        assert np.isclose(np.array(mae_tx_value), mae_tm_value.numpy())

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mae_weights_batch_dim(self, use_sample_weight):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        if use_sample_weight:
            sum = 0
            while sum == 0:
                sample_weight = np.random.choice([0, 1], 8)
                sum = sample_weight.sum()

        params = {"y_true": y_true, "y_pred": y_pred}
        mae_tx = tx.metrics.MeanAbsoluteError()
        if use_sample_weight:
            params.update({"sample_weight": sample_weight})
        mae_tx_value = mae_tx(**params)

        mae_tx = tx.metrics.MeanAbsoluteError()
        if use_sample_weight:
            y_true, y_pred = y_true[sample_weight == 1], y_pred[sample_weight == 1]
        mae_tx_no_sample_weight = mae_tx(y_true=y_true, y_pred=y_pred)

        assert np.isclose(mae_tx_value, mae_tx_no_sample_weight)

    @hp.given(
        use_sample_weight=st.booleans(),
    )
    @hp.settings(deadline=None, max_examples=10)
    def test_mae_weights_values_dim(self, use_sample_weight):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        params = {"y_true": y_true, "y_pred": y_pred}
        if use_sample_weight:
            sample_weight = np.random.choice([0, 1], 8 * 20).reshape((8, 20))
            params.update({"sample_weight": sample_weight})

        mae_tx = tx.metrics.MeanAbsoluteError()(**params)

        assert isinstance(mae_tx, jnp.ndarray)

    def test_accumulative_mae(self):
        mae_tx = tx.metrics.MeanAbsoluteError()
        mae_tm = tm.MeanAbsoluteError()
        for batch in range(2):

            y_true = np.random.randn(8, 5, 5)
            y_pred = np.random.randn(8, 5, 5)

            mae_tx(y_true=y_true, y_pred=y_pred)
            mae_tm(torch.from_numpy(y_pred), torch.from_numpy(y_true))

        assert np.isclose(
            np.array(mae_tx.compute()),
            mae_tm.compute().numpy(),
        )

    def test_mae_short(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        mae_tx_long = tx.metrics.MeanAbsoluteError()(y_true=y_true, y_pred=y_pred)
        mae_tx_short = tx.metrics.MAE()(y_true=y_true, y_pred=y_pred)
        assert np.isclose(np.array(mae_tx_long), np.array(mae_tx_short))
