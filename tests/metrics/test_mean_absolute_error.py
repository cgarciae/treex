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

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_tx(**{"y_true": y_true, "y_pred": y_pred})

        mean_absolute_error_tm = tm.MeanAbsoluteError()
        mae_tm = mean_absolute_error_tm(
            torch.from_numpy(y_pred), torch.from_numpy(y_true)
        )
        assert np.isclose(np.array(mae_treex), mae_tm.numpy())

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
        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        if use_sample_weight:
            params.update({"sample_weight": sample_weight})
        mae_treex = mean_absolute_error_tx(**params)

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        if use_sample_weight:
            y_true, y_pred = y_true[sample_weight == 1], y_pred[sample_weight == 1]
        mae_treex_no_sample_weight = mean_absolute_error_tx(
            **{"y_true": y_true, "y_pred": y_pred}
        )

        assert np.isclose(mae_treex, mae_treex_no_sample_weight)

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

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_tx(**params)

        assert isinstance(mae_treex, jnp.ndarray)

    def test_accumulative_mae(self):
        mean_absolute_error_instance = tx.metrics.MeanAbsoluteError()
        mean_absolute_error = tm.MeanAbsoluteError()
        for batch in range(2):

            y_true = np.random.randn(8, 5, 5)
            y_pred = np.random.randn(8, 5, 5)

            mean_absolute_error_instance(**{"y_true": y_true, "y_pred": y_pred})
            mean_absolute_error(torch.from_numpy(y_pred), torch.from_numpy(y_true))

        assert np.isclose(
            np.array(mean_absolute_error_instance.compute()),
            mean_absolute_error.compute().numpy(),
        )

    def test_mae_short(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_tx(**{"y_true": y_true, "y_pred": y_pred})

        mean_absolute_error_tx_short = tx.metrics.MAE()
        mae_treex_short = mean_absolute_error_tx_short(
            **{"y_true": y_true, "y_pred": y_pred}
        )
        assert np.isclose(np.array(mae_treex), np.array(mae_treex_short))
