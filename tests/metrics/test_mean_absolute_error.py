import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics as tm

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

    def test_mae_weights_batch_dim(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        sum = 0
        while sum == 0:
            sample_weight = np.random.choice([0, 1], 8)
            sum = sample_weight.sum()

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_tx(
            **{"y_true": y_true, "y_pred": y_pred, "sample_weight": sample_weight}
        )

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        y_true, y_pred = y_true[sample_weight == 1], y_pred[sample_weight == 1]
        mae_treex_no_sample_weight = mean_absolute_error_tx(
            **{"y_true": y_true, "y_pred": y_pred}
        )

        assert np.isclose(mae_treex, mae_treex_no_sample_weight)

    def test_mae_weights_values_dim(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)
        sample_weight = np.random.choice([0, 1], 8 * 20).reshape((8, 20))

        mean_absolute_error_tx = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_tx(
            **{"y_true": y_true, "y_pred": y_pred, "sample_weight": sample_weight}
        )

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
