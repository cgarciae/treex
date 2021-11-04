import jax.numpy as jnp
import numpy as np
import torch
import torchmetrics as tm

import treex as tx


class TestMAE:
    def test_mae_basic(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        mean_absolute_error_instance = tx.metrics.MeanAbsoluteError()
        mae_treex = mean_absolute_error_instance(**{"y_true": y_true, "y_pred": y_pred})

        mean_absolute_error = tm.MeanAbsoluteError()
        mae_tm = mean_absolute_error(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        assert np.isclose(np.array(mae_treex), mae_tm.numpy())

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
