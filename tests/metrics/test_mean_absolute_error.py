import jax.numpy as jnp
import numpy as np

import torch
from torchmetrics import MeanAbsoluteError

from treex.metrics.mean_absolute_error import MeanAbsolutError


class TestMAE:
    def test_mae_basic(self):

        y_true = np.random.randn(8, 20, 20)
        y_pred = np.random.randn(8, 20, 20)

        mean_absolute_error_instance = MeanAbsolutError()
        mae_treex = mean_absolute_error_instance(**{"y_true": y_true, "y_pred": y_pred})

        mean_absolute_error = MeanAbsoluteError()
        mae_tm = mean_absolute_error(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        assert np.isclose(np.array(mae_treex), mae_tm.numpy())

    def test_accumulative_mae(self):
        mean_absolute_error_instance = MeanAbsolutError()
        mean_absolute_error = MeanAbsoluteError()
        for batch in range(2):

            y_true = np.random.randn(8, 5, 5)
            y_pred = np.random.randn(8, 5, 5)

            mean_absolute_error_instance(**{"y_true": y_true, "y_pred": y_pred})
            mean_absolute_error(torch.from_numpy(y_pred), torch.from_numpy(y_true))

        assert np.isclose(
            np.array(mean_absolute_error_instance.compute()),
            mean_absolute_error.compute().numpy(),
        )
