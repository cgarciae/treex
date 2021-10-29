import numpy as np
import pytest

import treex as tx


class TestMLP:
    def test_basic(self):
        x = np.random.uniform(-1, 1, (10, 2))
        mlp = tx.MLP([32, 8, 4]).init(42, x)

        y = mlp(x)

        assert y.shape == (10, 4)

    def test_too_few_features(self):
        x = np.random.uniform(-1, 1, (10, 2))

        with pytest.raises(ValueError):
            mlp = tx.MLP([]).init(42, x)
