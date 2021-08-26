import numpy as np
import pytest
import treex as tx


class TestMLP:
    def test_basic(self):
        mlp = tx.MLP([2, 32, 8, 4]).init(42)

        x = np.random.uniform(-1, 1, (10, 2))
        y = mlp(x)

        assert y.shape == (10, 4)

    def test_too_few_features(self):

        with pytest.raises(ValueError):
            mlp = tx.MLP([2]).init(42)
