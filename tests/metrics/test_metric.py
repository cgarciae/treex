import inspect

import jax
import pytest

import treex as tx


class TestMetric:
    def test_basic(self):
        class MyMetric(tx.Metric):
            def update(self, y_true, y_pred):
                self.y_true = y_true
                self.y_pred = y_pred
                return self

            def compute(self):
                return self.y_true, self.y_pred

        metric = MyMetric()

        signature_parameters = inspect.signature(metric.update).parameters
        assert "y_true" in signature_parameters
        assert "y_pred" in signature_parameters

        signature_parameters = inspect.signature(metric.__call__).parameters
        assert "y_true" in signature_parameters
        assert "y_pred" in signature_parameters

    def test_on(self):
        class MyMetric(tx.Metric):
            def update(self, y_true, y_pred):
                self.y_true = y_true
                self.y_pred = y_pred
                return self

            def compute(self):
                return self.y_true, self.y_pred

        metric = MyMetric(on=("a", 0))

        y_true = {"a": [10]}
        y_pred = {"a": [20]}

        assert metric(y_true=y_true, y_pred=y_pred) == (10, 20)

    def test_raise_positional_arguments(self):
        class MyMetric(tx.Metric):
            def update(self, y_true, y_pred):
                self.y_true = y_true
                self.y_pred = y_pred
                return self

            def compute(self):
                return self.y_true, self.y_pred

        metric = MyMetric(on=("a", 0))

        y_true = {"a": [10]}
        y_pred = {"a": [20]}

        with pytest.raises(TypeError):
            metric(y_true, y_pred)

    def test_jit(self):
        class MyMetric(tx.Metric):
            a: int = tx.MetricState.node()

            def __init__(self) -> None:
                self.a = 0
                super().__init__()

            def update(self, n):
                self.a += n

            def compute(self):
                return self.a

        N = 0

        @jax.jit
        def f(m):
            nonlocal N
            N += 1
            m(n=2)
            return m

        metric = MyMetric()

        metric = f(metric)
        metric = f(metric)

        assert metric.a == 4
        assert N == 1
