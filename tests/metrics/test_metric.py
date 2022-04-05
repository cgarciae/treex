import inspect
import typing as tp

import jax
import pytest

import treex as tx


class MyMetric(tx.Metric):
    target: tp.Optional[int] = tx.node(None)
    preds: tp.Optional[int] = tx.node(None)

    def reset(self):
        return self.replace(
            target=0,
            preds=0,
        )

    def update(self, target, preds, **_):
        return self.replace(
            target=self.target + target,
            preds=self.preds + preds,
        )

    def compute(self):
        return self.target, self.preds


class TestMetric:
    def test_basic(self):

        metric = MyMetric()

        metric = metric.reset()

        assert metric.target == 0
        assert metric.preds == 0

        metric = metric.update(target=10, preds=20)

        assert metric.target == 10
        assert metric.preds == 20

        assert metric.compute() == (10, 20)

        values, metric = metric(target=5, preds=6)

        assert values == (5, 6)
        assert metric.target == 15
        assert metric.preds == 26

    def test_on(self):

        metric = MyMetric().index_into(target=("a", 0), preds=("a", 0)).reset()

        target = {"a": [10]}
        preds = {"a": [20]}

        values, metric = metric(target=target, preds=preds)

        assert values == (10, 20)

    def test_raise_positional_arguments(self):

        metric = MyMetric().index_into(target=("a", 0), preds=("a", 0))

        target = {"a": [10]}
        preds = {"a": [20]}

        with pytest.raises(TypeError):
            metric(target, preds)

    def test_jit(self):
        class MyMetric(tx.Metric):
            a: tp.Optional[int] = tx.MetricState.node()

            def __init__(self) -> None:
                self.a = None
                super().__init__()

            def reset(self):
                return self.replace(a=0)

            def update(self, n):
                return self.replace(a=self.a + n)

            def compute(self):
                return self.a

        N = 0

        @jax.jit
        def f(m):
            nonlocal N
            N += 1
            return m.update(n=2)

        metric = MyMetric().reset()

        metric = f(metric)
        metric = f(metric)

        assert metric.a == 4
        assert N == 1
