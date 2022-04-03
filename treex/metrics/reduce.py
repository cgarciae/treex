import enum
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

from treex import types
from treex.metrics.metric import Metric

M = tp.TypeVar("M", bound="Reduce")


class Reduction(enum.Enum):
    sum = enum.auto()
    sum_over_batch_size = enum.auto()
    weighted_mean = enum.auto()


class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values."""

    total: tp.Optional[jnp.ndarray] = types.MetricState.node()
    count: tp.Optional[jnp.ndarray] = types.MetricState.node()

    def __init__(
        self,
        reduction: tp.Union[Reduction, str],
        name: tp.Optional[str] = None,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__(name=name, dtype=dtype)
        self.reduction = (
            reduction if isinstance(reduction, Reduction) else Reduction[reduction]
        )
        self.total = None
        self.count = None

    def reset(self: M) -> M:
        # initialize states
        total = jnp.array(0.0, dtype=self.dtype)

        if self.reduction in (
            Reduction.sum_over_batch_size,
            Reduction.weighted_mean,
        ):
            count = jnp.array(0, dtype=jnp.uint32)
        else:
            count = None

        return self.replace(total=total, count=count)

    def update(
        self: M,
        values: jnp.ndarray,
        sample_weight: tp.Optional[jnp.ndarray] = None,
        **_,
    ) -> M:
        """
        Accumulates statistics for computing the reduction metric. For example, if `values` is [1, 3, 5, 7]
        and reduction=SUM_OVER_BATCH_SIZE, then the value of `result()` is 4. If the `sample_weight`
        is specified as [1, 1, 0, 0] then value of `result()` would be 2.

        Arguments:
            values: Per-example value.
            sample_weight: Optional weighting of each example. Defaults to 1.

        Returns:
            Array with the cumulative reduce.
        """

        if self.total is None:
            raise self._not_initialized_error()

        # perform update
        if sample_weight is not None:
            if sample_weight.ndim > values.ndim:
                raise Exception(
                    f"sample_weight dimention is higher than values, when masking values sample_weight dimention needs to be equal or lower than values dimension, currently values have shape equal to {values.shape}"
                )

            try:
                # Broadcast weights if possible.
                sample_weight = jnp.broadcast_to(sample_weight, values.shape)
            except ValueError:
                # Reduce values to same ndim as weight array
                values_ndim, weight_ndim = values.ndim, sample_weight.ndim
                if self.reduction == Reduction.sum:
                    values = jnp.sum(values, axis=list(range(weight_ndim, values_ndim)))
                else:
                    values = jnp.mean(
                        values, axis=list(range(weight_ndim, values_ndim))
                    )

            values = values * sample_weight

        value_sum = jnp.sum(values)

        total = (self.total + value_sum).astype(self.total.dtype)

        # Exit early if the reduction doesn't have a denominator.
        if self.reduction == Reduction.sum:
            num_values = None

        # Update `count` for reductions that require a denominator.
        elif self.reduction == Reduction.sum_over_batch_size:
            num_values = np.prod(values.shape)

        else:
            if sample_weight is None:
                num_values = np.prod(values.shape)
            else:
                num_values = jnp.sum(sample_weight)

        if self.count is not None:
            assert num_values is not None
            count = (self.count + num_values).astype(self.count.dtype)
        else:
            count = None

        return self.replace(total=total, count=count)

    def compute(self) -> jnp.ndarray:
        if self.total is None:
            raise self._not_initialized_error()

        if self.reduction == Reduction.sum:
            return self.total
        else:
            return self.total / self.count
