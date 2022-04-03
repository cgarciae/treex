# Implementation based on Tensorflow Keras:
# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/losses.py#L44-L201

import typing as tp
from abc import ABC, abstractmethod
from enum import Enum

import jax.numpy as jnp
import numpy as np
from numpy.lib.arraysetops import isin

from treex import types, utils


class Reduction(Enum):
    """
    Types of loss reduction.

    Contains the following values:
    * `NONE`: Weighted losses with one dimension reduced (axis=-1, or axis
        specified by loss function). When this reduction type used with built-in
        Keras training loops like `fit`/`evaluate`, the unreduced vector loss is
        passed to the optimizer but the reported loss will be a scalar value.
    * `SUM`: Scalar sum of weighted losses.
    * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
    """

    # AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    @classmethod
    def all(cls):
        return (
            # cls.AUTO,
            cls.NONE,
            cls.SUM,
            cls.SUM_OVER_BATCH_SIZE,
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError("Invalid Reduction Key %s." % key)


class Loss(ABC):
    """
    Loss base class.

    To be implemented by subclasses:

    * `call()`: Contains the logic for loss calculation.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):
        def call(self, target, preds):
            return jnp.mean(jnp.square(preds - target), axis=-1)
    ```

    Please see the [Modules, Losses, and Metrics Guide]
    (https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#losses) for more
    details on this.
    """

    def __init__(
        self,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[types.ScalarLike] = None,
        name: tp.Optional[str] = None,
    ):
        """
        Initializes `Loss` class.

        Arguments:
            reduction: (Optional) Type of `tx.losses.Reduction` to apply to
                loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`.
            weight: Optional weight contribution for the total loss. Defaults to `1`.
            name: Optional name for the instance, if not provided lower snake_case version
                of the name of the class is used instead.
        """
        self.name = name if name is not None else utils._get_name(self)
        self.weight = (
            jnp.asarray(weight, dtype=jnp.float32)
            if weight is not None
            else jnp.array(1.0, dtype=jnp.float32)
        )
        self.reduction = (
            reduction if reduction is not None else Reduction.SUM_OVER_BATCH_SIZE
        )

    def __call__(
        self,
        **kwargs,
    ) -> jnp.ndarray:

        sample_weight: tp.Optional[jnp.ndarray] = kwargs.pop("sample_weight", None)

        values = self.call(**kwargs)

        return reduce_loss(values, sample_weight, self.weight, self.reduction)

    @abstractmethod
    def call(self, **kwargs) -> jnp.ndarray:
        ...

    def slice(self, **kwargs: types.IndexLike) -> "SliceParamsLoss":
        return SliceParamsLoss(self, kwargs)


def reduce_loss(
    values: jnp.ndarray, sample_weight: tp.Optional[jnp.ndarray], weight, reduction
) -> jnp.ndarray:

    values = jnp.asarray(values)

    if sample_weight is not None:
        # expand `sample_weight` dimensions until it has the same rank as `values`
        while sample_weight.ndim < values.ndim:
            sample_weight = sample_weight[..., jnp.newaxis]

        values *= sample_weight

    if reduction == Reduction.NONE:
        loss = values
    elif reduction == Reduction.SUM:
        loss = jnp.sum(values)
    elif reduction == Reduction.SUM_OVER_BATCH_SIZE:
        loss = jnp.sum(values) / jnp.prod(jnp.array(values.shape))
    else:
        raise ValueError(f"Invalid reduction '{reduction}'")

    return loss * weight


Slice = tp.Tuple[tp.Union[int, str], ...]


class SliceParamsLoss(Loss):
    arg_slice: tp.Dict[str, Slice]
    loss: Loss

    def __init__(
        self,
        loss: Loss,
        arg_slice: tp.Dict[str, types.IndexLike],
    ):
        super().__init__(name=loss.name, weight=None, reduction=Reduction.NONE)
        self.loss = loss
        self.arg_slice = {
            key: tuple([index])
            if not isinstance(index, (list, tuple))
            else tuple(index)
            for key, index in arg_slice.items()
        }

    def __call__(
        self,
        **kwargs,
    ) -> jnp.ndarray:

        # slice the arguments
        for key, slices in self.arg_slice.items():
            for index in slices:
                kwargs[key] = kwargs[key][index]

        return self.loss(**kwargs)

    def call(self, **kwargs) -> jnp.ndarray:
        return self.loss.call(**kwargs)
