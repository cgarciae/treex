import typing as tp

import jax
import jax.numpy as jnp
import treeo as to

from treex import types
from treex.losses.loss import Loss, Reduction


class L1L2(Loss):
    r"""
    A regularizer that applies both L1 and L2 regularization penalties.

    The L1 regularization penalty is computed as:

    $$
    \ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|
    $$

    The L2 regularization penalty is computed as

    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$
    """

    def __init__(
        self,
        l1=0.0,
        l2=0.0,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):  # pylint: disable=redefined-outer-name
        super().__init__(reduction=reduction, weight=weight, on=on, name=name)

        self.l1 = l1
        self.l2 = l2

    def call(self, parameters: tp.Any) -> jnp.ndarray:
        """
        Computes the L1 and L2 regularization penalty simultaneously.

        Arguments:
            net_params: A structure with all the parameters of the model.
        """
        regularization: jnp.ndarray = jnp.array(0.0)

        if not self.l1 and not self.l2:
            return regularization

        leaves = jax.tree_leaves(parameters)

        if self.l1:
            regularization += self.l1 * sum(jnp.sum(jnp.abs(p)) for p in leaves)

        if self.l2:
            regularization += self.l2 * sum(jnp.sum(jnp.square(p)) for p in leaves)

        return regularization


class L1(L1L2):
    r"""
    Create a regularizer that applies an L1 regularization penalty.

    The L1 regularization penalty is computed as:

    $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$
    """

    def __init__(
        self,
        l: float = 0.01,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):
        super().__init__(l1=l, reduction=reduction, weight=weight, on=on, name=name)


class L2(L1L2):
    r"""
    Create a regularizer that applies an L2 regularization penalty.

    The L2 regularization penalty is computed as:

    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$
    ```
    """

    def __init__(
        self,
        l: float = 0.01,
        reduction: tp.Optional[Reduction] = None,
        weight: tp.Optional[float] = None,
        on: tp.Optional[types.IndexLike] = None,
        name: tp.Optional[str] = None,
    ):
        super().__init__(l2=l, reduction=reduction, weight=weight, on=on, name=name)
