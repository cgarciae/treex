import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import treeo as to
import typing_extensions as tpe

A = tp.TypeVar("A")
B = tp.TypeVar("B")

IndexLike = tp.Union[str, int, tp.Sequence[tp.Union[str, int]]]

# -----------------------------------------
# TreeParts
# -----------------------------------------


class TreePart(to.KindMixin):
    pass


class Parameter(TreePart):
    pass


class State(TreePart):
    pass


class Rng(State):
    pass


class MetricState(State):
    pass


class ModelState(State):
    pass


class BatchStat(ModelState):
    pass


class Cache(ModelState):
    pass


class OptState(State):
    pass


class Log(TreePart):
    pass


class LossLog(Log):
    pass


class MetricLog(Log):
    pass


@dataclass
class Named(to.Tree, tp.Generic[A]):
    value: A = to.node()
    name: str = to.static(opaque=True)


class Initializer(to.Tree):
    """Initialize a field from a function that expects a single argument with a PRNGKey.

    Initializers are called by `Module.init` and replace the value of the field they are assigned to.
    """

    f: tp.Callable[[jnp.ndarray], tp.Any]

    def __init__(self, f: tp.Callable[[jnp.ndarray], tp.Any]):
        """
        Arguments:
            f: A function that takes a PRNGKey and returns the initial value of the field.
        """
        self.f = f

    def __call__(self, x: jnp.ndarray) -> np.ndarray:
        return self.f(x)

    def __repr__(self) -> str:
        return "Initializer"


class Inputs:
    args: tp.Tuple[tp.Any, ...]
    kwargs: tp.Dict[str, tp.Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@tpe.runtime_checkable
class WrappedCall(tpe.Protocol):
    _orig_call: tp.Callable[..., tp.Any]

    def __call__(self, *args, **kwargs) -> tp.Any:
        ...
