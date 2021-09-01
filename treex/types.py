import typing as tp
from abc import ABCMeta

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

A = tp.TypeVar("A")


class IdentityGeneric(ABCMeta):
    def __getitem__(cls: tp.Type[A], *key: tp.Any) -> tp.Type[A]:
        return cls


class TreePart(metaclass=IdentityGeneric):
    pass


class _Parameter(TreePart):
    pass


class _State(TreePart):
    pass


class _Rng(_State):
    pass


class _ModelState(_State):
    pass


class _BatchStat(_ModelState):
    pass


class _Cache(_ModelState):
    pass


class _Log(_State):
    pass


class _Loss(_Log):
    pass


class _Metric(_Log):
    pass


class _OptState(_State):
    pass


class _Static(metaclass=IdentityGeneric):
    pass


# value annotation
class _ValueAnnotation(tp.Generic[A]):
    def __init__(self, value, annotation: tp.Type[A]):
        self.value = value
        self.annotation = annotation


# use cast to trick static analyzers into believing these types
Parameter = tp.Union  # static
globals()["Parameter"] = _Parameter  # real

State = tp.Union  # static
globals()["State"] = _State  # real

Rng = tp.Union  # static
globals()["Rng"] = _Rng  # real

ModelState = tp.Union  # static
globals()["ModelState"] = _ModelState  # real

BatchStat = tp.Union  # static
globals()["BatchStat"] = _BatchStat  # real

Cache = tp.Union  # static
globals()["Cache"] = _Cache  # real

Log = tp.Union  # static
globals()["Log"] = _Log  # real

Loss = tp.Union  # static
globals()["Loss"] = _Loss  # real

Metric = tp.Union  # static
globals()["Metric"] = _Metric  # real

OptState = tp.Union  # static
globals()["OptState"] = _OptState  # real

Static = tp.Union  # static
globals()["Static"] = _Static  # real


class Initializer:
    """Initialize a field from a function that expects a single argument with a PRNGKey.

    Initializers are called by `Module.init` and replace the value of the field they are assigned to.
    """

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


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls()

    def __repr__(self) -> str:
        return "Nothing"

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Nothing)
