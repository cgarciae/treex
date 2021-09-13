import typing as tp
from abc import ABCMeta

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

A = tp.TypeVar("A")
B = tp.TypeVar("B")

_GenericAlias = type(tp.List[int])  # int is just a dummy type, it could be anything


class IdentityGeneric(ABCMeta):
    def __getitem__(cls, *keys: tp.Any):
        return _GenericAlias(cls, keys)


class TreePart(object, metaclass=IdentityGeneric):
    pass


class _Parameter(TreePart):
    pass


class _DiffHyperParam(TreePart):
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


class _OptState(_State):
    pass


class _Static(metaclass=IdentityGeneric):
    pass


class _Log(TreePart):
    pass


class _Loss(_Log):
    pass


class _Metric(_Log):
    pass


# use cast to trick static analyzers into believing these types
Parameter = tp.Union  # static
globals()["Parameter"] = _Parameter  # real

DiffHyperParam = tp.Union  # static
globals()["DiffHyperParam"] = _DiffHyperParam  # real

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

OptState = tp.Union  # static
globals()["OptState"] = _OptState  # real

Static = tp.Union  # static
globals()["Static"] = _Static  # real

Log = tp.Union  # static
globals()["Log"] = _Log  # real

Loss = tp.Union  # static
globals()["Loss"] = _Loss  # real

Metric = tp.Union  # static
globals()["Metric"] = _Metric  # real


class Named(tp.Generic[A]):
    def __init__(self, name: str, value: A):
        super().__init__()
        self.name = name
        self.value = value

    def tree_flatten(self):
        tree = (self.value,)
        static = (self.name,)

        return tree, static

    @classmethod
    def tree_unflatten(cls, static, tree):
        module = cls.__new__(cls)
        module.name = static[0]
        module.value = tree[0]
        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)


@jax.tree_util.register_pytree_node_class
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

    # ------------------------
    # Pytree implementation
    # ------------------------
    def tree_flatten(self):
        tree = ()
        static = (self.f,)

        return tree, static

    @classmethod
    def tree_unflatten(cls, static, tree):
        obj = cls.__new__(cls)
        obj.f = static[0]
        return obj


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


class Hashable(tp.Generic[A]):
    """A hashable immutable wrapper around non-hashable values"""

    value: A

    def __init__(self, value: A):
        self.__dict__["value"] = value

    def __setattr__(self, name: str, value: tp.Any) -> None:
        raise AttributeError(f"Hashable is immutable")


class Inputs:
    args: tp.Tuple[tp.Any, ...]
    kwargs: tp.Dict[str, tp.Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
