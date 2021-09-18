import dataclasses
import typing as tp
from abc import ABCMeta
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

from treex import utils

A = tp.TypeVar("A")
B = tp.TypeVar("B")


@tp.runtime_checkable
class ArrayLike(tp.Protocol):
    shape: tp.Tuple[int, ...]
    dtype: np.dtype


# -----------------------------------------
# TreeParts
# -----------------------------------------


class FieldMixin:
    @classmethod
    def field(
        cls,
        default=dataclasses.MISSING,
        *,
        node: bool,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
    ) -> tp.Any:
        return utils.field(
            default=default,
            node=node,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
        )

    @classmethod
    def node(
        cls,
        default=dataclasses.MISSING,
        *,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
    ) -> tp.Any:
        return utils.node(
            default=default,
            kind=cls,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
        )

    @classmethod
    def static(
        cls,
        default=dataclasses.MISSING,
        default_factory=dataclasses.MISSING,
        init: bool = True,
        repr: bool = True,
        hash: tp.Optional[bool] = None,
        compare: bool = True,
    ) -> tp.Any:
        return cls.field(
            default=default,
            node=False,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
        )


class TreePart(FieldMixin):
    pass


class Parameter(TreePart):
    pass


class State(TreePart):
    pass


class Rng(State):
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


class Loss(Log):
    pass


class Metric(Log):
    pass


class TrivialPytree:
    def tree_flatten(self):
        tree = vars(self)
        children = (tree,)
        return (children, ())

    @classmethod
    def tree_unflatten(cls, aux, children):
        (tree,) = children

        obj = cls.__new__(cls)
        obj.__dict__.update(tree)

        return obj

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)


@dataclass
class FieldMetadata(TrivialPytree):
    node: bool
    kind: type


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


class Missing:
    pass
