import typing as tp
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


A = TypeVar("A")
T = TypeVar("T", bound="Module")
S = TypeVar("S", bound="Sliceable")


class TreePart:
    pass


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux_data, _children):
        return cls()

    def __repr__(self) -> str:
        return "Nothing"


class Sliceable(TreePart, ABC):
    @abstractmethod
    def slice(self: S, *filters: Type) -> S:
        raise NotImplementedError()

    @abstractmethod
    def _merge_one(self: S, other: S) -> S:
        raise NotImplementedError()


class Initializer(TreePart):
    def __init__(self, f: Callable[[jnp.ndarray], np.ndarray]):
        self.f = f

    def __call__(self, x: jnp.ndarray) -> np.ndarray:
        return self.f(x)

    def __repr__(self) -> str:
        return "Initializer"


def annotation(
    names: str, static: Type[A], real: Type = TreePart, generic: bool = False
) -> Type[A]:
    t = type(names, (real,), {})

    if generic:
        return Box(Box(t))
    else:
        return t


class Module(Sliceable):
    def init(self: T, key: Union[int, jnp.ndarray]) -> T:
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        def init_fn(x):
            nonlocal key
            key = tp.cast(jnp.ndarray, key)

            if isinstance(x, Initializer):
                key, next_key = jax.random.split(key, 2)
                x = x(next_key)

            return x

        return jax.tree_map(init_fn, self)

    def _parts(self) -> Tuple[Dict[str, Tuple[Type[TreePart], Any]], Dict[str, Any]]:

        annotations = getattr(self.__class__, "__annotations__", {})
        annotations = {
            k: v.unwrap() if isinstance(v, Box) else v for k, v in annotations.items()
        }
        fields = vars(self)

        tree_parts = {
            k: (cls, fields[k])
            for k, cls in annotations.items()
            if issubclass(cls, TreePart)
        }
        not_tree = {k: v for k, v in fields.items() if k not in tree_parts}

        return tree_parts, not_tree

    def tree_flatten(self):
        tree_parts, not_tree = self._parts()

        tree_parts = {k: v for k, (_cls, v) in tree_parts.items()}

        return tuple(tree_parts.values()), dict(
            tree_parts=tree_parts.keys(), not_tree=not_tree
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for i, k in enumerate(aux_data["tree_parts"]):
            setattr(module, k, children[i])

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def slice(self: T, *filters: Type) -> T:
        module: T = self.__class__.__new__(self.__class__)
        tree_parts, not_tree = self._parts()

        for k, (cls, v) in tree_parts.items():
            if issubclass(cls, Sliceable):
                v = cls.slice(v, *filters)
            elif not issubclass(cls, filters):
                v = Nothing()

            setattr(module, k, v)

        for k, v in not_tree.items():
            setattr(module, k, v)

        return module

    def _merge_one(self: T, other: T) -> T:
        module: T = self.__class__.__new__(self.__class__)
        tree_parts, not_tree = self._parts()

        for k, (cls, v1) in tree_parts.items():
            v2 = getattr(other, k)

            if (
                issubclass(cls, Module)
                and isinstance(v1, Module)
                and isinstance(v2, Module)
            ):
                v = cls._merge_one(v1, v2)
            else:
                v = v2 if not isinstance(v2, Nothing) else v1

            setattr(module, k, v)

        for k, v in not_tree.items():
            setattr(module, k, v)

        return module

    def merge(self: T, other: T, *rest: T) -> T:
        others = (other,) + rest
        acc = self

        for other in others:
            acc = acc._merge_one(other)

        return acc


### utils


class Box:
    def __init__(self, value: Type):
        self.value = value

    def __getitem__(self, item):
        return self.value

    def unwrap(self) -> Type:
        if isinstance(self.value, Box):
            return self.value.unwrap()
        else:
            return self.value


class _TreeList(List[S], Sliceable):
    def slice(self, *filters: Type) -> "List[S]":
        return [x.slice(*filters) for x in self]

    def _merge_one(self, other: List[S]) -> List[S]:
        return [a._merge_one(b) for a, b in zip(self, other)]


Parameter = annotation("Parameter", tp.Union[np.ndarray, Initializer])
State = annotation("State", tp.Union[np.ndarray, Initializer])
Rng = annotation("Rng", State, State)
ModuleList = annotation("ModuleList", List, _TreeList, generic=True)[S]
