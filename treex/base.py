import threading
import typing as tp
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class Context(threading.local):
    is_slicing: bool = False
    is_merging: bool = False


LOCAL: Context = Context()


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


class IdentityGetter:
    def __getitem__(self, item):
        return item


IDENTITY_GETTER = IdentityGetter()


A = TypeVar("A")
T = TypeVar("T", bound="Module")
S = TypeVar("S", bound="Sliceable")


class TreePart:
    pass


def annotation(static: Type[A], real: Any, generic: bool = False) -> Type[A]:

    if generic:
        return Box(real)
    else:
        return real


class _Parameter(TreePart):
    pass


class _State(TreePart):
    pass


class _Rng(_State):
    pass


class ModuleContainer(TreePart):
    pass


Parameter = annotation(tp.Union[np.ndarray, "Initializer"], _Parameter)
State = annotation(tp.Union[np.ndarray, "Initializer"], _State)
Rng = annotation(tp.Union[np.ndarray, "Initializer"], _Rng)
List = annotation(tp.List[A], IDENTITY_GETTER, generic=True)[A]
ModuleList = annotation(tp.List[T], Box(ModuleContainer), generic=True)[T]


class ValueAnnotation:
    def __init__(self, value, annotation):
        self.value = value
        self.annotation = annotation


class Dummy:
    pass


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        children = (Dummy(),) if LOCAL.is_merging else ()
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        if LOCAL.is_merging:
            value = children[0]

            if not isinstance(value, Dummy):
                return value

        return cls()

    def __repr__(self) -> str:
        return "Nothing"


class Initializer:
    def __init__(self, f: Callable[[jnp.ndarray], np.ndarray]):
        self.f = f

    def __call__(self, x: jnp.ndarray) -> np.ndarray:
        return self.f(x)

    def __repr__(self) -> str:
        return "Initializer"


class Module:
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

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        tree = {}
        not_tree = {}

        for name, value in fields.items():
            annotation = annotations.get(name, None)

            if isinstance(value, Module):
                tree[name] = value
            elif annotation is not None and issubclass(annotation, TreePart):
                if LOCAL.is_slicing:
                    tree[name] = jax.tree_map(
                        lambda x: ValueAnnotation(x, annotation), value
                    )
                else:
                    tree[name] = value
            else:
                not_tree[name] = value

        return tuple(tree.values()), dict(tree=tree.keys(), not_tree=not_tree)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for i, k in enumerate(aux_data["tree"]):
            setattr(module, k, children[i])

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def slice(self: T, *filters: Type) -> T:
        flat: tp.List[ValueAnnotation]

        old_slicing = LOCAL.is_slicing
        LOCAL.is_slicing = True

        try:
            flat, treedef = jax.tree_flatten(self)
            flat_out = [
                value_annotation.value
                if issubclass(value_annotation.annotation, filters)
                else Nothing()
                for value_annotation in flat
            ]
            module = jax.tree_unflatten(treedef, flat_out)
        finally:
            LOCAL.is_slicing = old_slicing

        return module

    def merge(self: T, other: T, *rest: T) -> T:
        modules = (self, other) + rest

        old_merging = LOCAL.is_merging
        LOCAL.is_merging = True

        def merge_fn(*xs):
            acc, *xs = xs
            for x in xs:
                if not isinstance(x, Dummy):
                    acc = x
            return acc

        try:
            flats, treedefs = zip(*[jax.tree_flatten(m) for m in modules])
            flat_out = jax.tree_util.tree_map(merge_fn, *flats)
            module = jax.tree_unflatten(treedefs[0], flat_out)
        finally:
            LOCAL.is_merging = old_merging

        return module


### utils
