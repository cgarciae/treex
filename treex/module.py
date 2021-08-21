import threading
import typing as tp
from abc import ABC, abstractmethod

from jax.nn import initializers
from treex import types


import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class Context(threading.local):
    is_slicing: bool = False
    is_merging: bool = False
    is_initializing: bool = False
    training: tp.Optional[bool] = None
    key: tp.Optional[jnp.ndarray] = None

    def next_key(self) -> jnp.ndarray:
        assert self.is_initializing and self.key is not None

        # key = self.key
        key, self.key = jax.random.split(self.key)

        return key


LOCAL: Context = Context()

A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="Module")


@jax.tree_util.register_pytree_node_class
class Nothing:
    def tree_flatten(self):
        children = (types.Dummy(),) if LOCAL.is_merging else ()
        return children, None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        if LOCAL.is_merging:
            value = children[0]

            if not isinstance(value, types.Dummy):
                return value

        return cls()

    def __repr__(self) -> str:
        return "Nothing"


class Module:
    _initialized = False
    _training = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def training(self) -> bool:
        return self._training

    def init(self: T, key: tp.Union[int, jnp.ndarray]) -> T:

        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        old_initializing = LOCAL.is_initializing
        old_key = LOCAL.key

        LOCAL.is_initializing = True
        LOCAL.key = key

        try:
            module = jax.tree_map(
                lambda initializer: (
                    initializer(LOCAL.next_key())
                    if isinstance(initializer, types.Initializer)
                    else initializer
                ),
                self,
            )
        finally:
            LOCAL.is_initializing = old_initializing
            LOCAL.key = old_key

        return module

    def module_init(self, key: jnp.ndarray) -> None:
        pass

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        tree = {}
        not_tree = {}

        for name, value in fields.items():
            annotation = annotations.get(name, None)
            annotation = _resolve_tree_type(name, annotation)

            if annotation is None:
                not_tree[name] = value

            elif issubclass(annotation, Module):
                tree[name] = value

            elif issubclass(annotation, types.TreePart):
                if LOCAL.is_slicing:
                    tree[name] = jax.tree_map(
                        lambda x: types.ValueAnnotation(x, annotation), value
                    )
                else:
                    tree[name] = value
            else:
                not_tree[name] = value

        return tuple(tree.values()), dict(
            tree=tree.keys(),
            not_tree=not_tree,
            props=dict(
                _initialized=self._initialized,
                _training=self._training,
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        module = cls.__new__(cls)

        for i, k in enumerate(aux_data["tree"]):
            setattr(module, k, children[i])

        for k, v in aux_data["not_tree"].items():
            setattr(module, k, v)

        for k, v in aux_data["props"].items():
            setattr(module, k, v)

        if LOCAL.is_initializing and not module._initialized:
            module.module_init(LOCAL.next_key())
            module._initialized = True

        if LOCAL.training is not None:
            module._training = LOCAL.training

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def filter(self: T, *filters: tp.Type) -> T:
        flat: tp.List[types.ValueAnnotation]

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

    def update(self: T, other: T, *rest: T) -> T:
        modules = (self, other) + rest

        old_merging = LOCAL.is_merging
        LOCAL.is_merging = True

        def merge_fn(*xs):
            acc, *xs = xs
            for x in xs:
                if not isinstance(x, types.Dummy):
                    acc = x
            return acc

        try:
            flats, treedefs = zip(*[jax.tree_flatten(m) for m in modules])
            flat_out = jax.tree_util.tree_map(merge_fn, *flats)
            module = jax.tree_unflatten(treedefs[0], flat_out)
        finally:
            LOCAL.is_merging = old_merging

        return module

    def train(self: T, mode: bool = True) -> T:
        old_training = LOCAL.training
        LOCAL.training = mode

        try:
            module = self.copy()  # trigger flatten / unflatten
        finally:
            LOCAL.training = old_training

        return module

    def eval(self: T) -> T:
        return self.train(False)


def _resolve_tree_type(name: str, t: tp.Optional[type]) -> tp.Optional[type]:
    if t is None:
        return None

    tree_types = [x for x in _all_types(t) if issubclass(x, (types.TreePart, Module))]

    if len(tree_types) > 1:
        # if its a type with many Module subtypes just mark them all as Module
        if all(issubclass(x, Module) for x in tree_types):
            return Module
        else:
            raise TypeError(
                f"Multiple tree parts found in annotation for field '{name}': {tree_types}"
            )
    elif len(tree_types) == 1:
        return tree_types[0]
    else:
        return t


def _all_types(t: tp.Type) -> tp.Iterable[tp.Type]:
    if hasattr(t, "__args__"):
        for arg in t.__args__:
            yield from _all_types(arg)
    else:
        yield t
