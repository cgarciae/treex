import threading
import typing as tp
from abc import ABC, abstractmethod
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
        first = True

        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        def init_fn(x):
            nonlocal key, first

            key = tp.cast(jnp.ndarray, key)

            if isinstance(x, types.Initializer):
                if first:
                    next_key = key
                    first = False
                else:
                    key, next_key = jax.random.split(key, 2)

                x = x(next_key)

            return x

        old_initializing = LOCAL.is_initializing
        LOCAL.is_initializing = True

        try:
            module = jax.tree_map(init_fn, self)
        finally:
            LOCAL.is_initializing = old_initializing

        return module

    def post_init(self):
        pass

    def tree_flatten(self):
        annotations = getattr(self.__class__, "__annotations__", {})
        fields = vars(self)

        tree = {}
        not_tree = {}

        for name, value in fields.items():
            annotation = annotations.get(name, None)

            if isinstance(value, Module):
                tree[name] = value
            elif annotation is not None and issubclass(annotation, types.TreePart):
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
            module.post_init()
            module._initialized = True

        if LOCAL.training is not None:
            module._training = LOCAL.training

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def slice(self: T, *filters: tp.Type) -> T:
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
