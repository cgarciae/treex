import threading
import typing as tp
from abc import ABC, abstractmethod


import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class Context(threading.local):
    is_slicing: bool = False
    is_merging: bool = False
    is_initializing: bool = False


LOCAL: Context = Context()


class Box:
    def __init__(self, value: tp.Type):
        self.value = value

    def __getitem__(self, *items):
        return self.value

    def unwrap(self) -> tp.Type:
        if isinstance(self.value, Box):
            return self.value.unwrap()
        else:
            return self.value


class IdentityGetter:
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return item[-1]
        else:
            return item


IDENTITY_GETTER = IdentityGetter()


A = tp.TypeVar("A")
B = tp.TypeVar("B")
T = tp.TypeVar("T", bound="Module")
S = tp.TypeVar("S", bound="Sliceable")


class _Optional:
    def __getitem__(self, item: T) -> tp.Type[tp.Optional[T]]:
        return None


class TreePart:
    pass


def annotation(static: tp.Type[A], real: tp.Any, generic: bool = False) -> tp.Type[A]:

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


# simple types
Parameter = annotation(tp.Union[np.ndarray, "Initializer"], _Parameter)
State = annotation(tp.Union[np.ndarray, "Initializer"], _State)
Rng = annotation(tp.Union[np.ndarray, "Initializer"], _Rng)

# composite types
List = annotation(tp.List[A], IDENTITY_GETTER, generic=True)[A]
Dict = annotation(tp.Dict[A, B], IDENTITY_GETTER, generic=True)[A, B]
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
    def __init__(self, f: tp.Callable[[jnp.ndarray], tp.Any]):
        self.f = f

    def __call__(self, x: jnp.ndarray) -> np.ndarray:
        return self.f(x)

    def __repr__(self) -> str:
        return "Initializer"


class Module:
    _initialized = False

    def init(self: T, key: tp.Union[int, jnp.ndarray]) -> T:
        if isinstance(key, int):
            key = jax.random.PRNGKey(key)

        def init_fn(x):
            nonlocal key
            key = tp.cast(jnp.ndarray, key)

            if isinstance(x, Initializer):
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

        if LOCAL.is_initializing and not module._initialized:
            module.post_init()
            module._initialized = True

        return module

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node_class(cls)

    def copy(self: T) -> T:
        return jax.tree_map(lambda x: x, self)

    def slice(self: T, *filters: tp.Type) -> T:
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
