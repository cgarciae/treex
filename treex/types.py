import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


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


class Initializer:
    def __init__(self, f: tp.Callable[[jnp.ndarray], tp.Any]):
        self.f = f

    def __call__(self, x: jnp.ndarray) -> np.ndarray:
        return self.f(x)

    def __repr__(self) -> str:
        return "Initializer"
