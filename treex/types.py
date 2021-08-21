import typing as tp

import jax.numpy as jnp
import numpy as np


class TreePart:
    pass


class _Parameter(TreePart):
    pass


class _State(TreePart):
    pass


class _Rng(_State):
    pass


# use cast to trick static analyzers into believing these types
Parameter = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Parameter)
State = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _State)
Rng = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Rng)


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
