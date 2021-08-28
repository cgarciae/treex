import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np


class TreePart:
    pass


class _Parameter(TreePart):
    pass


class _State(TreePart):
    pass


class _Rng(_State):
    pass


class _BatchStat(_State):
    pass


# use cast to trick static analyzers into believing these types
Parameter = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Parameter)
State = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _State)
Rng = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Rng)
BatchStat = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _BatchStat)


class _ValueAnnotation:
    def __init__(self, value, annotation):
        self.value = value
        self.annotation = annotation


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
