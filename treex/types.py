import typing as tp

import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np

A = tp.TypeVar("A")


class TreePart:
    pass


class _Parameter(TreePart):
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


class _Log(_State):
    pass


class _Loss(_Log):
    pass


class _Metric(_Log):
    pass


class _OptState(_State):
    pass


# value annotation
class _ValueAnnotation(tp.Generic[A]):
    def __init__(self, value, annotation: tp.Type[A]):
        self.value = value
        self.annotation = annotation


# static
class _Static(tp.Generic[A]):
    pass


class _GenericIdentity(tp.Protocol):
    def __getitem__(self, item: tp.Type[A]) -> tp.Type[A]:
        ...


# use cast to trick static analyzers into believing these types
Parameter = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Parameter)
State = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _State)
Rng = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Rng)
ModelState = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _ModelState)
BatchStat = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _BatchStat)
Cache = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Cache)
Log = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Log)
Loss = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Loss)
Metric = tp.cast(tp.Type[tp.Union[np.ndarray, "Initializer"]], _Metric)
OptState = tp.cast(tp.Type[tp.Any], _OptState)
Static = tp.Union  # for Union cast doesn't work, so we modify under the hood later


def _mod_types():
    globals()["Static"] = _Static


_mod_types()


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
