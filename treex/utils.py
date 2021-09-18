import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey
_pymap = map
_pyfilter = filter


def dynamic(
    default=dataclasses.MISSING,
    *,
    dynamic: bool = True,
    tree_type: type = type(None),
    **kwargs,
) -> tp.Any:
    return dataclasses.field(
        default=default,
        metadata={"dynamic": dynamic, "tree_type": tree_type},
        **kwargs,
    )


def static(
    default=dataclasses.MISSING,
    *,
    tree_type: type = type(None),
    **kwargs,
) -> tp.Any:
    return dynamic(default, dynamic=False, tree_type=tree_type, **kwargs)


def _get_all_annotations(cls: type) -> tp.Dict[str, type]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            d.update(**c.__annotations__)
    return d


def _get_all_vars(cls: type) -> tp.Dict[str, tp.Any]:
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


def _all_types(t: type) -> tp.Iterable[type]:
    return _pyfilter(lambda t: isinstance(t, tp.Type), _all_types_unfiltered(t))


def _all_types_unfiltered(t: type) -> tp.Iterable[type]:
    yield t

    if hasattr(t, "__args__"):
        yield t.__origin__

        for arg in t.__args__:
            yield from _all_types_unfiltered(arg)
