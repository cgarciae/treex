import dataclasses
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey
_pymap = map
_pyfilter = filter


def field(
    default=dataclasses.MISSING,
    *,
    node: bool,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
) -> tp.Any:
    return dataclasses.field(
        default=default,
        metadata={"node": node, "kind": kind},
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def node(
    default=dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
) -> tp.Any:
    return field(
        default=default,
        node=True,
        kind=kind,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


def static(
    default=dataclasses.MISSING,
    *,
    kind: type = type(None),
    default_factory=dataclasses.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: tp.Optional[bool] = None,
    compare: bool = True,
) -> tp.Any:
    return field(
        default,
        node=False,
        kind=kind,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
    )


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
