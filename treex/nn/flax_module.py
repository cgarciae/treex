import typing as tp

import flax
import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from flax.core.frozen_dict import FrozenDict
from flax.core.scope import FrozenVariableDict

from treex import types, utils
from treex.key_seq import KeySeq
from treex.module import Module


class FlaxModule(Module):

    # static
    module: to.Hashable[flax.linen.Module]
    mutable: tp.Tuple[str, ...]
    rngs: tp.Tuple[str, ...]
    init_rngs: tp.Tuple[str, ...]

    # dynamic
    _params: tp.Optional[tp.Dict[str, tp.Any]] = types.Parameter.node()
    _batch_stats: tp.Optional[tp.Dict[str, tp.Any]] = types.BatchStat.node()
    _cache: tp.Optional[tp.Dict[str, tp.Any]] = types.Cache.node()
    _variables: tp.Union[tp.Dict[str, tp.Dict[str, tp.Any]], None] = types.Log.node()
    next_key: KeySeq

    def __init__(
        self,
        module: flax.linen.Module,
        mutable: tp.Sequence[str] = ("batch_stats", "cache"),
        rngs: tp.Sequence[str] = ("dropout",),
        init_rngs: tp.Sequence[str] = ("params",),
        variables: tp.Optional[FrozenDict] = None,
        method: tp.Optional[str] = None,
    ) -> None:

        self.module = to.Hashable(module)
        self.mutable = tuple(mutable)
        self.rngs = tuple(rngs)
        self.init_rngs = tuple(init_rngs)
        self.next_key = KeySeq()
        self._params = None
        self._batch_stats = None
        self._cache = None
        self._variables = None
        self.method = method if method is not None else "__call__"

        if variables is not None:
            self._update_variables(variables)

    def __call__(self, *args, **kwargs):

        method: tp.Callable = getattr(self.module.value, self.method)

        if "training" not in kwargs:
            arg_names = utils._function_argument_names(method)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else False

        if self.initializing() and self._variables is None:
            rngs = self._get_rngs(self.rngs + self.init_rngs)
            output, _variables = self.module.value.init_with_output(
                rngs,
                *args,
                method=method,
                **kwargs,
            )
            self._update_variables(_variables)
            return output

        assert self._variables is not None
        variables = self._variables.copy()

        if self._params is not None:
            variables["params"] = self._params

        if self._batch_stats is not None:
            variables["batch_stats"] = self._batch_stats

        if self._cache is not None:
            variables["cache"] = self._cache

        rngs = self._get_rngs(self.rngs)

        output, updates = self.module.value.apply(
            variables,
            *args,
            mutable=self.mutable
            if self.initialized and self.training and not self.frozen
            else [],
            rngs=rngs,
            method=method,
            **kwargs,
        )
        variables.update(updates.unfreeze())
        self._update_variables(variables)

        return output

    def _get_rngs(self, collections: tp.Sequence[str]):
        all_collections = tuple(collections)

        if len(all_collections) == 0:
            rngs = {}
        elif len(all_collections) == 1:
            key = self.next_key()
            rngs = {all_collections[0]: key}
        elif len(all_collections) > 1:
            key = self.next_key()
            keys = utils.iter_split(key, len(all_collections))
            rngs = dict(zip(all_collections, keys))
        else:
            raise Exception("Not reachable")

        return rngs

    def _update_variables(
        self, variables: tp.Mapping[str, tp.Mapping[str, tp.Any]]
    ) -> None:

        if isinstance(variables, FrozenDict):
            variables = variables.unfreeze()

        assert isinstance(variables, dict)
        variables = tp.cast(tp.Dict[str, tp.Dict[str, tp.Any]], variables)

        if "params" in variables:
            self._params = variables.pop("params")

        if "batch_stats" in variables:
            self._batch_stats = variables.pop("batch_stats")

        if "cache" in variables:
            self._cache = variables.pop("cache")

        self._variables = variables
