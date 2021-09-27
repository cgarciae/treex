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
    sample_inputs: tp.Optional[types.Inputs]
    mutable: tp.Tuple[str, ...]
    rngs: tp.Tuple[str, ...]
    init_rngs: tp.Tuple[str, ...]

    # dynamic
    params: tp.Optional[tp.Dict[str, tp.Any]] = types.Parameter.node()
    batch_stats: tp.Optional[tp.Dict[str, tp.Any]] = types.BatchStat.node()
    cache: tp.Optional[tp.Dict[str, tp.Any]] = types.Cache.node()
    variables: tp.Union[tp.Dict[str, tp.Dict[str, tp.Any]], None] = types.Log.node()
    next_key: KeySeq

    def __init__(
        self,
        module: flax.linen.Module,
        sample_inputs: tp.Optional[types.Inputs] = None,
        mutable: tp.Sequence[str] = ("batch_stats", "cache"),
        rngs: tp.Sequence[str] = ("dropout",),
        init_rngs: tp.Sequence[str] = ("params",),
        variables: tp.Optional[FrozenDict] = None,
    ) -> None:

        self.module = to.Hashable(module)
        self.mutable = tuple(mutable)
        self.rngs = tuple(rngs)
        self.init_rngs = tuple(init_rngs)
        self.sample_inputs = (
            sample_inputs if sample_inputs is not None else types.Inputs()
        )
        self.next_key = KeySeq()
        self.params = None
        self.batch_stats = None
        self.cache = None
        self.variables = None

        if variables is not None:
            self._update_variables(variables)

    def rng_init(self, key: jnp.ndarray) -> None:
        if self.variables is None:
            assert self.sample_inputs is not None
            rngs = self._get_rngs(self.rngs + self.init_rngs)
            variables = self.module.value.init(
                rngs, *self.sample_inputs.args, **self.sample_inputs.kwargs
            )
            self._update_variables(variables)
            self.sample_inputs = None

    def __call__(self, *args, **kwargs):
        assert self.variables is not None
        variables = self.variables.copy()

        if self.params is not None:
            variables["params"] = self.params

        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats

        if self.cache is not None:
            variables["cache"] = self.cache

        rngs = self._get_rngs(self.rngs)

        output, updates = self.module.value.apply(
            variables,
            *args,
            mutable=self.mutable,
            rngs=rngs,
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
            self.params = variables.pop("params")

        if "batch_stats" in variables:
            self.batch_stats = variables.pop("batch_stats")

        if "cache" in variables:
            self.cache = variables.pop("cache")

        self.variables = variables
