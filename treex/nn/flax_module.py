import typing as tp

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from treex import types
from treex.module import Module
from treex.rnq_seq import RngSeq


class FlaxModule(Module):

    # static
    module: types.Hashable[flax.linen.Module]
    sample_inputs: tp.Optional[types.Inputs]
    mutable: tp.Tuple[str, ...]
    rngs: tp.Tuple[str, ...]

    # dynamic
    params: types.Parameter[tp.Mapping[str, tp.Any], None]
    batch_stats: types.BatchStat[tp.Mapping[str, tp.Any], None]
    cache: types.Cache[tp.Mapping[str, tp.Any], None]
    variables: types.TreePart[tp.Mapping[str, tp.Mapping[str, tp.Any]], None]
    rng_seq: RngSeq

    def __init__(
        self,
        module: flax.linen.Module,
        sample_inputs: tp.Optional[types.Inputs] = None,
        mutable: tp.Sequence[str] = ("params", "batch_stats", "cache"),
        rngs: tp.Sequence[str] = ("dropout",),
        variables: tp.Optional[FrozenDict] = None,
    ) -> None:
        super().__init__()
        self.module = types.Hashable(module)
        self.mutable = tuple(mutable)
        self.rngs = tuple(rngs)
        self.sample_inputs = (
            sample_inputs if sample_inputs is not None else types.Inputs()
        )
        self.rng_seq = RngSeq()
        self.params = None
        self.batch_stats = None
        self.cache = None
        self.variables = None

        if variables is not None:
            self._update_variables(variables)

    def module_init(self, key: jnp.ndarray) -> None:
        if self.variables is None:
            assert self.sample_inputs is not None
            rngs = self._get_rngs(self.rngs + self.mutable)
            variables = self.module.value.init(
                rngs, *self.sample_inputs.args, **self.sample_inputs.kwargs
            )
            self._update_variables(variables)
            self.sample_inputs = None

    def __call__(self, *args, **kwargs):

        variables = dict(**self.variables)

        if self.params is not None:
            variables["params"] = self.params

        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats

        if self.cache is not None:
            variables["cache"] = self.cache

        rngs = self._get_rngs(self.rngs)

        output, variables = self.module.value.apply(
            variables,
            *args,
            mutable=self.mutable,
            rngs=rngs,
            **kwargs,
        )
        self._update_variables(variables)

        return output

    def _get_rngs(self, collections: tp.Sequence[str]):
        all_collections = tuple(collections)

        if len(all_collections) == 0:
            rngs = {}
        elif len(all_collections) == 1:
            key = self.rng_seq.next()
            rngs = {all_collections[0]: key}
        elif len(all_collections) > 1:
            key = self.rng_seq.next()
            keys = jax.random.split(key, len(all_collections))
            rngs = dict(zip(all_collections, keys))
        else:
            raise Exception("Not reachable")

        return rngs

    def _update_variables(self, variables: FrozenDict) -> None:

        if "params" in variables:
            params: FrozenDict
            variables, params = variables.pop("params")
            self.params = params.unfreeze()

        if "batch_stats" in variables:
            batch_stats: FrozenDict
            variables, batch_stats = variables.pop("batch_stats")
            self.batch_stats = batch_stats.unfreeze()

        if "cache" in variables:
            cache: FrozenDict
            variables, cache = variables.pop("cache")
            self.cache = cache.unfreeze()

        self.variables = variables.unfreeze()
