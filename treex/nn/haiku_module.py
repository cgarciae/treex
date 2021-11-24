import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to

from treex import types, utils
from treex.key_seq import KeySeq
from treex.module import Module

try:
    import haiku as hk
    from haiku.data_structures import to_mutable_dict
except (ImportError, ModuleNotFoundError):
    raise types.OptionalDependencyNotFound("Haiku Unavailable")


class HaikuModule(Module):

    # static
    transform: hk.TransformedWithState

    # dynamic
    params_: tp.Optional[hk.Params] = types.Parameter.node()
    state_: tp.Optional[hk.State] = types.BatchStat.node()
    next_key: KeySeq

    def __init__(
        self,
        transform: tp.Union[
            hk.TransformedWithState,
            tp.Callable[..., tp.Any],
        ],
        params: tp.Optional[hk.Params] = None,
        states: tp.Optional[hk.State] = None,
    ) -> None:

        self.transform = (
            hk.transform_with_state(transform)
            if not isinstance(transform, hk.TransformedWithState)
            else transform
        )
        self.next_key = KeySeq()
        self.params_ = to_mutable_dict(params) if params is not None else None
        self.state_ = to_mutable_dict(states) if states is not None else None

    def __call__(self, *args, **kwargs):

        key = self.next_key()

        if "training" not in kwargs:
            _original_fn = self.transform.init._original_fn
            arg_names = utils._function_argument_names(_original_fn)

            if arg_names is not None and "training" in arg_names:
                kwargs["training"] = self.training if self.initialized else True

        if self.initializing() and self.params_ is None and self.state_ is None:
            self.params_, self.state_ = self.transform.init(
                key,
                *args,
                **kwargs,
            )
            output, _ = self.transform.apply(
                self.params_, self.state_, key, *args, **kwargs
            )
            self.params_ = to_mutable_dict(self.params_)
            self.state_ = to_mutable_dict(self.state_)
            return output

        output, next_state = self.transform.apply(
            self.params_, self.state_, key, *args, **kwargs
        )

        self.state_ = to_mutable_dict(next_state)

        return output
