import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from flax.linen import recurrent as flax_module

from treex import types
from treex.key_seq import KeySeq
from treex.module import Module, next_key
from treex.nn.linear import Linear

CallableModule = tp.Callable[..., jnp.ndarray]


class GRU(Module):
    gate_fn: CallableModule = flax_module.sigmoid
    activation_fn: CallableModule = flax_module.tanh
    kernel_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype], flax_module.Array
    ]
    recurrent_kernel_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype], flax_module.Array
    ]
    bias_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype], flax_module.Array
    ]
    params: tp.Any = types.Parameter.node()

    # static
    return_final: bool = False
    time_major: bool = False
    unroll: int = 1

    # TODO: Add typing info and documentation
    def __init__(
        self,
        *,
        gate_fn: CallableModule = flax_module.sigmoid,
        activation_fn: CallableModule = flax_module.tanh,
        kernel_init: CallableModule = flax_module.default_kernel_init,
        recurrent_kernel_init=flax_module.orthogonal(),
        bias_init=flax_module.zeros,
        return_final: bool = False,
        time_major: bool = False,
        unroll: int = 1
    ):
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.params = {}
        self.return_final = return_final
        self.time_major = time_major
        self.unroll = unroll

    @property
    def module(self):
        return flax_module.GRUCell(
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, carry, x):
        # Move time dimension to be the first so it can be looped over
        if not self.time_major:
            x = jnp.transpose(x, (1, 0, 2))

        if self.initializing():
            _variables = self.module.init(next_key(), carry, x[0, ...])
            self.params = _variables["params"]

        variables = dict(params=self.params)

        def iter_fn(carry, x):
            return self.module.apply(variables, carry, x)

        carry, hidden = jax.lax.scan(iter_fn, carry, x, unroll=self.unroll)
        # hidden = jnp.swapaxes(hidden, 0, self.axis)
        if not self.time_major:
            hidden = jnp.transpose(hidden, (1, 0, 2))

        if self.return_final:
            return carry
        return carry, hidden
