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
    initial_state_init: tp.Callable[
        [flax_module.PRNGKey, flax_module.Shape, flax_module.Dtype], flax_module.Array
    ]
    params: tp.Dict[str, tp.Dict[str, flax_module.Array]] = types.Parameter.node()

    # static
    hidden_units: int
    return_state: bool
    return_sequences: bool
    go_backwards: bool
    time_major: bool
    unroll: int

    def __init__(
        self, units: int,
        *,
        gate_fn: CallableModule = flax_module.sigmoid,
        activation_fn: CallableModule = flax_module.tanh,
        kernel_init: CallableModule = flax_module.default_kernel_init,
        recurrent_kernel_init: CallableModule = flax_module.orthogonal(),
        bias_init: CallableModule = flax_module.zeros,
        initial_state_init: CallableModule = flax_module.zeros,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        time_major: bool = False,
        unroll: int = 1
    ):
        self.hidden_units = units
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.initial_state_init = initial_state_init
        self.params = {}
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.time_major = time_major
        self.unroll = unroll

        self.next_key = KeySeq()

    @property
    def module(self):
        return flax_module.GRUCell(
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def initialize_state(self, batch_size):
        return self.module.initialize_carry(self.next_key(), (batch_size,), self.hidden_units, self.initial_state_init)

    def __call__(self, x, initial_state = None):
        # Move time dimension to be the first so it can be looped over
        if not self.time_major:
            x = jnp.transpose(x, (1, 0, 2))

        if self.go_backwards:
            x = x[::-1, :, :]

        if initial_state is None:
            initial_state = self.initialize_state(x.shape[1])

        if self.initializing():
            _variables = self.module.init(next_key(), initial_state, x[0, ...])
            self.params = _variables["params"]

        variables = dict(params=self.params)

        def iter_fn(state, x):
            return self.module.apply(variables, state, x)

        final_state, sequences = jax.lax.scan(iter_fn, initial_state, x, unroll=self.unroll)
        if not self.time_major:
            sequences = jnp.transpose(sequences, (1, 0, 2))

        if self.return_sequences and not self.return_state:
            return sequences
        if self.return_sequences and self.return_state:
            return sequences, final_state
        return final_state
