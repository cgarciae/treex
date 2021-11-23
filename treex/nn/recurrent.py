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


class GRUCell(Module):
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

    # TODO: Might remove the rand_key depending on how it affects the API
    next_key: KeySeq

    # TODO: Add typing info and documentation
    def __init__(
        self,
        *,
        gate_fn: CallableModule = flax_module.sigmoid,
        activation_fn: CallableModule = flax_module.tanh,
        kernel_init: CallableModule = flax_module.default_kernel_init,
        recurrent_kernel_init=flax_module.orthogonal(),
        bias_init=flax_module.zeros
    ):
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.next_key = KeySeq()

    @to.compact
    def __call__(self, carry: jnp.ndarray, inputs: jnp.ndarray):
        hidden_features = carry.shape[-1]
        dense_h = partial(
            Linear,
            features_out=hidden_features,
            use_bias=False,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
        )
        dense_i = partial(
            Linear,
            features_out=hidden_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        r = self.gate_fn(dense_i(name="ir")(inputs) + dense_h(name="hr")(carry))
        z = self.gate_fn(dense_i(name="iz")(inputs) + dense_h(name="hz")(carry))
        n = self.activation_fn(
            dense_i(name="in")(inputs) + r * dense_h(name="hn", use_bias=True)(carry)
        )
        new_h = (1.0 - z) * n + z * carry
        return new_h, new_h

    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=flax_module.zeros):
        mem_shape = batch_dims + (size,)
        return init_fn(rng, mem_shape)

    def init_carry(self, batch_dims, init_fn=flax_module.zeros):
        # TODO Check if `module.init(key, ...)` has been called
        return GRUCell.initialize_carry(
            self.next_key(), batch_dims, self.hn.kernel.shape[0], init_fn
        )
