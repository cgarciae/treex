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
    """Gated Recurrent Unit - Cho et al. 2014

    `GRU` is implemented as a wrapper on top of `flax.linen.GRUCell`, providing higher level
    functionality and features similar to what can be found in that of `tf.keras.layers.GRU`.
    """

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
    last_state: flax_module.Array = types.Cache.node()

    # static
    hidden_units: int
    return_state: bool
    return_sequences: bool
    go_backwards: bool
    stateful: bool
    time_axis: tp.Tuple[int]
    unroll: int

    def __init__(
        self,
        units: int,
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
        stateful: bool = False,
        time_axis: int = -2,
        unroll: int = 1
    ):
        """
        Arguments:
            units: dimensionality of the state space
            gate_fn: activation function used for gates. (default: `sigmoid`)
            kernel_init: initializer function for the kernels that transform the input
              (default: `lecun_normal`)
            recurrent_kernel_init: initializer function for the kernels that transform
              the hidden state (default: `orthogonal`)
            bias_init: initializer function for the bias parameters (default: `zeros`)
            initial_state_init: initializer function for the hidden state (default: `zeros`)
            return_sequences: whether to return the last state or the sequences (default: `False`)
            return_state: whether to return the last state in addition to the sequences
              (default: `False`)
            go_backwards: whether to process the input sequence backwards and return the
              reversed sequence (default: `False`)
            stateful: whether to use the last state of the current batch as the start_state
              of the next batch (default: `False`)
            time_axis: specifies which axis of the input corresponds to the timesteps. By default,
              `time_axis = -2` which corresponds to the input being of shape `[..., timesteps, :]`
            unroll: number of iterations to be unrolled into a single XLA iteration using
              `jax.lax.scan` (default: `1`)
        """
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
        self.stateful = stateful
        self.time_axis = (time_axis,)
        self.unroll = unroll

        self.next_key = KeySeq()
        self.last_state = None

    @property
    def module(self) -> flax_module.GRUCell:
        return flax_module.GRUCell(
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def initialize_state(self, batch_dim: tp.Union[tp.Tuple[int], int]) -> jnp.ndarray:
        """Initializes the hidden state of the GRU

        Arguments:
            batch_size: Number of elements in a batch

        Returns:
            The initial hidden state as specified by `initial_state_init`
        """
        if not isinstance(batch_dim, tp.Iterable):
            batch_dim = (batch_dim,)
        return self.module.initialize_carry(
            self.next_key(), batch_dim, self.hidden_units, self.initial_state_init
        )

    def __call__(
        self, x: jnp.ndarray, initial_state: tp.Optional[jnp.ndarray] = None
    ) -> tp.Union[jnp.ndarray, tp.Tuple[jnp.ndarray, jnp.ndarray]]:
        """Applies the GRU to the sequence of inputs `x` starting from the `initial_state`.

        Arguments:
            `x`: sequence of inputs to the GRU
            `initial_state`: optional initial hidden state. If nothing is specified,
              either the `last_state` (i.e. the output from the previous batch is used
              if `stateful == True`) or the `initial_state` is gotten from the `initial_state_init`
              function

        Returns:
            - The final state of the GRU by default
            - The full sequence of states (if `return_sequences == True`)
            - A tuple of both the sequence of states and final state (if both
                `return_state` and `return_sequences` are `True`)
        """
        # Move time axis to be the first so it can be looped over
        # Note: not needed with jax_utils.scan_in_dim
        x = jnp.swapaxes(x, self.time_axis, 0)

        if self.go_backwards:
            x = x[::-1]

        if initial_state is None:
            initial_state = self.initialize_state(x.shape[len(self.time_axis) : -1])
            if self.stateful and self.last_state is not None:
                initial_state = self.last_state

        if self.initializing():
            _variables = self.module.init(
                next_key(), initial_state, x[(0,) * len(self.time_axis)]
            )
            self.params = _variables["params"]

        variables = dict(params=self.params)

        def iter_fn(state, x):
            return self.module.apply(variables, state, x)

        final_state, sequences = jax.lax.scan(
            iter_fn, initial_state, x, unroll=self.unroll
        )
        if self.stateful and not self.initializing():
            self.last_state = final_state

        # Note: Not needed with jax_utils.scan_in_dim
        sequences = jnp.swapaxes(sequences, self.time_axis, 0)

        if self.return_sequences and not self.return_state:
            return sequences
        if self.return_sequences and self.return_state:
            return sequences, final_state
        return final_state
