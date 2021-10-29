import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import stochastic as flax_module

from treex import types
from treex.key_seq import KeySeq
from treex.module import Module


class Dropout(Module):
    """Create a dropout layer.

    `Dropout` is implemented as a wrapper over `flax.linen.Dropout`, its constructor
    arguments accept almost the same arguments including any Flax artifacts such as initializers.
    Main differences:

    * `deterministic` is not a constructor argument, but remains a `__call__` argument.
    * `self.training` state is used to indicate how Dropout should behave, interally
    `deterministic = not self.training or self.frozen` is used unless `deterministic` is explicitly
    passed via `__call__`.
    * Dropout maintains an `rng: Rng` state which is used to generate random masks unless `rng` is passed
    via `__call__`.
    """

    # pytree
    next_key: KeySeq

    # static
    rate: float
    broadcast_dims: tp.Iterable[int]

    def __init__(
        self,
        rate: float,
        broadcast_dims: tp.Iterable[int] = (),
    ):
        """
        Create a dropout layer.

        Arguments:
            rate: the dropout probability.  (_not_ the keep rate!)
            broadcast_dims: dimensions that will share the same dropout mask
        """

        self.rate = rate
        self.broadcast_dims = broadcast_dims
        self.next_key = KeySeq()

    @property
    def module(self) -> flax_module.Dropout:
        return flax_module.Dropout(
            rate=self.rate,
            broadcast_dims=self.broadcast_dims,
            deterministic=None,
        )

    def __call__(
        self, x: jnp.ndarray, deterministic: tp.Optional[bool] = None, rng=None
    ) -> jnp.ndarray:
        """Applies a random dropout mask to the input.

        Arguments:
            x: the inputs that should be randomly masked.
            deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
                masked, whereas if true, no mask is applied and the inputs are returned
                as is.
            rng: an optional `jax.random.PRNGKey`. By default `self.rng` will
                be used.

        Returns:
            The masked inputs reweighted to preserve mean.
        """
        variables = dict()

        training = (
            not deterministic
            if deterministic is not None
            else self.training and not self.frozen
        )

        if rng is None:
            rng = self.next_key() if training else self.next_key.key

        # call apply
        output = self.module.apply(
            variables,
            x,
            deterministic=not training,
            rng=rng,
        )

        return tp.cast(jnp.ndarray, output)
